import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

from data.dataset_converter import segmented_words_to_ws_tags, ws_tags_to_segmented_words
from hmm.viterbi import ViterbiSegmenter


LABELS_DPO_XML_TEMPLATE = """<View>
  <Header value="Vietnamese Word Segmentation — Pairwise Preference (HITL-DPO)" />
  <Text name="raw_text" value="$raw_text" style="font-size: 16px; font-weight: bold;" />
  <View style="display: flex; gap: 20px; margin-top: 12px;">
    <View style="flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 6px;">
      <Header value="Candidate A ($model_a)" size="4" />
      <Text name="cand_a" value="$candidate_a" />
    </View>
    <View style="flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 6px;">
      <Header value="Candidate B ($model_b)" size="4" />
      <Text name="cand_b" value="$candidate_b" />
    </View>
  </View>
  <Choices name="preference" toName="raw_text" choice="single-radio" showInline="true">
    <Choice value="A" alias="A">Prefer Candidate A</Choice>
    <Choice value="B" alias="B">Prefer Candidate B</Choice>
    <Choice value="tie" alias="tie">Both Equally Good / Tie</Choice>
  </Choices>
</View>
"""


def _segmented_to_tags_matching_syllables(segmented_text: str, target_syllables: List[str]) -> Optional[List[str]]:
    """Convert an underscore-segmented string to B/I tags if its syllable count matches target_syllables."""
    words = segmented_text.strip().split()
    _, tags = segmented_words_to_ws_tags(words)
    if len(tags) != len(target_syllables):
        return None
    return tags


def generate_candidate_segmentations(
    raw_text: str,
    syllables: List[str],
    hmm_segmenter: Optional[ViterbiSegmenter] = None,
    phobert_segmenter=None,
) -> Dict[str, Tuple[str, List[str]]]:
    """
    Generate candidate segmentations from multiple model families (HMM, PyVi, Underthesea, PhoBERT, Heuristic)
    to surface diverse negative/disagreement examples for DPO.
    """
    candidates: Dict[str, Tuple[str, List[str]]] = {}

    # 1. HMM Viterbi candidate
    if hmm_segmenter is not None:
        hmm_seg = hmm_segmenter.segment(raw_text)
        hmm_tags = _segmented_to_tags_matching_syllables(hmm_seg, syllables)
        if hmm_tags is not None:
            candidates["hmm"] = (hmm_seg, hmm_tags)

    # 2. PyVi candidate
    try:
        from pyvi import ViTokenizer

        pv_seg = ViTokenizer.tokenize(raw_text)
        pv_tags = _segmented_to_tags_matching_syllables(pv_seg, syllables)
        if pv_tags is not None:
            candidates["pyvi"] = (pv_seg, pv_tags)
    except Exception:
        pass

    # 3. Underthesea candidate
    try:
        from underthesea import word_tokenize

        ut_words = ["_".join(w.strip().split()) for w in word_tokenize(raw_text) if w.strip()]
        ut_seg = " ".join(ut_words)
        ut_tags = _segmented_to_tags_matching_syllables(ut_seg, syllables)
        if ut_tags is not None:
            candidates["underthesea"] = (ut_seg, ut_tags)
    except Exception:
        pass

    # 4. PhoBERT candidate (if loaded)
    if phobert_segmenter is not None:
        pb_tags = phobert_segmenter.predict_tags(syllables)
        if len(pb_tags) == len(syllables):
            pb_seg = " ".join(ws_tags_to_segmented_words(syllables, pb_tags))
            candidates["phobert_sft"] = (pb_seg, pb_tags)

    # 5. Unsegmented (all-B under-segmentation baseline) & greedy pair-merge Perturbation
    all_b_tags = ["B"] * len(syllables)
    candidates["under_segmented"] = (" ".join(syllables), all_b_tags)

    return candidates


def parse_hitl_pairwise_export(export_path: str) -> List[dict]:
    """
    Parse a HumanSignal / Label Studio pairwise A/B export where a human selected
    'A' or 'B' between candidate_a and candidate_b.
    """
    if not os.path.exists(export_path):
        return []
    with open(export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if isinstance(tasks, dict):
        tasks = [tasks]

    dpo_pairs: List[dict] = []
    for task in tasks:
        data = task.get("data", task)
        raw_text = data.get("raw_text") or data.get("text", "")
        cand_a = data.get("candidate_a", "")
        cand_b = data.get("candidate_b", "")
        if not raw_text or not cand_a or not cand_b:
            continue

        choice = data.get("preference")
        if not choice:
            for ann in task.get("annotations", []):
                for res in ann.get("result", []):
                    choices = res.get("value", {}).get("choices", [])
                    if choices:
                        choice = choices[0]
                        break

        if choice == "A":
            chosen, rejected = cand_a, cand_b
        elif choice == "B":
            chosen, rejected = cand_b, cand_a
        else:
            continue

        if chosen == rejected:
            continue

        syllables = raw_text.strip().split()
        chosen_tags = _segmented_to_tags_matching_syllables(chosen, syllables)
        rejected_tags = _segmented_to_tags_matching_syllables(rejected, syllables)
        if chosen_tags and rejected_tags:
            dpo_pairs.append(
                {
                    "prompt": f"Segment: {raw_text}",
                    "raw_text": raw_text,
                    "syllables": syllables,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_tags": chosen_tags,
                    "rejected_tags": rejected_tags,
                    "source": "human_pairwise_ab",
                }
            )
    return dpo_pairs


def build_dpo_dataset(
    ws_splits_path: str = "data/processed/ws_splits.json",
    hmm_weights_path: str = "models/hmm_weights.json",
    phobert_ckpt_path: Optional[str] = "models/phobert_ws",
    hitl_pairwise_path: Optional[str] = None,
    output_dir: str = "data/processed",
) -> Dict[str, object]:
    """
    Build a DPO preference dataset (dpo_preferences.json & dpo_preferences.jsonl) from:
      1. Gold human annotations in the TRAIN and VAL splits vs. imperfect model candidates
         (strictly excluding the test split to prevent any data leakage).
      2. Optional HumanSignal pairwise A/B exports.
      3. Also exports a HumanSignal pairwise labeling batch (`label_studio_dpo_pairs.json`)
         and XML config (`label_studio_dpo_config.xml`) for active human-in-the-loop review.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(ws_splits_path, "r", encoding="utf-8") as f:
        ws_splits = json.load(f)

    hmm_seg = ViterbiSegmenter(hmm_weights_path) if os.path.exists(hmm_weights_path) else None
    pb_seg = None
    if phobert_ckpt_path and os.path.exists(phobert_ckpt_path):
        try:
            from llm.phobert import PhoBERTWordSegmenter

            pb_seg = PhoBERTWordSegmenter(model_name_or_path=phobert_ckpt_path)
        except Exception:
            pb_seg = None

    splits_out: Dict[str, List[dict]] = {"train": [], "val": []}
    hitl_tasks_for_label_studio: List[dict] = []

    for split_name in ("train", "val"):
        for item in ws_splits.get(split_name, []):
            syllables = item["syllables"]
            gold_tags = item["tags"]
            gold_seg = item["segmented_text"]
            raw_text = item["raw_text"]

            candidates = generate_candidate_segmentations(
                raw_text=raw_text,
                syllables=syllables,
                hmm_segmenter=hmm_seg,
                phobert_segmenter=pb_seg,
            )

            seen_rejected = set()
            disagreement_models = []
            for model_name, (cand_seg, cand_tags) in candidates.items():
                if cand_tags != gold_tags and cand_seg not in seen_rejected:
                    seen_rejected.add(cand_seg)
                    disagreement_models.append((model_name, cand_seg))
                    splits_out[split_name].append(
                        {
                            "prompt": (
                                "Segment the following Vietnamese sentence into compound words connected by underscores ('_'):\n"
                                f"{raw_text}"
                            ),
                            "raw_text": raw_text,
                            "syllables": syllables,
                            "chosen": gold_seg,
                            "rejected": cand_seg,
                            "chosen_tags": gold_tags,
                            "rejected_tags": cand_tags,
                            "rejected_source": model_name,
                        }
                    )

            # Create a pairwise A/B review task for HumanSignal whenever models disagree
            if disagreement_models:
                neg_model, neg_seg = disagreement_models[0]
                hitl_tasks_for_label_studio.append(
                    {
                        "data": {
                            "raw_text": raw_text,
                            "candidate_a": gold_seg,
                            "candidate_b": neg_seg,
                            "model_a": "phobert_or_gold",
                            "model_b": neg_model,
                        }
                    }
                )

    if hitl_pairwise_path:
        extra_pairs = parse_hitl_pairwise_export(hitl_pairwise_path)
        splits_out["train"].extend(extra_pairs)

    dpo_json_path = os.path.join(output_dir, "dpo_preferences.json")
    dpo_jsonl_path = os.path.join(output_dir, "dpo_preferences.jsonl")
    ls_pairs_path = os.path.join(output_dir, "label_studio_dpo_pairs.json")
    ls_xml_path = os.path.join(output_dir, "label_studio_dpo_config.xml")

    with open(dpo_json_path, "w", encoding="utf-8") as f:
        json.dump(splits_out, f, ensure_ascii=False, indent=2)

    with open(dpo_jsonl_path, "w", encoding="utf-8") as f:
        for row in splits_out["train"]:
            f.write(
                json.dumps(
                    {
                        "prompt": row["prompt"],
                        "chosen": row["chosen"],
                        "rejected": row["rejected"],
                        "rejected_source": row.get("rejected_source", "model"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(ls_pairs_path, "w", encoding="utf-8") as f:
        json.dump(hitl_tasks_for_label_studio, f, ensure_ascii=False, indent=2)

    with open(ls_xml_path, "w", encoding="utf-8") as f:
        f.write(LABELS_DPO_XML_TEMPLATE)

    return {
        "train_pairs": len(splits_out["train"]),
        "val_pairs": len(splits_out["val"]),
        "hitl_ab_tasks": len(hitl_tasks_for_label_studio),
        "dpo_json": dpo_json_path,
        "dpo_jsonl": dpo_jsonl_path,
        "label_studio_dpo_pairs": ls_pairs_path,
        "label_studio_dpo_xml": ls_xml_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Build HITL-DPO preference triples for Vietnamese Word Segmentation")
    parser.add_argument("--ws-splits", default="data/processed/ws_splits.json")
    parser.add_argument("--hmm-weights", default="models/hmm_weights.json")
    parser.add_argument("--phobert-ckpt", default="models/phobert_ws")
    parser.add_argument("--hitl-export", default=None, help="Optional HumanSignal pairwise A/B export JSON")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    summary = build_dpo_dataset(
        ws_splits_path=args.ws_splits,
        hmm_weights_path=args.hmm_weights,
        phobert_ckpt_path=args.phobert_ckpt,
        hitl_pairwise_path=args.hitl_export,
        output_dir=args.output_dir,
    )
    print(
        f"Built HITL-DPO dataset: {summary['train_pairs']} train preference pairs, "
        f"{summary['val_pairs']} val preference pairs"
    )
    print(f"  DPO JSONL:             {summary['dpo_jsonl']}")
    print(f"  HumanSignal A/B tasks: {summary['label_studio_dpo_pairs']} ({summary['hitl_ab_tasks']} tasks)")
    print(f"  HumanSignal XML UI:    {summary['label_studio_dpo_xml']}")


if __name__ == "__main__":
    main()
