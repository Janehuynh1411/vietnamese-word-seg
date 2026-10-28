"""Build HITL-DPO preference pairs for Vietnamese word segmentation."""

import argparse
import json
import os
from typing import Dict, List, Optional

from data.dataset_converter import segmented_words_to_ws_tags, ws_tags_to_segmented_words

LABELS_DPO_XML_TEMPLATE = """<View>
  <Header value="Vietnamese Word Segmentation — Pairwise Preference (HITL)"/>
  <Text name="raw_text" value="$raw_text"/>
  <Header value="Candidate A"/>
  <Text name="candidate_a" value="$candidate_a"/>
  <Header value="Candidate B"/>
  <Text name="candidate_b" value="$candidate_b"/>
  <Choices name="preference" toName="raw_text" choice="single" required="true">
    <Choice value="A"/>
    <Choice value="B"/>
  </Choices>
</View>
"""


def _segmented_to_tags_matching_syllables(segmented_text: str, target_syllables: List[str]) -> Optional[List[str]]:
    words = segmented_text.strip().split()
    if not words:
        return None
    syllables, tags = segmented_words_to_ws_tags(words)
    if len(syllables) != len(target_syllables):
        return None
    if [s.lower() for s in syllables] != [s.lower() for s in target_syllables]:
        return None
    return tags


def generate_candidate_segmentations(
    raw_text: str,
    syllables: List[str],
    hmm_segmenter=None,
    phobert_segmenter=None,
) -> Dict[str, str]:
    """
    Candidate segmentations used as rejected (or alternate) sides of a preference pair.
    Sources that are not installed are skipped.
    """
    candidates: Dict[str, str] = {}
    if hmm_segmenter is not None:
        try:
            segmented = hmm_segmenter.segment(raw_text)
            if _segmented_to_tags_matching_syllables(segmented, syllables):
                candidates["hmm"] = segmented
        except Exception:
            pass

    try:
        from pyvi import ViTokenizer

        segmented = ViTokenizer.tokenize(raw_text.strip())
        if _segmented_to_tags_matching_syllables(segmented, syllables):
            candidates["pyvi"] = segmented
    except Exception:
        pass

    try:
        from underthesea import word_tokenize

        tokens = ["_".join(word.strip().split()) for word in word_tokenize(raw_text.strip()) if word.strip()]
        segmented = " ".join(tokens)
        if _segmented_to_tags_matching_syllables(segmented, syllables):
            candidates["underthesea"] = segmented
    except Exception:
        pass

    if phobert_segmenter is not None:
        try:
            tags = phobert_segmenter.predict_tags(syllables)
            if len(tags) == len(syllables):
                candidates["phobert_sft"] = " ".join(ws_tags_to_segmented_words(syllables, tags))
        except Exception:
            pass

    candidates["under_segmented"] = " ".join(syllables)
    return candidates


def parse_hitl_pairwise_export(export_path: str) -> List[dict]:
    """
    Parse a Label Studio pairwise export where an annotator chose A or B
    between candidate_a and candidate_b.
    """
    if not export_path or not os.path.exists(export_path):
        return []
    with open(export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if isinstance(tasks, dict):
        tasks = [tasks]

    pairs: List[dict] = []
    for task in tasks:
        data = task.get("data") or {}
        raw_text = (data.get("raw_text") or data.get("text") or "").strip()
        candidate_a = (data.get("candidate_a") or "").strip()
        candidate_b = (data.get("candidate_b") or "").strip()
        syllables = raw_text.split()
        if not syllables or not candidate_a or not candidate_b:
            continue

        choice = None
        annotations = task.get("annotations") or task.get("completions") or []
        for annotation in annotations:
            for result in annotation.get("result") or []:
                value = result.get("value") or {}
                choices = value.get("choices") or []
                if choices:
                    choice = str(choices[0]).strip().upper()
        if choice not in ("A", "B"):
            continue

        chosen_text = candidate_a if choice == "A" else candidate_b
        rejected_text = candidate_b if choice == "A" else candidate_a
        chosen_tags = _segmented_to_tags_matching_syllables(chosen_text, syllables)
        rejected_tags = _segmented_to_tags_matching_syllables(rejected_text, syllables)
        if not chosen_tags or not rejected_tags or chosen_tags == rejected_tags:
            continue
        pairs.append(
            {
                "raw_text": raw_text,
                "syllables": syllables,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "chosen_tags": chosen_tags,
                "rejected_tags": rejected_tags,
                "source": "hitl",
            }
        )
    return pairs


def build_dpo_dataset(
    ws_splits_path: str = "data/processed/ws_splits.json",
    hmm_weights_path: str = "models/hmm_weights.json",
    phobert_ckpt_path: Optional[str] = None,
    hitl_pairwise_path: Optional[str] = None,
    output_dir: str = "data/processed",
) -> Dict[str, object]:
    """
    Write dpo_preferences.json, dpo_preferences.jsonl, and a Label Studio config.

    Gold train/val segmentations are the chosen side. Disagreeing model candidates
    are the rejected side. Optional HITL A/B exports are appended.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(ws_splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    hmm_segmenter = None
    if hmm_weights_path and os.path.exists(hmm_weights_path):
        from hmm.viterbi import ViterbiSegmenter

        hmm_segmenter = ViterbiSegmenter(hmm_weights_path)

    phobert_segmenter = None
    if phobert_ckpt_path and os.path.exists(phobert_ckpt_path):
        try:
            from llm.phobert import PhoBERTWordSegmenter

            phobert_segmenter = PhoBERTWordSegmenter(model_name_or_path=phobert_ckpt_path)
        except Exception:
            phobert_segmenter = None

    pairs: List[dict] = []
    seen = set()
    for split_name in ("train", "val"):
        for record in splits.get(split_name) or []:
            syllables = record.get("syllables") or []
            gold_tags = record.get("tags") or record.get("ws_tags") or []
            raw_text = record.get("raw_text") or " ".join(syllables)
            chosen = record.get("segmented_text") or " ".join(ws_tags_to_segmented_words(syllables, gold_tags))
            if not syllables or len(gold_tags) != len(syllables):
                continue
            candidates = generate_candidate_segmentations(
                raw_text, syllables, hmm_segmenter=hmm_segmenter, phobert_segmenter=phobert_segmenter
            )
            for source, segmented in candidates.items():
                rejected_tags = _segmented_to_tags_matching_syllables(segmented, syllables)
                if not rejected_tags or rejected_tags == gold_tags:
                    continue
                key = (chosen, segmented)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "raw_text": raw_text,
                        "syllables": syllables,
                        "chosen": chosen,
                        "rejected": segmented,
                        "chosen_tags": gold_tags,
                        "rejected_tags": rejected_tags,
                        "source": source,
                        "split": split_name,
                    }
                )

    if hitl_pairwise_path:
        pairs.extend(parse_hitl_pairwise_export(hitl_pairwise_path))

    json_path = os.path.join(output_dir, "dpo_preferences.json")
    jsonl_path = os.path.join(output_dir, "dpo_preferences.jsonl")
    xml_path = os.path.join(output_dir, "label_studio_dpo_config.xml")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"pairs": pairs}, f, ensure_ascii=False, indent=2)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(LABELS_DPO_XML_TEMPLATE)

    summary = {
        "pairs": len(pairs),
        "json": json_path,
        "jsonl": jsonl_path,
        "label_studio_config": xml_path,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build HITL-DPO preference triples for Vietnamese Word Segmentation")
    parser.add_argument("--ws-splits", default="data/processed/ws_splits.json")
    parser.add_argument("--hmm-weights", default="models/hmm_weights.json")
    parser.add_argument("--phobert-ckpt", default="models/phobert_ws")
    parser.add_argument("--hitl-export", default=None, help="Optional Label Studio pairwise A/B export JSON")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    summary = build_dpo_dataset(
        ws_splits_path=args.ws_splits,
        hmm_weights_path=args.hmm_weights,
        phobert_ckpt_path=args.phobert_ckpt,
        hitl_pairwise_path=args.hitl_export,
        output_dir=args.output_dir,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
