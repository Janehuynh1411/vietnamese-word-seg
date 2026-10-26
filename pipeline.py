import csv
import json
import os
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None

from data.dataset_converter import normalize_pos_tag, prepare_dataset, ws_tags_to_segmented_words
from eval.metrics import evaluate_pos_tagging, evaluate_word_segmentation
from hmm.train import train_hmm
from hmm.viterbi import ViterbiSegmenter
from llm.client import LLMWordSegmenterClient
from pos.tagger import HMMPosTagger, read_corpus


DEFAULT_CONFIG = {
    "hmm": {
        "smoothing_factor": 0.1,
        "data_split": 0.8,
        "ws_weights": "models/hmm_weights.json",
        "pos_weights": "models/pos_hmm.json",
    },
    "phobert": {
        "model_name": "vinai/phobert-base",
        "max_length": 256,
        "batch_size": 16,
        "learning_rate": 2.0e-5,
        "epochs": 5,
        "weight_decay": 0.01,
        "seed": 42,
        "ws_checkpoint": "models/phobert_ws",
        "pos_checkpoint": "models/phobert_pos",
    },
    "llm": {
        "primary_model": "groq/llama-3.1-70b-versatile",
        "fallback_model": "gemini/gemini-1.5-flash",
        "params": {
            "temperature": 0.0,
            "max_tokens": 512,
        },
        "prompting": {
            "strategy": "few-shot",
        },
    },
    "routing": {
        "hmm_confidence_threshold": 0.85,
        "phobert_confidence_threshold": 0.70,
    },
    "paths": {
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "ws_splits": "data/processed/ws_splits.json",
        "pos_splits": "data/processed/pos_splits.json",
        "checkpoints": "models/hmm_weights.json",
        "logs": "logs/experiment_results.csv",
    },
}


def load_config(config_path: str = "config.yaml") -> dict:
    if yaml is not None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return dict(DEFAULT_CONFIG)


class VietnameseNLPPipeline:
    """
    Two-stage Vietnamese NLP Pipeline with Asymmetric Confidence-Based Routing:
      Stage 1: Word Segmentation ('hmm', 'phobert', 'hybrid', 'cascade', 'pyvi', 'underthesea', 'llm')
      Stage 2: POS Tagging ('hmm', 'phobert', 'pyvi', 'underthesea', 'llm')
    """

    def __init__(
        self,
        ws_mode: str = "phobert",
        pos_mode: str = "phobert",
        config_path: str = "config.yaml",
        device: Optional[str] = None,
        hmm_threshold: float = 0.85,
        phobert_threshold: float = 0.70,
    ):
        self.config = load_config(config_path)
        self.ws_mode = ws_mode
        self.pos_mode = pos_mode
        self.device = device
        self.hmm_threshold = hmm_threshold
        self.phobert_threshold = phobert_threshold

        self.hmm_ws: Optional[ViterbiSegmenter] = None
        self.phobert_ws = None
        self.hmm_pos: Optional[HMMPosTagger] = None
        self.phobert_pos = None
        self.llm_client: Optional[LLMWordSegmenterClient] = None

        self.routing_stats = {
            "total_syllables": 0,
            "tier1_hmm_syllables": 0,
            "tier2_phobert_syllables": 0,
            "tier3_llm_syllables": 0,
            "sentences_processed": 0,
        }

        self._init_models()

    def _init_models(self) -> None:
        hmm_cfg = self.config.get("hmm", {})
        phobert_cfg = self.config.get("phobert", {})
        llm_cfg = self.config.get("llm", {})

        ws_weights = hmm_cfg.get("ws_weights", "models/hmm_weights.json")
        pos_weights = hmm_cfg.get("pos_weights", "models/pos_hmm.json")
        phobert_base = phobert_cfg.get("model_name", "vinai/phobert-base")
        phobert_ws_ckpt = phobert_cfg.get("ws_checkpoint", "models/phobert_ws")
        phobert_pos_ckpt = phobert_cfg.get("pos_checkpoint", "models/phobert_pos")
        max_len = int(phobert_cfg.get("max_length", 256))

        if self.ws_mode in ("hmm", "hybrid", "cascade"):
            if os.path.exists(ws_weights):
                self.hmm_ws = ViterbiSegmenter(weights_path=ws_weights)

        if self.ws_mode in ("phobert", "hybrid", "cascade"):
            from llm.phobert import PhoBERTWordSegmenter

            ws_model_path = phobert_ws_ckpt if os.path.exists(phobert_ws_ckpt) else phobert_base
            self.phobert_ws = PhoBERTWordSegmenter(
                model_name_or_path=ws_model_path,
                max_length=max_len,
                device=self.device,
            )

        if self.ws_mode in ("llm", "cascade") or self.pos_mode == "llm":
            self.llm_client = LLMWordSegmenterClient(
                primary_model=llm_cfg.get("primary_model", "groq/llama-3.1-70b-versatile"),
                fallback_model=llm_cfg.get("fallback_model", "gemini/gemini-1.5-flash"),
                strategy=llm_cfg.get("prompting", {}).get("strategy", "few-shot"),
                temperature=float(llm_cfg.get("params", {}).get("temperature", 0.0)),
                max_tokens=int(llm_cfg.get("params", {}).get("max_tokens", 512)),
            )

        if self.pos_mode == "hmm":
            if os.path.exists(pos_weights):
                self.hmm_pos = HMMPosTagger.load(pos_weights)

        elif self.pos_mode == "phobert":
            from llm.phobert import PhoBERTPosTagger

            pos_model_path = phobert_pos_ckpt if os.path.exists(phobert_pos_ckpt) else phobert_base
            self.phobert_pos = PhoBERTPosTagger(
                model_name_or_path=pos_model_path,
                max_length=max_len,
                device=self.device,
            )

    def segment(self, text: str) -> str:
        syllables = text.strip().split()
        if not syllables:
            return ""

        self.routing_stats["sentences_processed"] += 1
        self.routing_stats["total_syllables"] += len(syllables)

        if self.ws_mode == "pyvi":
            try:
                from pyvi import ViTokenizer

                return ViTokenizer.tokenize(text.strip())
            except Exception:
                pass

        if self.ws_mode == "underthesea":
            try:
                from underthesea import word_tokenize

                tokens = ["_".join(w.strip().split()) for w in word_tokenize(text.strip()) if w.strip()]
                return " ".join(tokens)
            except Exception:
                pass

        if self.ws_mode == "llm" and self.llm_client is not None:
            self.routing_stats["tier3_llm_syllables"] += len(syllables)
            return self.llm_client.segment(text.strip())

        if self.ws_mode == "hmm" and self.hmm_ws is not None:
            self.routing_stats["tier1_hmm_syllables"] += len(syllables)
            return self.hmm_ws.segment(text)

        if self.ws_mode == "phobert" and self.phobert_ws is not None:
            self.routing_stats["tier2_phobert_syllables"] += len(syllables)
            return self.phobert_ws.segment(text)

        if self.ws_mode in ("hybrid", "cascade"):
            if self.hmm_ws is not None and self.phobert_ws is not None:
                _, hmm_tags, hmm_confs = self.hmm_ws.segment_with_confidence(text)
                # Identify low-confidence or OOV syllable positions (plus their immediate neighbors)
                uncertain_indices: Set[int] = set()
                for idx, conf in enumerate(hmm_confs):
                    if conf < self.hmm_threshold:
                        uncertain_indices.add(idx)
                        if idx > 0:
                            uncertain_indices.add(idx - 1)
                        if idx + 1 < len(syllables):
                            uncertain_indices.add(idx + 1)

                if not uncertain_indices:
                    self.routing_stats["tier1_hmm_syllables"] += len(syllables)
                    return " ".join(ws_tags_to_segmented_words(syllables, hmm_tags))

                pb_tags, pb_confs = self.phobert_ws.predict_tags_with_confidence(syllables)
                llm_tags: Optional[List[str]] = None

                # Tier 3 escalation in 'cascade' mode when PhoBERT confidence is also below threshold
                very_uncertain = [idx for idx in uncertain_indices if pb_confs[idx] < self.phobert_threshold]
                if self.ws_mode == "cascade" and very_uncertain and self.llm_client is not None:
                    llm_seg = self.llm_client.segment(text.strip(), ambiguous_indices=sorted(very_uncertain))
                    cand_tags = [
                        "B" if sub_i == 0 else "I"
                        for w in llm_seg.split()
                        for sub_i, _ in enumerate(w.split("_"))
                    ]
                    if len(cand_tags) == len(syllables):
                        llm_tags = cand_tags

                merged_tags: List[str] = []
                for idx in range(len(syllables)):
                    if idx == 0:
                        merged_tags.append("B")
                        self.routing_stats["tier1_hmm_syllables"] += 1
                    elif idx in very_uncertain and llm_tags is not None:
                        merged_tags.append(llm_tags[idx])
                        self.routing_stats["tier3_llm_syllables"] += 1
                    elif idx in uncertain_indices:
                        merged_tags.append(pb_tags[idx])
                        self.routing_stats["tier2_phobert_syllables"] += 1
                    else:
                        merged_tags.append(hmm_tags[idx])
                        self.routing_stats["tier1_hmm_syllables"] += 1

                return " ".join(ws_tags_to_segmented_words(syllables, merged_tags))
            elif self.phobert_ws is not None:
                return self.phobert_ws.segment(text)
            elif self.hmm_ws is not None:
                return self.hmm_ws.segment(text)

        return text.strip()

    def tag_segmented(self, segmented_text: str) -> List[Tuple[str, str]]:
        words = segmented_text.strip().split()
        if not words:
            return []

        if self.pos_mode == "pyvi":
            try:
                from pyvi import ViPosTagger

                _, tags = ViPosTagger.postagging(" ".join(words))
                if len(tags) == len(words):
                    return [(w, normalize_pos_tag(t)) for w, t in zip(words, tags)]
            except Exception:
                pass

        if self.pos_mode == "underthesea":
            try:
                from underthesea import pos_tag

                raw = " ".join(w.replace("_", " ") for w in words)
                tagged = pos_tag(raw)
                ut_words = ["_".join(w.strip().split()) for w, _ in tagged if w.strip()]
                ut_tags = [normalize_pos_tag(t) for w, t in tagged if w.strip()]
                if len(ut_words) == len(words):
                    return list(zip(words, ut_tags))
                # Map by word lookup if segmentation differs slightly
                tag_lookup = {uw.lower(): ut for uw, ut in zip(ut_words, ut_tags)}
                return [(w, tag_lookup.get(w.lower(), "CH" if all(not c.isalnum() for c in w) else "N")) for w in words]
            except Exception:
                pass

        if self.pos_mode == "llm" and self.llm_client is not None:
            return self.llm_client.tag_segmented(segmented_text)

        if self.pos_mode == "hmm" and self.hmm_pos is not None:
            return [(w, normalize_pos_tag(t)) for w, t in self.hmm_pos.tag(words)]

        if self.pos_mode == "phobert" and self.phobert_pos is not None:
            return [(w, normalize_pos_tag(t)) for w, t in self.phobert_pos.tag(words)]

        return [(w, "CH" if all(not c.isalnum() for c in w) else "N") for w in words]

    def process(self, text: str) -> Dict[str, object]:
        segmented = self.segment(text)
        pos_tagged = self.tag_segmented(segmented)
        return {
            "raw_text": text,
            "segmented_text": segmented,
            "pos_tags": pos_tagged,
            "formatted_pos": " ".join(f"{w}/{t}" for w, t in pos_tagged),
            "routing_stats": dict(self.routing_stats),
        }


def train_hmm_models(config: dict) -> Dict[str, str]:
    paths_cfg = config.get("paths", {})
    hmm_cfg = config.get("hmm", {})
    processed_dir = paths_cfg.get("processed_data", "data/processed")

    train_txt = os.path.join(processed_dir, "train.txt")
    pos_train_txt = os.path.join(processed_dir, "pos_train.txt")

    if not os.path.exists(train_txt) or not os.path.exists(pos_train_txt):
        prepare_dataset(
            raw_dir=paths_cfg.get("raw_data", "data/raw"),
            output_dir=processed_dir,
        )

    ws_weights = hmm_cfg.get("ws_weights", "models/hmm_weights.json")
    pos_weights = hmm_cfg.get("pos_weights", "models/pos_hmm.json")
    os.makedirs(os.path.dirname(ws_weights) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(pos_weights) or ".", exist_ok=True)

    ws_model = train_hmm(train_txt)
    with open(ws_weights, "w", encoding="utf-8") as f:
        json.dump(ws_model, f, ensure_ascii=False, indent=4)

    pos_sentences = read_corpus(pos_train_txt)
    pos_tagger = HMMPosTagger(smoothing=float(hmm_cfg.get("smoothing_factor", 0.1))).train(pos_sentences)
    pos_tagger.save(pos_weights)

    return {"hmm_ws": ws_weights, "hmm_pos": pos_weights}


def train_phobert_models(
    config: dict,
    tasks: Tuple[str, ...] = ("ws", "pos"),
    epochs_override: Optional[int] = None,
) -> Dict[str, str]:
    from llm.phobert import PhoBERTPosTagger, PhoBERTWordSegmenter

    paths_cfg = config.get("paths", {})
    phobert_cfg = config.get("phobert", {})
    processed_dir = paths_cfg.get("processed_data", "data/processed")

    ws_splits_path = paths_cfg.get("ws_splits", os.path.join(processed_dir, "ws_splits.json"))
    pos_splits_path = paths_cfg.get("pos_splits", os.path.join(processed_dir, "pos_splits.json"))

    if not os.path.exists(ws_splits_path) or not os.path.exists(pos_splits_path):
        prepare_dataset(
            raw_dir=paths_cfg.get("raw_data", "data/raw"),
            output_dir=processed_dir,
        )

    model_name = phobert_cfg.get("model_name", "vinai/phobert-base")
    max_length = int(phobert_cfg.get("max_length", 256))
    batch_size = int(phobert_cfg.get("batch_size", 16))
    lr = float(phobert_cfg.get("learning_rate", 2e-5))
    weight_decay = float(phobert_cfg.get("weight_decay", 0.01))
    epochs = int(epochs_override if epochs_override is not None else phobert_cfg.get("epochs", 5))
    ws_ckpt = phobert_cfg.get("ws_checkpoint", "models/phobert_ws")
    pos_ckpt = phobert_cfg.get("pos_checkpoint", "models/phobert_pos")

    saved: Dict[str, str] = {}

    if "ws" in tasks:
        with open(ws_splits_path, "r", encoding="utf-8") as f:
            ws_splits = json.load(f)
        ws_clf = PhoBERTWordSegmenter(model_name_or_path=model_name, max_length=max_length)
        ws_clf.fit(
            train_samples=ws_splits["train"],
            val_samples=ws_splits.get("val"),
            token_key="syllables",
            label_key="tags",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            save_dir=ws_ckpt,
        )
        saved["phobert_ws"] = ws_ckpt

    if "pos" in tasks:
        with open(pos_splits_path, "r", encoding="utf-8") as f:
            pos_splits = json.load(f)
        all_samples = pos_splits["train"] + pos_splits.get("val", []) + pos_splits.get("test", [])
        label2id = PhoBERTPosTagger.build_label_map_from_samples(all_samples, label_key="tags")
        pos_clf = PhoBERTPosTagger(model_name_or_path=model_name, label2id=label2id, max_length=max_length)
        pos_clf.fit(
            train_samples=pos_splits["train"],
            val_samples=pos_splits.get("val"),
            token_key="words",
            label_key="tags",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            save_dir=pos_ckpt,
        )
        saved["phobert_pos"] = pos_ckpt

    return saved


def _evaluate_single_pipeline(
    name: str,
    pipe: VietnameseNLPPipeline,
    raw_test_sentences: List[str],
    gold_ws_sentences: List[str],
    gold_pos_sentences: List[List[Tuple[str, str]]],
    train_vocab: Set[str],
) -> Dict[str, object]:
    ws_preds = [pipe.segment(raw) for raw in raw_test_sentences]
    ws_metrics = evaluate_word_segmentation(gold_ws_sentences, ws_preds, train_vocab=train_vocab)

    pos_gold_seg = [pipe.tag_segmented(seg) for seg in gold_ws_sentences]
    pos_e2e = [pipe.tag_segmented(seg) for seg in ws_preds]
    pos_metrics = evaluate_pos_tagging(gold_pos_sentences, pos_gold_seg, train_vocab=train_vocab)
    e2e_metrics = evaluate_pos_tagging(gold_pos_sentences, pos_e2e, train_vocab=train_vocab)

    return {
        "model": name,
        "ws_precision": round(ws_metrics["precision"], 4),
        "ws_recall": round(ws_metrics["recall"], 4),
        "ws_f1": round(ws_metrics["f1_score"], 4),
        "ws_oov_recall": round(ws_metrics["oov_recall"], 4),
        "ws_bi_accuracy": round(ws_metrics["bi_accuracy"], 4),
        "pos_accuracy": round(pos_metrics["accuracy"], 4),
        "pos_macro_f1": round(pos_metrics["macro_f1"], 4),
        "pos_oov_accuracy": round(pos_metrics["oov_accuracy"], 4),
        "e2e_joint_f1": round(e2e_metrics["joint_f1"], 4),
    }


def run_full_evaluation(
    config: dict,
    include_phobert: bool = True,
) -> List[Dict[str, object]]:
    paths_cfg = config.get("paths", {})
    processed_dir = paths_cfg.get("processed_data", "data/processed")
    ws_splits_path = paths_cfg.get("ws_splits", os.path.join(processed_dir, "ws_splits.json"))
    pos_splits_path = paths_cfg.get("pos_splits", os.path.join(processed_dir, "pos_splits.json"))

    if not os.path.exists(ws_splits_path) or not os.path.exists(pos_splits_path):
        prepare_dataset(
            raw_dir=paths_cfg.get("raw_data", "data/raw"),
            output_dir=processed_dir,
        )

    with open(ws_splits_path, "r", encoding="utf-8") as f:
        ws_splits = json.load(f)
    with open(pos_splits_path, "r", encoding="utf-8") as f:
        pos_splits = json.load(f)

    train_vocab: Set[str] = {
        w.lower()
        for item in pos_splits["train"]
        for w in item.get("words", [])
    }

    test_ws = ws_splits["test"]
    test_pos = pos_splits["test"]

    gold_ws_sentences = [item["segmented_text"] for item in test_ws]
    raw_test_sentences = [item["raw_text"] for item in test_ws]
    gold_pos_sentences = [list(zip(item["words"], item["tags"])) for item in test_pos]

    results: List[Dict[str, object]] = []

    hmm_cfg = config.get("hmm", {})
    if not os.path.exists(hmm_cfg.get("ws_weights", "models/hmm_weights.json")):
        train_hmm_models(config)

    # 1. HMM Baseline
    hmm_pipe = VietnameseNLPPipeline(ws_mode="hmm", pos_mode="hmm")
    results.append(
        _evaluate_single_pipeline(
            "hmm",
            hmm_pipe,
            raw_test_sentences,
            gold_ws_sentences,
            gold_pos_sentences,
            train_vocab,
        )
    )

    # 2. PyVi Baseline (from config.yaml baseline_comparison: "pyvi")
    try:
        import pyvi  # noqa: F401

        pyvi_pipe = VietnameseNLPPipeline(ws_mode="pyvi", pos_mode="pyvi")
        results.append(
            _evaluate_single_pipeline(
                "pyvi_baseline",
                pyvi_pipe,
                raw_test_sentences,
                gold_ws_sentences,
                gold_pos_sentences,
                train_vocab,
            )
        )
    except ImportError:
        pass

    # 3. Underthesea Baseline
    try:
        import underthesea  # noqa: F401

        ut_pipe = VietnameseNLPPipeline(ws_mode="underthesea", pos_mode="underthesea")
        results.append(
            _evaluate_single_pipeline(
                "underthesea_baseline",
                ut_pipe,
                raw_test_sentences,
                gold_ws_sentences,
                gold_pos_sentences,
                train_vocab,
            )
        )
    except ImportError:
        pass

    if include_phobert:
        # 4. PhoBERT (vinai/phobert-base)
        phobert_pipe = VietnameseNLPPipeline(ws_mode="phobert", pos_mode="phobert")
        results.append(
            _evaluate_single_pipeline(
                "phobert",
                phobert_pipe,
                raw_test_sentences,
                gold_ws_sentences,
                gold_pos_sentences,
                train_vocab,
            )
        )

        # 5. Hybrid (HMM + PhoBERT Confidence Routing)
        hybrid_pipe = VietnameseNLPPipeline(ws_mode="hybrid", pos_mode="phobert")
        results.append(
            _evaluate_single_pipeline(
                "hybrid_hmm_phobert",
                hybrid_pipe,
                raw_test_sentences,
                gold_ws_sentences,
                gold_pos_sentences,
                train_vocab,
            )
        )

        # 6. 3-Tier Asymmetric Cascade (HMM -> PhoBERT -> LLM/Dict Fallback)
        cascade_pipe = VietnameseNLPPipeline(ws_mode="cascade", pos_mode="phobert")
        results.append(
            _evaluate_single_pipeline(
                "cascade_3tier",
                cascade_pipe,
                raw_test_sentences,
                gold_ws_sentences,
                gold_pos_sentences,
                train_vocab,
            )
        )

    log_path = paths_cfg.get("logs", "logs/experiment_results.csv")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    return results
