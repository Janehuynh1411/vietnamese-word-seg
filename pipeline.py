import csv
import json
import os
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None

from data.dataset_converter import prepare_dataset, ws_tags_to_segmented_words
from eval.metrics import evaluate_pos_tagging, evaluate_word_segmentation
from hmm.train import train_hmm
from hmm.viterbi import ViterbiSegmenter
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
    Two-stage Vietnamese NLP Pipeline:
      Stage 1: Word Segmentation ('hmm', 'phobert', 'hybrid', or 'pyvi')
      Stage 2: POS Tagging ('hmm' or 'phobert')
    """

    def __init__(
        self,
        ws_mode: str = "phobert",
        pos_mode: str = "phobert",
        config_path: str = "config.yaml",
        device: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        self.ws_mode = ws_mode
        self.pos_mode = pos_mode
        self.device = device

        self.hmm_ws: Optional[ViterbiSegmenter] = None
        self.phobert_ws = None
        self.hmm_pos: Optional[HMMPosTagger] = None
        self.phobert_pos = None

        self._init_models()

    def _init_models(self) -> None:
        hmm_cfg = self.config.get("hmm", {})
        phobert_cfg = self.config.get("phobert", {})

        ws_weights = hmm_cfg.get("ws_weights", "models/hmm_weights.json")
        pos_weights = hmm_cfg.get("pos_weights", "models/pos_hmm.json")
        phobert_base = phobert_cfg.get("model_name", "vinai/phobert-base")
        phobert_ws_ckpt = phobert_cfg.get("ws_checkpoint", "models/phobert_ws")
        phobert_pos_ckpt = phobert_cfg.get("pos_checkpoint", "models/phobert_pos")
        max_len = int(phobert_cfg.get("max_length", 256))

        if self.ws_mode in ("hmm", "hybrid"):
            if os.path.exists(ws_weights):
                self.hmm_ws = ViterbiSegmenter(weights_path=ws_weights)

        if self.ws_mode in ("phobert", "hybrid"):
            from llm.phobert import PhoBERTWordSegmenter

            ws_model_path = phobert_ws_ckpt if os.path.exists(phobert_ws_ckpt) else phobert_base
            self.phobert_ws = PhoBERTWordSegmenter(
                model_name_or_path=ws_model_path,
                max_length=max_len,
                device=self.device,
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

        if self.ws_mode == "pyvi":
            try:
                from pyvi import ViTokenizer

                return ViTokenizer.tokenize(text.strip())
            except Exception:
                pass

        if self.ws_mode == "hmm" and self.hmm_ws is not None:
            return self.hmm_ws.segment(text)

        if self.ws_mode == "phobert" and self.phobert_ws is not None:
            return self.phobert_ws.segment(text)

        if self.ws_mode == "hybrid":
            if self.hmm_ws is not None and self.phobert_ws is not None:
                # Check if sentence has OOV syllables for HMM; if so, refine those positions with PhoBERT
                known_syllables = set(self.hmm_ws.emissions.get("B", {}).keys()) | set(
                    self.hmm_ws.emissions.get("I", {}).keys()
                )
                has_oov = any(s.lower() not in known_syllables for s in syllables)
                if not has_oov:
                    return self.hmm_ws.segment(text)
                hmm_seg = self.hmm_ws.segment(text)
                hmm_tags = [
                    "B" if idx == 0 else "I"
                    for w in hmm_seg.split()
                    for idx, _ in enumerate(w.split("_"))
                ]
                phobert_tags = self.phobert_ws.predict_tags(syllables)
                merged_tags = []
                for idx, syll in enumerate(syllables):
                    if idx == 0:
                        merged_tags.append("B")
                    elif syll.lower() not in known_syllables or (
                        idx > 0 and syllables[idx - 1].lower() not in known_syllables
                    ):
                        merged_tags.append(phobert_tags[idx])
                    else:
                        merged_tags.append(hmm_tags[idx] if idx < len(hmm_tags) else phobert_tags[idx])
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
        if self.pos_mode == "hmm" and self.hmm_pos is not None:
            return self.hmm_pos.tag(words)
        if self.pos_mode == "phobert" and self.phobert_pos is not None:
            return self.phobert_pos.tag(words)
        return [(w, "N") for w in words]

    def process(self, text: str) -> Dict[str, object]:
        segmented = self.segment(text)
        pos_tagged = self.tag_segmented(segmented)
        return {
            "raw_text": text,
            "segmented_text": segmented,
            "pos_tags": pos_tagged,
            "formatted_pos": " ".join(f"{w}/{t}" for w, t in pos_tagged),
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

    # Ensure HMM weights exist
    hmm_cfg = config.get("hmm", {})
    if not os.path.exists(hmm_cfg.get("ws_weights", "models/hmm_weights.json")):
        train_hmm_models(config)

    # 1. HMM Word Segmentation + HMM POS Tagging
    hmm_pipe = VietnameseNLPPipeline(ws_mode="hmm", pos_mode="hmm")
    hmm_ws_preds = [hmm_pipe.segment(raw) for raw in raw_test_sentences]
    hmm_ws_metrics = evaluate_word_segmentation(gold_ws_sentences, hmm_ws_preds, train_vocab=train_vocab)

    hmm_pos_gold_seg = [hmm_pipe.tag_segmented(seg) for seg in gold_ws_sentences]
    hmm_pos_e2e = [hmm_pipe.tag_segmented(seg) for seg in hmm_ws_preds]
    hmm_pos_metrics = evaluate_pos_tagging(gold_pos_sentences, hmm_pos_gold_seg, train_vocab=train_vocab)
    hmm_e2e_metrics = evaluate_pos_tagging(gold_pos_sentences, hmm_pos_e2e, train_vocab=train_vocab)

    results.append(
        {
            "model": "hmm",
            "ws_precision": round(hmm_ws_metrics["precision"], 4),
            "ws_recall": round(hmm_ws_metrics["recall"], 4),
            "ws_f1": round(hmm_ws_metrics["f1_score"], 4),
            "ws_oov_recall": round(hmm_ws_metrics["oov_recall"], 4),
            "ws_bi_accuracy": round(hmm_ws_metrics["bi_accuracy"], 4),
            "pos_accuracy": round(hmm_pos_metrics["accuracy"], 4),
            "pos_macro_f1": round(hmm_pos_metrics["macro_f1"], 4),
            "pos_oov_accuracy": round(hmm_pos_metrics["oov_accuracy"], 4),
            "e2e_joint_f1": round(hmm_e2e_metrics["joint_f1"], 4),
        }
    )

    if include_phobert:
        # 2. PhoBERT Word Segmentation + PhoBERT POS Tagging
        phobert_pipe = VietnameseNLPPipeline(ws_mode="phobert", pos_mode="phobert")
        pb_ws_preds = [phobert_pipe.segment(raw) for raw in raw_test_sentences]
        pb_ws_metrics = evaluate_word_segmentation(gold_ws_sentences, pb_ws_preds, train_vocab=train_vocab)

        pb_pos_gold_seg = [phobert_pipe.tag_segmented(seg) for seg in gold_ws_sentences]
        pb_pos_e2e = [phobert_pipe.tag_segmented(seg) for seg in pb_ws_preds]
        pb_pos_metrics = evaluate_pos_tagging(gold_pos_sentences, pb_pos_gold_seg, train_vocab=train_vocab)
        pb_e2e_metrics = evaluate_pos_tagging(gold_pos_sentences, pb_pos_e2e, train_vocab=train_vocab)

        results.append(
            {
                "model": "phobert",
                "ws_precision": round(pb_ws_metrics["precision"], 4),
                "ws_recall": round(pb_ws_metrics["recall"], 4),
                "ws_f1": round(pb_ws_metrics["f1_score"], 4),
                "ws_oov_recall": round(pb_ws_metrics["oov_recall"], 4),
                "ws_bi_accuracy": round(pb_ws_metrics["bi_accuracy"], 4),
                "pos_accuracy": round(pb_pos_metrics["accuracy"], 4),
                "pos_macro_f1": round(pb_pos_metrics["macro_f1"], 4),
                "pos_oov_accuracy": round(pb_pos_metrics["oov_accuracy"], 4),
                "e2e_joint_f1": round(pb_e2e_metrics["joint_f1"], 4),
            }
        )

        # 3. Hybrid (HMM + PhoBERT OOV refinement)
        hybrid_pipe = VietnameseNLPPipeline(ws_mode="hybrid", pos_mode="phobert")
        hyb_ws_preds = [hybrid_pipe.segment(raw) for raw in raw_test_sentences]
        hyb_ws_metrics = evaluate_word_segmentation(gold_ws_sentences, hyb_ws_preds, train_vocab=train_vocab)
        hyb_pos_e2e = [hybrid_pipe.tag_segmented(seg) for seg in hyb_ws_preds]
        hyb_e2e_metrics = evaluate_pos_tagging(gold_pos_sentences, hyb_pos_e2e, train_vocab=train_vocab)

        results.append(
            {
                "model": "hybrid_hmm_phobert",
                "ws_precision": round(hyb_ws_metrics["precision"], 4),
                "ws_recall": round(hyb_ws_metrics["recall"], 4),
                "ws_f1": round(hyb_ws_metrics["f1_score"], 4),
                "ws_oov_recall": round(hyb_ws_metrics["oov_recall"], 4),
                "ws_bi_accuracy": round(hyb_ws_metrics["bi_accuracy"], 4),
                "pos_accuracy": round(pb_pos_metrics["accuracy"], 4),
                "pos_macro_f1": round(pb_pos_metrics["macro_f1"], 4),
                "pos_oov_accuracy": round(pb_pos_metrics["oov_accuracy"], 4),
                "e2e_joint_f1": round(hyb_e2e_metrics["joint_f1"], 4),
            }
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
