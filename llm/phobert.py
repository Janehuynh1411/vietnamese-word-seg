import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

from data.dataset_converter import ws_tags_to_segmented_words


def get_device(prefer_device: Optional[str] = None) -> torch.device:
    if prefer_device:
        return torch.device(prefer_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_phobert_features(
    sentence: str = "Tôi là sinh_viên trường đại_học Công_nghệ .",
    model_name: str = "vinai/phobert-base",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract contextualized embeddings from PhoBERT for an underscore-segmented sentence.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    phobert = AutoModel.from_pretrained(model_name)
    phobert.eval()

    input_ids = torch.tensor([tokenizer.encode(sentence)])
    with torch.no_grad():
        outputs = phobert(input_ids)
        last_hidden_state = outputs.last_hidden_state

    return input_ids, last_hidden_state


def align_labels_with_tokens(
    tokenizer,
    tokens: List[str],
    labels: Optional[List[str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    max_length: int = 256,
) -> Dict[str, object]:
    """
    Tokenize pre-split tokens (syllables for WS, or underscore-joined words for POS)
    using PhoBERT's BPE tokenizer and align token-level labels with BPE subwords.

    First subword of each token receives its label ID (and word index);
    continuation subwords ('@@') and special tokens (<s>, </s>, <pad>) receive -100.
    """
    subword_ids: List[int] = [tokenizer.bos_token_id]
    label_ids: List[int] = [-100]
    word_ids: List[Optional[int]] = [None]

    max_subwords = max_length - 2
    unk_id = tokenizer.unk_token_id

    for word_idx, token in enumerate(tokens):
        subwords = tokenizer.tokenize(token)
        if not subwords:
            subwords = [tokenizer.unk_token]
        ids = tokenizer.convert_tokens_to_ids(subwords)
        if isinstance(ids, int):
            ids = [ids]

        if len(subword_ids) - 1 + len(ids) > max_subwords:
            break

        for sub_idx, sub_id in enumerate(ids):
            subword_ids.append(sub_id if sub_id is not None else unk_id)
            if sub_idx == 0:
                word_ids.append(word_idx)
                if labels is not None and label2id is not None:
                    label_ids.append(label2id.get(labels[word_idx], 0))
                else:
                    label_ids.append(-100)
            else:
                word_ids.append(None)
                label_ids.append(-100)

    subword_ids.append(tokenizer.eos_token_id)
    label_ids.append(-100)
    word_ids.append(None)

    attention_mask = [1] * len(subword_ids)

    return {
        "input_ids": subword_ids,
        "attention_mask": attention_mask,
        "labels": label_ids,
        "word_ids": word_ids,
    }


class PhoBERTTokenDataset(Dataset):
    def __init__(
        self,
        samples: List[dict],
        tokenizer,
        label2id: Dict[str, int],
        token_key: str = "syllables",
        label_key: str = "tags",
        max_length: int = 256,
    ):
        self.encoded_samples = []
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

        for item in samples:
            tokens = item.get(token_key, [])
            labels = item.get(label_key, [])
            if not tokens or len(tokens) != len(labels):
                continue
            aligned = align_labels_with_tokens(
                tokenizer=tokenizer,
                tokens=tokens,
                labels=labels,
                label2id=label2id,
                max_length=max_length,
            )
            self.encoded_samples.append(aligned)

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx: int) -> dict:
        return self.encoded_samples[idx]

    def collate_fn(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class BasePhoBERTTokenClassifier:
    def __init__(
        self,
        model_name_or_path: str = "vinai/phobert-base",
        label2id: Optional[Dict[str, int]] = None,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.device = get_device(device)

        label_map_path = os.path.join(model_name_or_path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                saved_map = json.load(f)
            label2id = saved_map["label2id"]

        if label2id is None:
            label2id = {"B": 0, "I": 1}

        self.label2id = dict(label2id)
        self.id2label = {int(v): k for k, v in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        ).to(self.device)

    def fit(
        self,
        train_samples: List[dict],
        val_samples: Optional[List[dict]] = None,
        token_key: str = "syllables",
        label_key: str = "tags",
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        train_dataset = PhoBERTTokenDataset(
            samples=train_samples,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            token_key=token_key,
            label_key=label_key,
            max_length=self.max_length,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            steps = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                steps += 1

            avg_train_loss = total_loss / max(1, steps)
            history["train_loss"].append(avg_train_loss)

            val_loss_str = ""
            if val_samples:
                val_loss = self.evaluate_loss(val_samples, token_key=token_key, label_key=label_key, batch_size=batch_size)
                history["val_loss"].append(val_loss)
                val_loss_str = f" | val_loss={val_loss:.4f}"

            print(f"Epoch {epoch}/{epochs} | train_loss={avg_train_loss:.4f}{val_loss_str}")

        if save_dir:
            self.save(save_dir)

        return history

    def evaluate_loss(
        self,
        samples: List[dict],
        token_key: str = "syllables",
        label_key: str = "tags",
        batch_size: int = 16,
    ) -> float:
        dataset = PhoBERTTokenDataset(
            samples=samples,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            token_key=token_key,
            label_key=label_key,
            max_length=self.max_length,
        )
        if len(dataset) == 0:
            return 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        self.model.eval()
        total_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                total_loss += float(outputs.loss.item())
                steps += 1
        return total_loss / max(1, steps)

    def predict_tokens(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        self.model.eval()
        aligned = align_labels_with_tokens(
            tokenizer=self.tokenizer,
            tokens=tokens,
            max_length=self.max_length,
        )
        input_ids = torch.tensor([aligned["input_ids"]], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([aligned["attention_mask"]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        default_label = self.id2label.get(0, "B")
        predictions = [default_label] * len(tokens)
        for sub_idx, word_idx in enumerate(aligned["word_ids"]):
            if word_idx is not None and 0 <= word_idx < len(tokens):
                predictions[word_idx] = self.id2label.get(pred_ids[sub_idx], default_label)

        return predictions

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump({"label2id": self.label2id, "id2label": self.id2label}, f, ensure_ascii=False, indent=2)


class PhoBERTWordSegmenter(BasePhoBERTTokenClassifier):
    """
    Syllable-level B/I token classifier using PhoBERT for Vietnamese Word Segmentation.
    """

    def __init__(
        self,
        model_name_or_path: str = "vinai/phobert-base",
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            label2id={"B": 0, "I": 1},
            max_length=max_length,
            device=device,
        )

    def predict_tags(self, syllables: List[str]) -> List[str]:
        tags = self.predict_tags_raw(syllables)
        if tags:
            tags[0] = "B"
        return tags

    def predict_tags_raw(self, syllables: List[str]) -> List[str]:
        return self.predict_tokens(syllables)

    def segment(self, text: str) -> str:
        syllables = text.strip().split()
        if not syllables:
            return ""
        tags = self.predict_tags(syllables)
        words = ws_tags_to_segmented_words(syllables, tags)
        return " ".join(words)


class PhoBERTPosTagger(BasePhoBERTTokenClassifier):
    """
    Word-level POS token classifier using PhoBERT on underscore-segmented Vietnamese words.
    """

    def __init__(
        self,
        model_name_or_path: str = "vinai/phobert-base",
        label2id: Optional[Dict[str, int]] = None,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        if label2id is None and not os.path.exists(os.path.join(model_name_or_path, "label_map.json")):
            default_tags = ["N", "V", "A", "P", "R", "E", "M", "L", "Nc", "Np", "C", "CC", "T", "I", "CH"]
            label2id = {t: idx for idx, t in enumerate(default_tags)}
        super().__init__(
            model_name_or_path=model_name_or_path,
            label2id=label2id,
            max_length=max_length,
            device=device,
        )

    @staticmethod
    def build_label_map_from_samples(samples: List[dict], label_key: str = "tags") -> Dict[str, int]:
        unique_tags = sorted({tag for s in samples for tag in s.get(label_key, [])})
        if not unique_tags:
            unique_tags = ["N", "V", "A", "P", "R", "E", "M", "Np", "CH"]
        return {tag: idx for idx, tag in enumerate(unique_tags)}

    def tag(self, words: List[str]) -> List[Tuple[str, str]]:
        if not words:
            return []
        tags = self.predict_tokens(words)
        return list(zip(words, tags))

    def tag_text(self, segmented_text: str) -> List[Tuple[str, str]]:
        words = segmented_text.strip().split()
        return self.tag(words)


def main():
    parser = argparse.ArgumentParser(description="PhoBERT Feature Extraction, Word Segmentation (B/I), and POS Tagging")
    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo", help="Run the PhoBERT feature extraction demo")
    p_demo.add_argument("--model", default="vinai/phobert-base")
    p_demo.add_argument("--sentence", default="Tôi là sinh_viên trường đại_học Công_nghệ .")

    p_train = sub.add_parser("train", help="Fine-tune PhoBERT for Word Segmentation (ws) or POS Tagging (pos)")
    p_train.add_argument("--task", choices=["ws", "pos", "both"], default="both")
    p_train.add_argument("--ws-data", default="data/processed/ws_splits.json")
    p_train.add_argument("--pos-data", default="data/processed/pos_splits.json")
    p_train.add_argument("--model", default="vinai/phobert-base")
    p_train.add_argument("--ws-save-dir", default="models/phobert_ws")
    p_train.add_argument("--pos-save-dir", default="models/phobert_pos")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=2e-5)

    p_pred = sub.add_parser("predict", help="Run PhoBERT WS and/or POS prediction on input text")
    p_pred.add_argument("--task", choices=["ws", "pos", "both"], default="both")
    p_pred.add_argument("--text", default="Tôi là sinh viên trường đại học Công nghệ .")
    p_pred.add_argument("--ws-model", default="models/phobert_ws")
    p_pred.add_argument("--pos-model", default="models/phobert_pos")
    p_pred.add_argument("--base-model", default="vinai/phobert-base")

    args = parser.parse_args()

    if args.command in (None, "demo"):
        sentence = getattr(args, "sentence", "Tôi là sinh_viên trường đại_học Công_nghệ .")
        model_name = getattr(args, "model", "vinai/phobert-base")
        input_ids, last_hidden_state = extract_phobert_features(sentence=sentence, model_name=model_name)
        print("Input IDs:", input_ids)
        print("Features shape:", last_hidden_state.shape)

    elif args.command == "train":
        if args.task in ("ws", "both"):
            with open(args.ws_data, "r", encoding="utf-8") as f:
                ws_splits = json.load(f)
            print(f"Training PhoBERT Word Segmenter on {len(ws_splits['train'])} sentences...")
            ws_clf = PhoBERTWordSegmenter(model_name_or_path=args.model)
            ws_clf.fit(
                train_samples=ws_splits["train"],
                val_samples=ws_splits.get("val"),
                token_key="syllables",
                label_key="tags",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                save_dir=args.ws_save_dir,
            )
            print(f"Saved PhoBERT WS model to {args.ws_save_dir}")

        if args.task in ("pos", "both"):
            with open(args.pos_data, "r", encoding="utf-8") as f:
                pos_splits = json.load(f)
            label2id = PhoBERTPosTagger.build_label_map_from_samples(
                pos_splits["train"] + pos_splits.get("val", []) + pos_splits.get("test", []),
                label_key="tags",
            )
            print(f"Training PhoBERT POS Tagger ({len(label2id)} tags) on {len(pos_splits['train'])} sentences...")
            pos_clf = PhoBERTPosTagger(model_name_or_path=args.model, label2id=label2id)
            pos_clf.fit(
                train_samples=pos_splits["train"],
                val_samples=pos_splits.get("val"),
                token_key="words",
                label_key="tags",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                save_dir=args.pos_save_dir,
            )
            print(f"Saved PhoBERT POS model to {args.pos_save_dir}")

    elif args.command == "predict":
        ws_path = args.ws_model if os.path.exists(args.ws_model) else args.base_model
        pos_path = args.pos_model if os.path.exists(args.pos_model) else args.base_model

        text = args.text
        if args.task in ("ws", "both"):
            ws_clf = PhoBERTWordSegmenter(model_name_or_path=ws_path)
            segmented = ws_clf.segment(text)
            print(f"Segmented: {segmented}")
            text = segmented
        if args.task in ("pos", "both"):
            pos_clf = PhoBERTPosTagger(model_name_or_path=pos_path)
            tagged = pos_clf.tag_text(text)
            print("POS Tagged:", " ".join(f"{w}/{t}" for w, t in tagged))


if __name__ == "__main__":
    main()
