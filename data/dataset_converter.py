import argparse
import json
import os
import random
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from pos.tagger import DEMO_CORPUS


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def read_text_file(file_path: str) -> Optional[str]:
    """Read a text file supporting UTF-8 and UTF-16 encodings."""
    with open(file_path, "rb") as f:
        raw = f.read()
    if not raw:
        return None
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("utf-16")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
    text = normalize_text(text)
    return text if text else None


def segmented_words_to_ws_tags(words: List[str]) -> Tuple[List[str], List[str]]:
    """
    Convert underscore-segmented words into syllable tokens and B/I boundary tags.
    Example: ["Tôi", "là", "sinh_viên"] -> (["Tôi", "là", "sinh", "viên"], ["B", "B", "B", "I"])
    """
    syllables: List[str] = []
    tags: List[str] = []
    for word in words:
        parts = [p for p in word.split("_") if p]
        for idx, part in enumerate(parts):
            syllables.append(part)
            tags.append("B" if idx == 0 else "I")
    return syllables, tags


def ws_tags_to_segmented_words(syllables: List[str], tags: List[str]) -> List[str]:
    """Reconstruct underscore-segmented words from syllables and B/I tags."""
    if not syllables:
        return []
    words: List[str] = []
    current: List[str] = []
    for syll, tag in zip(syllables, tags):
        if tag == "B" or not current:
            if current:
                words.append("_".join(current))
            current = [syll]
        else:
            current.append(syll)
    if current:
        words.append("_".join(current))
    return words


def normalize_pos_tag(tag: str) -> str:
    """
    Normalize POS tags across HumanSignal / Label Studio, Underthesea, PyVi, and VLSP.
    Unifies punctuation tags ('CH', 'F', 'PU') to 'CH' and extracts short codes from
    verbose Label Studio labels such as 'N (Danh từ / Noun)'.
    """
    if not tag:
        return "N"
    short = tag.strip().split()[0]
    mapping = {
        "F": "CH",
        "PU": "CH",
        "PUNCT": "CH",
        "Nb": "N",
        "Ny": "Np",
        "Vb": "V",
        "Cc": "C",
        "CC": "C",
        "S": "E",
    }
    return mapping.get(short, short)


def parse_pos_line(line: str) -> Optional[Tuple[List[str], List[str]]]:
    """
    Parse a line of 'word/POS word/POS' into (words, pos_tags).
    Returns None if the line is not in word/POS format.
    """
    tokens = normalize_text(line).split()
    if not tokens:
        return None
    words: List[str] = []
    pos_tags: List[str] = []
    for tok in tokens:
        if "/" not in tok or tok == "/":
            return None
        word, tag = tok.rsplit("/", 1)
        if not word or not tag:
            return None
        words.append(word)
        pos_tags.append(normalize_pos_tag(tag))
    return words, pos_tags


def silver_annotate_sentence(sentence: str) -> Tuple[List[str], List[str]]:
    """
    Generate silver word segmentation and POS tags for an unsegmented sentence.
    Tries underthesea first, then pyvi, then falls back to heuristic tokenization.
    """
    sentence = normalize_text(sentence)
    try:
        from underthesea import pos_tag

        tagged = pos_tag(sentence)
        words = ["_".join(w.strip().split()) for w, _ in tagged if w.strip()]
        tags = [normalize_pos_tag(t) for w, t in tagged if w.strip()]
        if words and len(words) == len(tags):
            return words, tags
    except Exception:
        pass

    try:
        from pyvi import ViPosTagger, ViTokenizer

        tokenized = ViTokenizer.tokenize(sentence)
        words, tags = ViPosTagger.postagging(tokenized)
        if words and len(words) == len(tags):
            return list(words), [normalize_pos_tag(t) for t in tags]
    except Exception:
        pass

    # Fallback: treat whitespace tokens as words and assign basic heuristic POS tags
    raw_tokens = re.findall(r"\w+|[^\w\s]", sentence, flags=re.UNICODE)
    words = []
    tags = []
    for tok in raw_tokens:
        words.append(tok)
        if all(not c.isalnum() for c in tok):
            tags.append("CH")
        elif tok.isdigit():
            tags.append("M")
        elif tok[0].isupper():
            tags.append("Np")
        else:
            tags.append("N")
    return words, tags


def parse_label_studio_task(task: dict, use_silver_fallback: bool = True) -> Optional[Tuple[List[str], List[str]]]:
    """
    Convert a single Label Studio task dictionary into (segmented_words, pos_tags).
    Supports:
    1. Full Label Studio export (`annotations` -> `result` -> `value` -> `{start, end, text, labels}`)
    2. JSON-MIN export (`label` list of `{start, end, text, labels}`)
    3. Pre-segmented text or silver fallback if unannotated.
    """
    data_obj = task.get("data", task)
    text = normalize_text(data_obj.get("text", ""))
    if not text:
        return None

    spans = []
    annotations = task.get("annotations") or task.get("completions") or []
    if annotations:
        for ann in annotations:
            results = ann.get("result", [])
            for res in results:
                val = res.get("value", {})
                labels = val.get("labels") or val.get("tag") or []
                if "start" in val and "end" in val and labels:
                    spans.append(
                        (
                            int(val["start"]),
                            int(val["end"]),
                            val.get("text", text[int(val["start"]) : int(val["end"])]),
                            labels[0],
                        )
                    )
            if spans:
                break
    elif isinstance(task.get("label"), list):
        for val in task["label"]:
            labels = val.get("labels") or []
            if "start" in val and "end" in val and labels:
                spans.append(
                    (
                        int(val["start"]),
                        int(val["end"]),
                        val.get("text", text[int(val["start"]) : int(val["end"])]),
                        labels[0],
                    )
                )

    if spans:
        spans.sort(key=lambda x: (x[0], x[1]))
        # Deduplicate exact or overlapping spans
        dedup_spans = []
        last_end = -1
        for start, end, span_text, lbl in spans:
            if start < last_end:
                continue
            dedup_spans.append((start, end, span_text, lbl))
            last_end = end
        spans = dedup_spans
        # Check whether labels are B/I syllable tags or word/POS span labels
        label_set = {s[3] for s in spans}
        if label_set.issubset({"B", "I", "B-W", "I-W", "B_W", "I_W"}):
            syllables = []
            bi_tags = []
            for _, _, span_text, lbl in spans:
                parts = span_text.strip().split()
                norm_lbl = "B" if lbl.startswith("B") else "I"
                for idx, part in enumerate(parts):
                    syllables.append(part)
                    bi_tags.append(norm_lbl if idx == 0 else "I")
            words = ws_tags_to_segmented_words(syllables, bi_tags)
            pos_tags = ["N"] * len(words)
            return words, pos_tags
        else:
            # Each span represents a segmented word (possibly multi-syllable) with a POS/entity label
            words = []
            pos_tags = []
            cursor = 0
            for start, end, span_text, lbl in spans:
                if start > cursor:
                    gap = text[cursor:start].strip()
                    if gap:
                        for gap_tok in gap.split():
                            words.append(gap_tok)
                            pos_tags.append("CH" if all(not c.isalnum() for c in gap_tok) else "N")
                clean_span = "_".join(span_text.strip().split())
                if clean_span:
                    words.append(clean_span)
                    pos_tags.append(normalize_pos_tag(lbl))
                cursor = max(cursor, end)
            if cursor < len(text):
                tail = text[cursor:].strip()
                if tail:
                    for tail_tok in tail.split():
                        words.append(tail_tok)
                        pos_tags.append("CH" if all(not c.isalnum() for c in tail_tok) else "N")
            if words:
                return words, pos_tags

    # Check if text is already in word/POS format
    parsed_pos = parse_pos_line(text)
    if parsed_pos is not None:
        return parsed_pos

    # Check if text already has underscore word segmentation
    if "_" in text:
        words = text.split()
        pos_tags = ["CH" if all(not c.isalnum() for c in w) else "N" for w in words]
        return words, pos_tags

    if use_silver_fallback:
        return silver_annotate_sentence(text)

    return None


def make_record(words: List[str], pos_tags: List[str]) -> dict:
    norm_tags = [normalize_pos_tag(t) for t in pos_tags]
    syllables, ws_tags = segmented_words_to_ws_tags(words)
    return {
        "words": words,
        "pos_tags": norm_tags,
        "syllables": syllables,
        "ws_tags": ws_tags,
        "segmented_text": " ".join(words),
        "raw_text": " ".join(syllables),
        "pos_text": " ".join(f"{w}/{t}" for w, t in zip(words, norm_tags)),
    }


def load_records_from_sources(
    raw_dir: str = "data/raw",
    label_studio_paths: Optional[List[str]] = None,
    use_silver: bool = True,
) -> List[dict]:
    records: List[dict] = []

    # 1. Load from explicit Label Studio JSON files if provided or discovered in data/processed
    ls_files = list(label_studio_paths or [])
    if not ls_files and os.path.exists("data/processed"):
        for fname in sorted(os.listdir("data/processed")):
            if fname.startswith("label_studio") and fname.endswith(".json"):
                ls_files.append(os.path.join("data/processed", fname))

    for ls_path in ls_files:
        if not os.path.exists(ls_path):
            continue
        with open(ls_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        if isinstance(tasks, dict):
            tasks = [tasks]
        for task in tasks:
            parsed = parse_label_studio_task(task, use_silver_fallback=use_silver)
            if parsed:
                words, pos_tags = parsed
                if words:
                    records.append(make_record(words, pos_tags))

    # 2. Load from raw text files in raw_dir
    if os.path.exists(raw_dir):
        if os.path.isfile(raw_dir):
            raw_files = [raw_dir]
        else:
            raw_files = []
            for root, _, files in os.walk(raw_dir):
                for fname in sorted(files):
                    if fname.endswith(".txt"):
                        raw_files.append(os.path.join(root, fname))

        for fpath in raw_files:
            content = read_text_file(fpath)
            if not content:
                continue
            for line in content.splitlines():
                line = normalize_text(line)
                if not line or line.startswith("#"):
                    continue
                parsed_pos = parse_pos_line(line)
                if parsed_pos is not None:
                    words, pos_tags = parsed_pos
                    records.append(make_record(words, pos_tags))
                elif "_" in line:
                    words = line.split()
                    pos_tags = ["CH" if all(not c.isalnum() for c in w) else "N" for w in words]
                    records.append(make_record(words, pos_tags))
                elif use_silver:
                    # Split paragraph into sentences if needed
                    try:
                        from underthesea import sent_tokenize

                        sentences = sent_tokenize(line)
                    except Exception:
                        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
                    for sent in sentences:
                        words, pos_tags = silver_annotate_sentence(sent)
                        if words:
                            records.append(make_record(words, pos_tags))

    # 3. Fallback to built-in DEMO_CORPUS if no external files were found
    if not records:
        for line in normalize_text(DEMO_CORPUS).splitlines():
            if not line.strip():
                continue
            parsed = parse_pos_line(line)
            if parsed:
                words, pos_tags = parsed
                records.append(make_record(words, pos_tags))

    return records


def split_records(
    records: List[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    if n < 3:
        return {"train": items, "val": items, "test": items}

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    train_set = items[:n_train]
    val_set = items[n_train : n_train + n_val] or train_set[:1]
    test_set = items[n_train + n_val :] or train_set[-1:]

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }


def save_processed_splits(
    splits: Dict[str, List[dict]],
    output_dir: str = "data/processed",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    ws_splits = {
        split_name: [
            {
                "syllables": r["syllables"],
                "tags": r["ws_tags"],
                "segmented_text": r["segmented_text"],
                "raw_text": r["raw_text"],
            }
            for r in split_records_list
        ]
        for split_name, split_records_list in splits.items()
    }

    pos_splits = {
        split_name: [
            {
                "words": r["words"],
                "tags": r["pos_tags"],
                "syllables": r["syllables"],
                "ws_tags": r["ws_tags"],
                "segmented_text": r["segmented_text"],
                "pos_text": r["pos_text"],
            }
            for r in split_records_list
        ]
        for split_name, split_records_list in splits.items()
    }

    ws_path = os.path.join(output_dir, "ws_splits.json")
    pos_path = os.path.join(output_dir, "pos_splits.json")

    with open(ws_path, "w", encoding="utf-8") as f:
        json.dump(ws_splits, f, ensure_ascii=False, indent=2)

    with open(pos_path, "w", encoding="utf-8") as f:
        json.dump(pos_splits, f, ensure_ascii=False, indent=2)

    # Save plain-text files expected by hmm/train.py and pos/tagger.py
    train_txt = os.path.join(output_dir, "train.txt")
    test_txt = os.path.join(output_dir, "test.txt")
    pos_train_txt = os.path.join(output_dir, "pos_train.txt")
    pos_test_txt = os.path.join(output_dir, "pos_test.txt")

    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(r["segmented_text"] for r in splits["train"]) + "\n")

    with open(test_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(r["segmented_text"] for r in splits["test"]) + "\n")

    with open(pos_train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(r["pos_text"] for r in splits["train"]) + "\n")

    with open(pos_test_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(r["pos_text"] for r in splits["test"]) + "\n")

    # Populate data/dictionary.txt from unique segmented words across splits
    dict_path = "data/dictionary.txt"
    os.makedirs(os.path.dirname(dict_path) or ".", exist_ok=True)
    vocab_words = sorted(
        {
            w.lower()
            for split_list in splits.values()
            for r in split_list
            for w in r["words"]
            if any(c.isalnum() for c in w)
        }
    )
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_words) + "\n")

    return {
        "ws_splits": ws_path,
        "pos_splits": pos_path,
        "train_txt": train_txt,
        "test_txt": test_txt,
        "pos_train_txt": pos_train_txt,
        "pos_test_txt": pos_test_txt,
        "dictionary": dict_path,
    }


def prepare_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    label_studio_paths: Optional[List[str]] = None,
    use_silver: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, object]:
    records = load_records_from_sources(
        raw_dir=raw_dir,
        label_studio_paths=label_studio_paths,
        use_silver=use_silver,
    )
    splits = split_records(records, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    paths = save_processed_splits(splits, output_dir=output_dir)
    return {
        "total_sentences": len(records),
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "paths": paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw text or Label Studio JSON exports into WS (B/I) and POS dataset splits."
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Directory or file with raw/segmented text")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for splits")
    parser.add_argument(
        "--label-studio",
        nargs="*",
        default=None,
        help="Optional path(s) to Label Studio JSON exports",
    )
    parser.add_argument(
        "--silver",
        action="store_true",
        default=True,
        help="Use underthesea/pyvi silver annotations for unannotated raw text (default: True)",
    )
    parser.add_argument(
        "--no-silver",
        action="store_false",
        dest="silver",
        help="Disable silver annotation fallback for raw text",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        label_studio_paths=args.label_studio,
        use_silver=args.silver,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        f"Prepared {summary['total_sentences']} sentences "
        f"(train={summary['train_count']}, val={summary['val_count']}, test={summary['test_count']})"
    )
    for k, v in summary["paths"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
