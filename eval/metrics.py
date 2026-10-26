from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


def segmented_sentence_to_spans(segmented_text: str) -> List[Tuple[int, int, str]]:
    """
    Convert an underscore-segmented sentence into syllable-index spans:
    [(start_syllable_idx, end_syllable_idx, word_str), ...]
    Example: "Tôi là sinh_viên" -> [(0, 1, "Tôi"), (1, 2, "là"), (2, 4, "sinh_viên")]
    """
    words = segmented_text.strip().split()
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for word in words:
        syllables = [s for s in word.split("_") if s]
        length = max(1, len(syllables))
        spans.append((cursor, cursor + length, word))
        cursor += length
    return spans


def evaluate_ws_bi_tags(
    gold_tag_sequences: List[List[str]],
    pred_tag_sequences: List[List[str]],
) -> Dict[str, float]:
    """Compute syllable-level B/I accuracy and macro-F1."""
    correct = 0
    total = 0
    counts = {tag: {"tp": 0, "fp": 0, "fn": 0} for tag in ("B", "I")}

    for gold_seq, pred_seq in zip(gold_tag_sequences, pred_tag_sequences):
        for g, p in zip(gold_seq, pred_seq):
            total += 1
            if g == p:
                correct += 1
                if g in counts:
                    counts[g]["tp"] += 1
            else:
                if p in counts:
                    counts[p]["fp"] += 1
                if g in counts:
                    counts[g]["fn"] += 1

    f1_scores = []
    for tag in ("B", "I"):
        c = counts[tag]
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_scores.append(f1)

    return {
        "bi_accuracy": correct / total if total else 0.0,
        "bi_macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


def evaluate_word_segmentation(
    gold_sentences: List[str],
    pred_sentences: List[str],
    train_vocab: Optional[ Iterable[str] ] = None,
) -> Dict[str, float]:
    """
    Compute standard VLSP word-level exact boundary Precision, Recall, F1, and OOV Recall.
    """
    vocab_set: Optional[Set[str]] = (
        {w.lower() for w in train_vocab} if train_vocab is not None else None
    )

    total_gold_words = 0
    total_pred_words = 0
    correct_words = 0

    oov_gold_words = 0
    oov_correct_words = 0

    gold_bi_all: List[List[str]] = []
    pred_bi_all: List[List[str]] = []

    for gold_sent, pred_sent in zip(gold_sentences, pred_sentences):
        gold_spans = segmented_sentence_to_spans(gold_sent)
        pred_spans = segmented_sentence_to_spans(pred_sent)

        gold_span_set = {(s, e) for s, e, _ in gold_spans}
        pred_span_set = {(s, e) for s, e, _ in pred_spans}

        total_gold_words += len(gold_spans)
        total_pred_words += len(pred_spans)
        correct_words += len(gold_span_set & pred_span_set)

        if vocab_set is not None:
            for s, e, w in gold_spans:
                if w.lower() not in vocab_set:
                    oov_gold_words += 1
                    if (s, e) in pred_span_set:
                        oov_correct_words += 1

        # Build B/I sequences for syllable-level metrics
        gold_bi = ["B" if idx == 0 else "I" for _, _, w in gold_spans for idx, _ in enumerate(w.split("_"))]
        pred_bi = ["B" if idx == 0 else "I" for _, _, w in pred_spans for idx, _ in enumerate(w.split("_"))]
        if len(gold_bi) == len(pred_bi):
            gold_bi_all.append(gold_bi)
            pred_bi_all.append(pred_bi)

    precision = correct_words / total_pred_words if total_pred_words else 0.0
    recall = correct_words / total_gold_words if total_gold_words else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    oov_recall = oov_correct_words / oov_gold_words if oov_gold_words else 0.0

    bi_metrics = evaluate_ws_bi_tags(gold_bi_all, pred_bi_all)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "oov_recall": oov_recall,
        "bi_accuracy": bi_metrics["bi_accuracy"],
        "bi_macro_f1": bi_metrics["bi_macro_f1"],
        "gold_words": total_gold_words,
        "pred_words": total_pred_words,
    }


def evaluate_pos_tagging(
    gold_tagged_sentences: List[List[Tuple[str, str]]],
    pred_tagged_sentences: List[List[Tuple[str, str]]],
    train_vocab: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate POS tagging on both:
    - Token-level accuracy, macro-F1, weighted-F1, and OOV accuracy (when word lengths match)
    - Joint span+POS F1 (works even when predicted word segmentation differs from gold)
    """
    vocab_set: Optional[Set[str]] = (
        {w.lower() for w in train_vocab} if train_vocab is not None else None
    )

    token_correct = 0
    token_total = 0
    oov_correct = 0
    oov_total = 0

    per_tag = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    joint_gold_total = 0
    joint_pred_total = 0
    joint_correct = 0

    for gold_sent, pred_sent in zip(gold_tagged_sentences, pred_tagged_sentences):
        # Joint syllable-span + POS evaluation
        g_cursor = 0
        gold_Typed_spans = set()
        for word, tag in gold_sent:
            length = max(1, len([s for s in word.split("_") if s]))
            gold_Typed_spans.add((g_cursor, g_cursor + length, tag))
            g_cursor += length

        p_cursor = 0
        pred_Typed_spans = set()
        for word, tag in pred_sent:
            length = max(1, len([s for s in word.split("_") if s]))
            pred_Typed_spans.add((p_cursor, p_cursor + length, tag))
            p_cursor += length

        joint_gold_total += len(gold_Typed_spans)
        joint_pred_total += len(pred_Typed_spans)
        joint_correct += len(gold_Typed_spans & pred_Typed_spans)

        # Token-level evaluation when lengths match (or up to aligned words)
        for (g_word, g_tag), (p_word, p_tag) in zip(gold_sent, pred_sent):
            token_total += 1
            is_oov = vocab_set is not None and g_word.lower() not in vocab_set
            if is_oov:
                oov_total += 1
            if g_tag == p_tag:
                token_correct += 1
                per_tag[g_tag]["tp"] += 1
                if is_oov:
                    oov_correct += 1
            else:
                per_tag[p_tag]["fp"] += 1
                per_tag[g_tag]["fn"] += 1

    macro_f1_list = []
    weighted_f1_sum = 0.0
    total_support = 0

    for tag, c in per_tag.items():
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = c["tp"] + c["fn"]
        if support > 0:
            macro_f1_list.append(f1)
            weighted_f1_sum += f1 * support
            total_support += support

    joint_prec = joint_correct / joint_pred_total if joint_pred_total else 0.0
    joint_rec = joint_correct / joint_gold_total if joint_gold_total else 0.0
    joint_f1 = 2 * joint_prec * joint_rec / (joint_prec + joint_rec) if (joint_prec + joint_rec) else 0.0

    return {
        "accuracy": token_correct / token_total if token_total else 0.0,
        "macro_f1": sum(macro_f1_list) / len(macro_f1_list) if macro_f1_list else 0.0,
        "weighted_f1": weighted_f1_sum / total_support if total_support else 0.0,
        "oov_accuracy": oov_correct / oov_total if oov_total else 0.0,
        "joint_precision": joint_prec,
        "joint_recall": joint_rec,
        "joint_f1": joint_f1,
        "tokens": token_total,
    }


def hmm_eval_metrics(y_true: List[str], y_pred: List[str], train_vocab: Optional[Iterable[str]] = None) -> Dict[str, float]:
    return evaluate_word_segmentation(y_true, y_pred, train_vocab=train_vocab)


def llama_eval_metrics(y_true: List[str], y_pred: List[str], train_vocab: Optional[Iterable[str]] = None) -> Dict[str, float]:
    return evaluate_word_segmentation(y_true, y_pred, train_vocab=train_vocab)


def hybrid_eval_metrics(y_true: List[str], y_pred: List[str], train_vocab: Optional[Iterable[str]] = None) -> Dict[str, float]:
    return evaluate_word_segmentation(y_true, y_pred, train_vocab=train_vocab)
