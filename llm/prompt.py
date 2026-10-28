from typing import Dict, Iterable, List, Optional, Tuple

from data.dataset_converter import normalized_ws_syllables


DEFAULT_FEW_SHOT_EXAMPLES: List[Tuple[str, str, str]] = [
    (
        "Tôi là sinh viên trường đại học Công nghệ .",
        "Tôi là sinh_viên trường đại_học Công_nghệ .",
        "Tôi/P là/V sinh_viên/N trường/N đại_học/N Công_nghệ/Np ./CH",
    ),
    (
        "Ẩm thực làm thăng hoa tình yêu trong cuộc sống lứa đôi .",
        "Ẩm_thực làm thăng_hoa tình_yêu trong cuộc_sống lứa_đôi .",
        "Ẩm_thực/N làm/V thăng_hoa/V tình_yêu/N trong/E cuộc_sống/N lứa_đôi/N ./CH",
    ),
    (
        "Các nhà khoa học đã thống nhất rằng ngưỡng tiếng ồn vượt quá 85 dB có thể gây điếc .",
        "Các nhà_khoa_học đã thống_nhất rằng ngưỡng tiếng_ồn vượt quá 85 dB có_thể gây_điếc .",
        "Các/R nhà_khoa_học/N đã/R thống_nhất/V rằng/C ngưỡng/N tiếng_ồn/N vượt/V quá/R 85/M dB/N có_thể/R gây_điếc/V ./CH",
    ),
]


SYSTEM_PROMPT_WS = (
    "You are an expert Vietnamese computational linguist specializing in Vietnamese Word Segmentation. "
    "Join syllables belonging to the same compound word using underscores ('_'), while keeping separate words "
    "separated by spaces. Preserve every syllable and punctuation token in the exact original order. "
    "Return ONLY the segmented sentence with no extra commentary."
)


SYSTEM_PROMPT_POS = (
    "You are an expert Vietnamese computational linguist specializing in Part-of-Speech (POS) tagging. "
    "Given an underscore-segmented Vietnamese sentence, attach a slash and POS tag to each token in the format "
    "word/TAG (using standard tags: N, V, A, P, R, E, M, Np, C, CH, X). "
    "Return ONLY the space-separated word/TAG tokens with no extra commentary."
)


def _constraint_hint(
    preferred_compounds: Optional[Iterable[str]] = None,
    rejected_compounds: Optional[Iterable[str]] = None,
    limit: int = 40,
) -> str:
    lines = []
    preferred = sorted({item.strip() for item in (preferred_compounds or []) if item and str(item).strip()})
    rejected = sorted({item.strip() for item in (rejected_compounds or []) if item and str(item).strip()})
    if preferred:
        shown = ", ".join(preferred[:limit])
        lines.append(f"Prefer these compound joins: {shown}.")
    if rejected:
        shown = ", ".join(rejected[:limit])
        lines.append(f"Do not join these rejected boundaries: {shown}.")
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


def build_ws_messages(
    raw_sentence: str,
    strategy: str = "few-shot",
    examples: Optional[List[Tuple[str, str, str]]] = None,
    ambiguous_indices: Optional[List[int]] = None,
    preferred_compounds: Optional[Iterable[str]] = None,
    rejected_compounds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    """
    Build chat messages for Vietnamese Word Segmentation using zero-shot, few-shot, or chain-of-thought.
    Optional preference compounds from models/llm_dpo_memory.json are injected as constraints.
    """
    shots = examples if examples is not None else DEFAULT_FEW_SHOT_EXAMPLES
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_WS}]

    if strategy == "few-shot":
        for raw_ex, seg_ex, _ in shots:
            messages.append({"role": "user", "content": f"Segment: {raw_ex}"})
            messages.append({"role": "assistant", "content": seg_ex})

    focus_hint = ""
    if ambiguous_indices:
        syllables = raw_sentence.split()
        focus_tokens = [syllables[i] for i in ambiguous_indices if 0 <= i < len(syllables)]
        if focus_tokens:
            focus_hint = f"\nPay special attention to ambiguous boundaries around: {', '.join(focus_tokens)}."
    focus_hint += _constraint_hint(preferred_compounds, rejected_compounds)

    if strategy == "chain-of-thought":
        user_prompt = (
            f"Segment the following Vietnamese sentence into compound words connected by underscores ('_')."
            f"{focus_hint}\n"
            f"Think step-by-step about compound word boundaries, then output the final segmented sentence on the "
            f"last line prefixed with 'OUTPUT: '.\nSentence: {raw_sentence}"
        )
    else:
        user_prompt = f"Segment: {raw_sentence}{focus_hint}"

    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_pos_messages(
    segmented_sentence: str,
    strategy: str = "few-shot",
    examples: Optional[List[Tuple[str, str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build chat messages for Vietnamese POS tagging on an underscore-segmented sentence.
    """
    shots = examples if examples is not None else DEFAULT_FEW_SHOT_EXAMPLES
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_POS}]

    if strategy == "few-shot":
        for _, seg_ex, pos_ex in shots:
            messages.append({"role": "user", "content": f"POS Tag: {seg_ex}"})
            messages.append({"role": "assistant", "content": pos_ex})

    if strategy == "chain-of-thought":
        user_prompt = (
            "Assign POS tags (N, V, A, P, R, E, M, Np, C, CH, X) to each token in the segmented sentence. "
            "Output the final tagged sequence on the last line prefixed with 'OUTPUT: '.\n"
            f"Sentence: {segmented_sentence}"
        )
    else:
        user_prompt = f"POS Tag: {segmented_sentence}"

    messages.append({"role": "user", "content": user_prompt})
    return messages


def parse_llm_ws_output(response_text: str, fallback_raw_sentence: str) -> str:
    """
    Extract the clean segmented sentence from LLM response text and verify syllable preservation.

    Comparison is NFC-normalized and ignores punctuation that the model attached to a
    neighboring syllable. A candidate that still does not match the original syllable
    sequence is discarded in favor of the unsegmented sentence.
    """
    if not response_text or not response_text.strip():
        return fallback_raw_sentence.strip()

    lines = [ln.strip() for ln in response_text.strip().splitlines() if ln.strip()]
    candidate = lines[-1]
    for ln in reversed(lines):
        if ln.upper().startswith("OUTPUT:"):
            candidate = ln.split(":", 1)[1].strip()
            break
        if ln.lower().startswith("segmented:"):
            candidate = ln.split(":", 1)[1].strip()
            break

    candidate = candidate.strip("`\"' ")
    orig_syllables = normalized_ws_syllables(fallback_raw_sentence)
    cand_syllables = normalized_ws_syllables(candidate)

    if cand_syllables != orig_syllables:
        return fallback_raw_sentence.strip()

    return candidate


def parse_llm_pos_output(response_text: str, segmented_words: List[str]) -> List[Tuple[str, str]]:
    """
    Extract (word, pos_tag) pairs from LLM output, aligning safely to segmented_words.
    """
    if not segmented_words:
        return []
    if not response_text or not response_text.strip():
        return [(w, "CH" if all(not c.isalnum() for c in w) else "N") for w in segmented_words]

    lines = [ln.strip() for ln in response_text.strip().splitlines() if ln.strip()]
    candidate = lines[-1]
    for ln in reversed(lines):
        if ln.upper().startswith("OUTPUT:"):
            candidate = ln.split(":", 1)[1].strip()
            break

    raw_tokens = candidate.strip("`\"' ").split()
    parsed_tags: List[str] = []
    for tok in raw_tokens:
        if "/" in tok and tok != "/":
            _, tag = tok.rsplit("/", 1)
            parsed_tags.append(tag.strip() or "N")

    if len(parsed_tags) == len(segmented_words):
        return list(zip(segmented_words, parsed_tags))

    return [(w, "CH" if all(not c.isalnum() for c in w) else "N") for w in segmented_words]
