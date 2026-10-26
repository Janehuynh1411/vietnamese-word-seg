import os
import time
from typing import Dict, List, Optional, Set, Tuple

from data.dataset_converter import normalize_pos_tag, ws_tags_to_segmented_words
from llm.prompt import (
    build_pos_messages,
    build_ws_messages,
    parse_llm_pos_output,
    parse_llm_ws_output,
)


class LLMWordSegmenterClient:
    """
    LLM client for Vietnamese Word Segmentation & POS Disambiguation.
    Supports:
      1. Cloud LLM routing via LiteLLM (Groq Llama-3.1-70B primary, Gemini-1.5-Flash fallback)
         when GROQ_API_KEY or GEMINI_API_KEY is present in the environment.
      2. Local dictionary + Underthesea/PyVi maximum-matching fallback when running offline
         or without external API keys, ensuring deterministic zero-cost local execution.
    """

    def __init__(
        self,
        primary_model: str = "groq/llama-3.1-70b-versatile",
        fallback_model: str = "gemini/gemini-1.5-flash",
        strategy: str = "few-shot",
        temperature: float = 0.0,
        max_tokens: int = 512,
        dictionary_path: str = "data/dictionary.txt",
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.strategy = strategy
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dictionary_path = dictionary_path
        self.vocab: Set[str] = self._load_dictionary(dictionary_path)

        self.stats = {
            "api_calls": 0,
            "local_fallback_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_latency_ms": 0.0,
        }

    @staticmethod
    def _load_dictionary(path: str) -> Set[str]:
        vocab: Set[str] = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        vocab.add(w)
        return vocab

    def _has_api_keys(self) -> bool:
        return bool(
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

    def _call_chat_completion(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if not self._has_api_keys():
            return None
        try:
            import litellm

            for model_name in (self.primary_model, self.fallback_model):
                if not model_name:
                    continue
                try:
                    t0 = time.perf_counter()
                    response = litellm.completion(
                        model=model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    self.stats["api_calls"] += 1
                    self.stats["total_latency_ms"] += elapsed_ms
                    usage = getattr(response, "usage", None)
                    if usage:
                        self.stats["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0))
                        self.stats["completion_tokens"] += int(getattr(usage, "completion_tokens", 0))
                    return response.choices[0].message.content
                except Exception:
                    continue
        except ImportError:
            pass
        return None

    def _local_dictionary_segment(self, syllables: List[str]) -> List[str]:
        """
        Longest-match syllable bundler backed by data/dictionary.txt and underthesea/pyvi.
        Returns B/I tags of length len(syllables).
        """
        if not syllables:
            return []

        raw_sentence = " ".join(syllables)
        try:
            from underthesea import word_tokenize

            ut_words = ["_".join(w.strip().split()) for w in word_tokenize(raw_sentence) if w.strip()]
            ut_bi = [
                "B" if idx == 0 else "I"
                for w in ut_words
                for idx, _ in enumerate(w.split("_"))
            ]
            if len(ut_bi) == len(syllables):
                return ut_bi
        except Exception:
            pass

        # Longest-match forward scan up to 4-syllable compounds in self.vocab
        tags = ["B"] * len(syllables)
        i = 0
        n = len(syllables)
        while i < n:
            matched_len = 1
            for span_len in range(min(4, n - i), 1, -1):
                candidate = "_".join(s.lower() for s in syllables[i : i + span_len])
                if candidate in self.vocab:
                    matched_len = span_len
                    break
            tags[i] = "B"
            for offset in range(1, matched_len):
                tags[i + offset] = "I"
            i += matched_len
        return tags

    def segment(
        self,
        raw_sentence: str,
        ambiguous_indices: Optional[List[int]] = None,
    ) -> str:
        syllables = raw_sentence.strip().split()
        if not syllables:
            return ""

        messages = build_ws_messages(
            raw_sentence=raw_sentence,
            strategy=self.strategy,
            ambiguous_indices=ambiguous_indices,
        )
        llm_reply = self._call_chat_completion(messages)
        if llm_reply is not None:
            return parse_llm_ws_output(llm_reply, raw_sentence)

        t0 = time.perf_counter()
        self.stats["local_fallback_calls"] += 1
        bi_tags = self._local_dictionary_segment(syllables)
        self.stats["total_latency_ms"] += (time.perf_counter() - t0) * 1000.0
        return " ".join(ws_tags_to_segmented_words(syllables, bi_tags))

    def tag_segmented(self, segmented_sentence: str) -> List[Tuple[str, str]]:
        words = segmented_sentence.strip().split()
        if not words:
            return []

        messages = build_pos_messages(segmented_sentence=segmented_sentence, strategy=self.strategy)
        llm_reply = self._call_chat_completion(messages)
        if llm_reply is not None:
            tagged = parse_llm_pos_output(llm_reply, words)
            return [(w, normalize_pos_tag(t)) for w, t in tagged]

        self.stats["local_fallback_calls"] += 1
        try:
            from pyvi import ViPosTagger

            _, tags = ViPosTagger.postagging(" ".join(words))
            if len(tags) == len(words):
                return [(w, normalize_pos_tag(t)) for w, t in zip(words, tags)]
        except Exception:
            pass

        return [(w, "CH" if all(not c.isalnum() for c in w) else "N") for w in words]
