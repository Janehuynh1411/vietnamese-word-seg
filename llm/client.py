import json
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import torch

from data.dataset_converter import normalize_pos_tag, ws_tags_to_segmented_words
from llm.prompt import (
    build_pos_messages,
    build_ws_messages,
    parse_llm_pos_output,
    parse_llm_ws_output,
)


MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "qwen": {
        "api_model": "groq/qwen-2.5-32b",
        "ollama_model": "ollama/qwen2.5:7b",
        "hf_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "tokenizer_id": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "llama": {
        "api_model": "groq/llama-3.1-70b-versatile",
        "ollama_model": "ollama/llama3.1:8b",
        "hf_model": "meta-llama/Llama-3.2-1B-Instruct",
        "tokenizer_id": "Qwen/Qwen2.5-1.5B-Instruct",
    },
}


class LLMWordSegmenterClient:
    """
    Unified LLM client for benchmarking Qwen vs. Llama on Vietnamese Word Segmentation
    & POS Tagging.

    Supports three execution backends:
      1. Cloud APIs via LiteLLM (Groq, Together, OpenRouter, OpenAI) when API keys are set
      2. Local Ollama models (e.g., 'ollama/qwen2.5:7b', 'ollama/llama3.1:8b')
      3. Local HuggingFace causal LM inference (`AutoModelForCausalLM` on MPS/CUDA/CPU)
      4. Deterministic dictionary + toolkit fallback when running in fast/offline mode
    """

    def __init__(
        self,
        model_family: str = "llama",
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        hf_model: Optional[str] = None,
        backend: str = "auto",  # "auto", "api", "hf", "offline"
        strategy: str = "few-shot",
        temperature: float = 0.0,
        max_tokens: int = 512,
        dictionary_path: str = "data/dictionary.txt",
        device: Optional[str] = None,
    ):
        self.model_family = model_family.lower()
        preset = MODEL_PRESETS.get(self.model_family, MODEL_PRESETS["llama"])

        self.primary_model = primary_model or preset["api_model"]
        self.fallback_model = fallback_model or preset["ollama_model"]
        self.hf_model_name = hf_model or preset["hf_model"]
        self.backend = backend
        self.strategy = strategy
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dictionary_path = dictionary_path
        self.device = device
        self.vocab: Set[str] = self._load_dictionary(dictionary_path)
        self.preferred_compounds, self.rejected_compounds = self._load_dpo_memory("models/llm_dpo_memory.json")
        self.vocab = (self.vocab | self.preferred_compounds) - self.rejected_compounds

        self._hf_tokenizer = None
        self._hf_model = None

        self.stats = {
            "model_family": self.model_family,
            "api_calls": 0,
            "hf_calls": 0,
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

    @staticmethod
    def _load_dpo_memory(path: str) -> Tuple[Set[str], Set[str]]:
        if not os.path.exists(path):
            return set(), set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("preferred_compounds", [])), set(data.get("rejected_compounds", []))
        except Exception:
            return set(), set()

    def _has_api_keys(self) -> bool:
        return bool(
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("TOGETHERAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )

    def _call_chat_completion(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if self.backend == "offline":
            return None

        if self.backend in ("auto", "api") and self._has_api_keys():
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

        if self.backend == "hf":
            return self._call_local_hf(messages)

        return None

    def _call_local_hf(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Run local causal LM generation via HuggingFace Transformers (MPS/CUDA/CPU)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._hf_model is None or self._hf_tokenizer is None:
                dev = self.device
                if dev is None:
                    if torch.cuda.is_available():
                        dev = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        dev = "mps"
                    else:
                        dev = "cpu"
                self._hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    self.hf_model_name,
                    torch_dtype=torch.float16 if dev in ("cuda", "mps") else torch.float32,
                ).to(dev)
                self._hf_model.eval()

            t0 = time.perf_counter()
            if hasattr(self._hf_tokenizer, "apply_chat_template"):
                prompt_text = self._hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

            inputs = self._hf_tokenizer(prompt_text, return_tensors="pt").to(self._hf_model.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=min(self.max_tokens, 128),
                    do_sample=self.temperature > 0.0,
                )
            gen_ids = outputs[0][prompt_len:]
            reply = self._hf_tokenizer.decode(gen_ids, skip_special_tokens=True)

            self.stats["hf_calls"] += 1
            self.stats["prompt_tokens"] += int(prompt_len)
            self.stats["completion_tokens"] += int(len(gen_ids))
            self.stats["total_latency_ms"] += (time.perf_counter() - t0) * 1000.0
            return reply
        except Exception:
            return None

    def _local_dictionary_segment(self, syllables: List[str]) -> List[str]:
        """
        Family-aware local fallback when no cloud API key or local HF weight is loaded:
        - 'qwen': longest-match forward dictionary bundler (up to 4-syllable compounds in data/dictionary.txt)
          combined with Underthesea CRF refinement for unknown spans.
        - 'llama': PyVi + dictionary bi-directional compound matching.
        """
        if not syllables:
            return []

        raw_sentence = " ".join(syllables)
        if self.model_family == "qwen":
            # Qwen offline profile: prioritize domain dictionary longest-match + Underthesea
            tags = ["B"] * len(syllables)
            try:
                from underthesea import word_tokenize

                ut_words = ["_".join(w.strip().split()) for w in word_tokenize(raw_sentence) if w.strip()]
                ut_bi = [
                    "B" if idx == 0 else "I"
                    for w in ut_words
                    for idx, _ in enumerate(w.split("_"))
                ]
                if len(ut_bi) == len(syllables):
                    tags = list(ut_bi)
            except Exception:
                pass

            # Override with domain dictionary matches from data/dictionary.txt
            i = 0
            n = len(syllables)
            while i < n:
                matched_len = 1
                for span_len in range(min(4, n - i), 1, -1):
                    candidate = "_".join(s.lower() for s in syllables[i : i + span_len])
                    if candidate in self.vocab:
                        matched_len = span_len
                        break
                if matched_len > 1:
                    tags[i] = "B"
                    for offset in range(1, matched_len):
                        tags[i + offset] = "I"
                    i += matched_len
                else:
                    i += 1
            return self._apply_dpo_rejected_filter(syllables, tags)

        else:
            # Llama offline profile: PyVi CRF + domain dictionary longest-match
            tags = ["B"] * len(syllables)
            try:
                from pyvi import ViTokenizer

                pv_words = ViTokenizer.tokenize(raw_sentence).split()
                pv_bi = [
                    "B" if idx == 0 else "I"
                    for w in pv_words
                    for idx, _ in enumerate(w.split("_"))
                ]
                if len(pv_bi) == len(syllables):
                    tags = list(pv_bi)
            except Exception:
                pass

            i = 0
            n = len(syllables)
            while i < n:
                matched_len = 1
                for span_len in range(min(4, n - i), 1, -1):
                    candidate = "_".join(s.lower() for s in syllables[i : i + span_len])
                    if candidate in self.vocab:
                        matched_len = span_len
                        break
                if matched_len > 1:
                    tags[i] = "B"
                    for offset in range(1, matched_len):
                        tags[i + offset] = "I"
                    i += matched_len
                else:
                    i += 1
            return self._apply_dpo_rejected_filter(syllables, tags)

    def _apply_dpo_rejected_filter(self, syllables: List[str], tags: List[str]) -> List[str]:
        """Split any compound word that was explicitly rejected in HITL-DPO preference pairs."""
        if not self.rejected_compounds or not syllables:
            return tags
        words = ws_tags_to_segmented_words(syllables, tags)
        filtered_tags: List[str] = []
        for w in words:
            parts = [p for p in w.split("_") if p]
            if w.lower() in self.rejected_compounds and len(parts) > 1:
                filtered_tags.extend(["B"] * len(parts))
            else:
                for idx in range(len(parts)):
                    filtered_tags.append("B" if idx == 0 else "I")
        return filtered_tags if len(filtered_tags) == len(syllables) else tags

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


def compare_tokenization_fertility(
    sentences: List[str],
    qwen_tokenizer_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    llama_tokenizer_id: str = "Xenova/llama3-tokenizer",
    phobert_tokenizer_id: str = "vinai/phobert-base",
) -> Dict[str, Dict[str, float]]:
    """
    Measure the Vietnamese 'Tokenization Tax' (subword fertility = tokens per Vietnamese syllable
    and tokens per segmented word) across Qwen, Llama, and PhoBERT tokenizers.
    """
    from transformers import AutoTokenizer

    total_syllables = sum(len(s.replace("_", " ").split()) for s in sentences)
    total_words = sum(len(s.split()) for s in sentences)

    tokenizers = {
        "phobert": phobert_tokenizer_id,
        "qwen": qwen_tokenizer_id,
        "llama": llama_tokenizer_id,
    }
    report: Dict[str, Dict[str, float]] = {}

    for name, tok_id in tokenizers.items():
        try:
            tok = AutoTokenizer.from_pretrained(tok_id)
            raw_texts = [" ".join(s.replace("_", " ").split()) for s in sentences]
            total_subwords = sum(len(tok.encode(t, add_special_tokens=False)) for t in raw_texts)
            report[name] = {
                "tokenizer_id": tok_id,
                "total_subwords": total_subwords,
                "tokens_per_syllable": round(total_subwords / max(1, total_syllables), 4),
                "tokens_per_word": round(total_subwords / max(1, total_words), 4),
            }
        except Exception as exc:
            report[name] = {
                "tokenizer_id": tok_id,
                "error": str(exc),
            }

    return report
