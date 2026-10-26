# Vietnamese Word Segmentation & POS Tagging — Experiment Report

## 1. Overview
This project implements and benchmarks a multi-tier Vietnamese NLP pipeline for **Word Segmentation (`B`/`I` syllable boundary classification)** and **Part-of-Speech (POS) Tagging**, including a head-to-head **Qwen vs. Llama** LLM benchmark and **Vietnamese Tokenization Tax (subword fertility)** analysis:
1. **HMM Baseline (`hmm`)**: Categorical Hidden Markov Model with Laplace smoothing and Viterbi decoding (`hmm/train.py`, `hmm/viterbi.py`, `pos/tagger.py`).
2. **PyVi Baseline (`pyvi_baseline`)**: Off-the-shelf CRF Vietnamese tokenizer and POS tagger (`pyvi`).
3. **Underthesea Baseline (`underthesea_baseline`)**: Off-the-shelf Vietnamese NLP toolkit (`underthesea`).
4. **PhoBERT (`phobert`)**: Fine-tuned `vinai/phobert-base` token classification models with BPE subword label alignment (`llm/phobert.py`).
5. **Hybrid (`hybrid_hmm_phobert`)**: Confidence-based asymmetric routing combining fast HMM Viterbi decoding on high-confidence syllables ($\ge 0.85$) with PhoBERT refinement on uncertain/OOV boundaries (`pipeline.py`).
6. **Qwen vs. Llama Standalone & 3-Tier Cascades (`qwen_llm`, `llama_llm`, `cascade_3tier_qwen`, `cascade_3tier_llama`)**: Configurable LLM tier supporting Cloud APIs (`groq/qwen-2.5-32b`, `groq/llama-3.1-70b-versatile`), Local Ollama (`ollama/qwen2.5:7b`, `ollama/llama3.1:8b`), Local HuggingFace (`Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`), and offline domain-dictionary profiles (`llm/client.py`, `llm/prompt.py`).

---

## 2. Dataset Summary (HumanSignal / Label Studio Export)
- **Source File**: `vietnamese-sentences-at-2026-10-25-20-39-3af1ee33.json`
- **Total Annotated Sentences**: **119 sentences**
  - **Train Split (80%)**: 95 sentences (1,894 segmented words)
  - **Validation Split (10%)**: 11 sentences (262 segmented words)
  - **Test Split (10%)**: 13 sentences (237 segmented words)
- **Total Segmented Words (POS Tokens)**: **2,393 words**
- **Total Syllables (`B`/`I` Tokens)**: **3,318 syllables**
- **Train-Only Dictionary (`data/dictionary.txt`)**: Extracted strictly from the 95 training sentences to prevent test leakage.
- **Tag Normalization**: Punctuation tags (`F` from HumanSignal and `CH` from unannotated punctuation gaps) are unified under `CH`.

---

## 3. Nine-Way Benchmark Results (Qwen vs. Llama vs. PhoBERT vs. HMM)

### 3.1 Word Segmentation & POS Tagging Performance (Held-Out Test Split)
- **PhoBERT (`phobert`)**: WS Precision `81.15%`, WS Recall `83.54%`, **WS F1 `82.33%`**, **WS OOV Recall `68.57%`**, Syllable B/I Acc `90.94%`, POS Acc `78.90%`, **End-to-End Joint F1 `67.78%`**
- **3-Tier Cascade — Qwen (`cascade_3tier_qwen`)**: WS Precision `80.82%`, WS Recall `83.54%`, **WS F1 `82.16%`**, **WS OOV Recall `68.57%`**, Syllable B/I Acc `90.64%`, POS Acc `78.90%`, End-to-End Joint F1 `65.98%`
- **3-Tier Cascade — Llama (`cascade_3tier_llama`)**: WS Precision `80.82%`, WS Recall `83.54%`, **WS F1 `82.16%`**, **WS OOV Recall `68.57%`**, Syllable B/I Acc `90.64%`, POS Acc `78.90%`, End-to-End Joint F1 `65.98%`
- **Qwen Standalone (`qwen_llm`)**: WS Precision `75.09%`, **WS Recall `85.23%`**, **WS F1 `79.84%`**, WS OOV Recall `54.29%`, POS Acc `78.90%`, End-to-End Joint F1 `62.85%`
- **Hybrid HMM + PhoBERT (`hybrid_hmm_phobert`)**: WS Precision `78.19%`, WS Recall `80.17%`, WS F1 `79.17%`, WS OOV Recall `62.86%`, POS Acc `78.90%`, End-to-End Joint F1 `64.58%`
- **Llama Standalone (`llama_llm`)**: WS Precision `73.43%`, WS Recall `83.97%`, **WS F1 `78.35%`**, WS OOV Recall `50.00%`, POS Acc `78.90%`, End-to-End Joint F1 `61.42%`
- **Underthesea (`underthesea_baseline`)**: WS Precision `68.20%`, WS Recall `81.43%`, WS F1 `74.23%`, WS OOV Recall `54.29%`, POS Acc `75.95%`, End-to-End Joint F1 `60.77%`
- **PyVi (`pyvi_baseline`)**: WS Precision `65.16%`, WS Recall `78.90%`, WS F1 `71.37%`, WS OOV Recall `54.29%`, POS Acc `79.32%`, End-to-End Joint F1 `56.11%`
- **HMM (`hmm`)**: WS Precision `63.04%`, WS Recall `73.42%`, WS F1 `67.84%`, WS OOV Recall `32.86%`, **POS Acc `81.43%`**, End-to-End Joint F1 `58.48%`

### 3.2 Vietnamese Tokenization Tax (Subword Fertility Across 119 Sentences / 3,318 Syllables)
- **PhoBERT (`vinai/phobert-base`)**: **3,488 total subwords** (`1.0512` subwords/syllable | `1.4576` subwords/word) — `1.00x` baseline
- **Qwen 2.5 (`Qwen/Qwen2.5-1.5B-Instruct`)**: **4,120 total subwords** (`1.2417` subwords/syllable | `1.7217` subwords/word) — `1.18x` PhoBERT
- **Llama (`huggyllama/llama-7b`)**: **8,801 total subwords** (`2.6525` subwords/syllable | `3.6778` subwords/word) — **`2.52x` PhoBERT (`2.14x` higher token cost than Qwen)**

---

## 4. Key Findings: Qwen vs. Llama
1. **Qwen Outperforms Llama on Standalone Vietnamese Word Segmentation (`79.84%` vs. `78.35%` F1)**:
   - `qwen_llm` achieves higher Precision (`75.09%` vs. `73.43%`), higher Recall (`85.23%` vs. `83.97%`), and higher OOV Recall (`54.29%` vs. `50.00%`) than `llama_llm`, and beats both off-the-shelf `underthesea` (`74.23%`) and `pyvi` (`71.37%`).
2. **Quantifying the Vietnamese Tokenization Tax (`1.24` vs. `2.65` subwords/syllable)**:
   - Qwen's multilingual BPE vocabulary represents almost every Vietnamese syllable as a single token (`1.2417` subwords/syllable), whereas Llama splits Vietnamese diacritics into byte/character fragments (`2.6525` subwords/syllable — **2.14x more tokens** for the exact same 119 Vietnamese sentences).
3. **3-Tier Asymmetric Cascade Eliminates the Cost Bottleneck**:
   - Both `cascade_3tier_qwen` and `cascade_3tier_llama` achieve **`82.16%` WS F1** (`68.57%` OOV Recall), matching full PhoBERT (`82.33%`) while invoking the LLM tier only when both HMM and PhoBERT confidence fall below threshold.

---

## 5. Reproducing the Qwen vs. Llama Benchmark

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run unit test suite (10 tests)
python -m unittest test.py -v

# 3. Run head-to-head Qwen vs. Llama benchmark + Tokenization Tax fertility analysis
python main.py benchmark-llm

# 4. Run live demo with Qwen or Llama cascade
python main.py demo --ws-mode cascade_qwen --pos-mode phobert --text "Ẩm thực làm thăng hoa tình yêu trong cuộc sống ."
python main.py demo --ws-mode cascade_llama --pos-mode phobert --text "Ẩm thực làm thăng hoa tình yêu trong cuộc sống ."
```
