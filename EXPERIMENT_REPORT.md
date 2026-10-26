# Vietnamese Word Segmentation & POS Tagging — Experiment Report

## 1. Overview
This project implements and benchmarks a multi-tier Vietnamese NLP pipeline for **Word Segmentation (`B`/`I` syllable boundary classification)** and **Part-of-Speech (POS) Tagging** across six architectures:
1. **HMM Baseline (`hmm`)**: Categorical Hidden Markov Model with Laplace smoothing and Viterbi decoding (`hmm/train.py`, `hmm/viterbi.py`, `pos/tagger.py`).
2. **PyVi Baseline (`pyvi_baseline`)**: Off-the-shelf CRF Vietnamese tokenizer and POS tagger (`pyvi`).
3. **Underthesea Baseline (`underthesea_baseline`)**: Off-the-shelf Vietnamese NLP toolkit (`underthesea`).
4. **PhoBERT (`phobert`)**: Fine-tuned `vinai/phobert-base` token classification models with BPE subword label alignment (`llm/phobert.py`).
5. **Hybrid (`hybrid_hmm_phobert`)**: Confidence-based asymmetric routing combining fast HMM Viterbi decoding on high-confidence syllables ($\ge 0.85$) with PhoBERT refinement on uncertain/OOV boundaries (`pipeline.py`).
6. **3-Tier Asymmetric Cascade (`cascade_3tier`)**: Routes high-confidence syllables to HMM (Tier 1), uncertain syllables to PhoBERT (Tier 2), and low-confidence PhoBERT boundaries ($< 0.70$) to the LLM / Dictionary module (`llm/client.py`, `llm/prompt.py`, `data/dictionary.txt`).

---

## 2. Dataset Summary (HumanSignal / Label Studio Export)
- **Source File**: `vietnamese-sentences-at-2026-10-25-20-39-3af1ee33.json`
- **Total Annotated Sentences**: **119 sentences**
  - **Train Split (80%)**: 95 sentences (1,894 segmented words)
  - **Validation Split (10%)**: 11 sentences (262 segmented words)
  - **Test Split (10%)**: 13 sentences (237 segmented words)
- **Total Segmented Words (POS Tokens)**: **2,393 words**
- **Total Syllables (`B`/`I` Tokens)**: **3,318 syllables**
- **Extracted Dictionary (`data/dictionary.txt`)**: **803 unique normalized Vietnamese words**
- **Tag Normalization**: Punctuation tags (`F` from HumanSignal and `CH` from unannotated punctuation gaps) are unified under `CH`.

---

## 3. Six-Way Benchmark Results (119 Annotated Sentences, 10 Epochs, MPS)

| Model | WS Precision | WS Recall | WS Exact F1 | WS OOV Recall | Syllable B/I Acc | POS Token Acc | POS Macro-F1 | POS OOV Acc | End-to-End Joint F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HMM (`hmm`)** | 63.04% | 73.42% | 67.84% | 32.86% | 81.58% | **81.43%** | **63.17%** | 62.86% | 58.48% |
| **PyVi (`pyvi_baseline`)** | 65.16% | 78.90% | 71.37% | 54.29% | 87.28% | 79.32% | 62.26% | 65.71% | 56.11% |
| **Underthesea (`underthesea_baseline`)** | 68.20% | 81.43% | 74.23% | 54.29% | 85.96% | 75.95% | 60.78% | 64.29% | 60.77% |
| **PhoBERT (`phobert`)** | **81.15%** | 83.54% | **82.33%** | **68.57%** | **90.94%** | 78.90% | 42.61% | **77.14%** | **67.78%** |
| **Hybrid (`hybrid_hmm_phobert`)** | 78.19% | 80.17% | 79.17% | 62.86% | 89.47% | 78.90% | 42.61% | **77.14%** | 64.58% |
| **3-Tier Cascade (`cascade_3tier`)** | 80.32% | **84.39%** | 82.30% | **68.57%** | 90.64% | 78.90% | 42.61% | **77.14%** | 65.84% |

---

## 4. Key Findings & Analysis
1. **PhoBERT & 3-Tier Cascade Outperform Off-the-Shelf Toolkits (`PyVi` & `Underthesea`)**:
   - Fine-tuned **PhoBERT (`82.33%` WS F1)** and the **3-Tier Asymmetric Cascade (`82.30%` WS F1, `84.39%` Recall)** outperform `underthesea` (`74.23%` WS F1), `pyvi` (`71.37%` WS F1), and `hmm` (`67.84%` WS F1) on your domain-specific corpus.
2. **Unifying Punctuation Tags (`F` $\rightarrow$ `CH`) Boosted POS & End-to-End Accuracy Across All Models**:
   - Merging duplicate punctuation labels raised HMM POS Accuracy from `75.53%` $\rightarrow$ **`81.43%`** (`63.17%` Macro-F1), PhoBERT POS Accuracy from `75.95%` $\rightarrow$ **`78.90%`**, and PhoBERT End-to-End Joint F1 from `66.40%` $\rightarrow$ **`67.78%`**.
3. **Asymmetric Cascade Matches Full PhoBERT F1 at Lower Compute Cost**:
   - `cascade_3tier` achieves **`82.30%` WS F1** (within `0.03%` of full PhoBERT's `82.33%`) and **beats full PhoBERT on WS Recall (`84.39%` vs. `83.54%`)**, while offloading high-confidence syllables to the lightweight HMM Viterbi tier.

---

## 5. Repository Structure & Modules
- `data/dataset_converter.py`: Converts HumanSignal / Label Studio JSON exports and raw text into train/val/test splits (`ws_splits.json`, `pos_splits.json`) and populates `data/dictionary.txt`.
- `hmm/train.py` & `hmm/viterbi.py`: Syllable-level `B`/`I` HMM trainer and Viterbi decoder with posterior margin confidence estimation (`segment_with_confidence`).
- `pos/tagger.py`: Word-level log-space Viterbi HMM POS tagger with word-shape OOV emission modeling.
- `llm/phobert.py`: PhoBERT feature extractor (`extract_phobert_features`) and BPE subword-aligned token classifiers (`PhoBERTWordSegmenter`, `PhoBERTPosTagger`) with softmax confidence scoring.
- `llm/prompt.py` & `llm/client.py`: Zero-shot, few-shot, and chain-of-thought prompt builders and `LLMWordSegmenterClient` supporting LiteLLM (`groq/llama-3.1-70b-versatile`, `gemini/gemini-1.5-flash`) with offline dictionary/toolkit fallback.
- `eval/metrics.py`: Exact word-boundary VLSP Precision/Recall/F1, OOV Recall, and joint span+POS evaluation.
- `pipeline.py` & `main.py`: End-to-end 3-tier asymmetric cascade pipeline and CLI runner.
- `test.py`: 10 automated unit tests covering all modules.

---

## 6. Reproducing the Experiment & Running Tests

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run unit test suite (10 tests)
python -m unittest test.py -v

# 3. Convert HumanSignal / Label Studio JSON export into splits & dictionary
python main.py prepare-data --label-studio data/processed/label_studio_export.json --no-silver

# 4. Train HMM and PhoBERT models (WS + POS)
python main.py train --model all --epochs 10

# 5. Evaluate all 6 models (HMM, PyVi, Underthesea, PhoBERT, Hybrid, 3-Tier Cascade)
python main.py evaluate

# 6. Run live demo with 3-tier cascade
python main.py demo --ws-mode cascade --pos-mode phobert --text "Ẩm thực làm thăng hoa tình yêu trong cuộc sống ."
```
