# Vietnamese Word Segmentation & POS Tagging — Experiment Report

## 1. Overview
This project implements and benchmarks a two-stage Vietnamese NLP pipeline for **Word Segmentation (`B`/`I` syllable boundary classification)** and **Part-of-Speech (POS) Tagging** across three architectures:
- **HMM Baseline (`hmm`)**: Categorical Hidden Markov Model with Laplace smoothing and Viterbi decoding (`hmm/train.py`, `hmm/viterbi.py`, `pos/tagger.py`).
- **PhoBERT (`phobert`)**: Fine-tuned `vinai/phobert-base` token classification models with BPE subword label alignment (`llm/phobert.py`).
- **Hybrid (`hybrid_hmm_phobert`)**: Fast HMM Viterbi decoding with PhoBERT boundary refinement on Out-of-Vocabulary (OOV) syllables (`pipeline.py`).

---

## 2. Dataset Summary (HumanSignal / Label Studio Export)
- **Source File**: `vietnamese-sentences-at-2026-10-25-20-39-3af1ee33.json`
- **Total Annotated Sentences**: **119 sentences**
  - **Train Split (80%)**: 95 sentences (1,894 segmented words)
  - **Validation Split (10%)**: 11 sentences (262 segmented words)
  - **Test Split (10%)**: 13 sentences (237 segmented words)
- **Total Segmented Words (POS Tokens)**: **2,393 words**
- **Total Syllables (`B`/`I` Tokens)**: **3,318 syllables**

### POS Tag Distribution Across Splits
- **Train (1,894 words)**: `N` (609), `V` (283), `A` (220), `F` (190), `E` (190), `R` (103), `CH` (95), `M` (67), `C` (60), `Np` (36), `P` (23), `X` (18)
- **Validation (262 words)**: `N` (71), `V` (46), `A` (28), `E` (26), `F` (25), `C` (14), `R` (12), `CH` (12), `M` (10), `P` (9), `Np` (8), `X` (1)
- **Test (237 words)**: `N` (73), `V` (35), `A` (25), `E` (24), `F` (22), `CH` (18), `M` (13), `R` (11), `C` (9), `Np` (5), `X` (1), `P` (1)

---

## 3. Benchmark Results

### 3.1 Latest Run: 119 Annotated Sentences (10 Epochs, MPS)

| Model | WS Precision | WS Recall | WS Exact F1 | WS OOV Recall | Syllable B/I Acc | POS Token Acc | POS Macro-F1 | POS OOV Acc | End-to-End Joint F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HMM (`hmm`)** | 63.04% | 73.42% | 67.84% | 32.86% | 81.58% | 75.53% | **59.73%** | 62.86% | 55.36% |
| **PhoBERT (`phobert`)** | **83.46%** | **89.45%** | **86.35%** | **71.43%** | **93.27%** | **75.95%** | 46.27% | **78.57%** | **66.40%** |
| **Hybrid (`hybrid_hmm_phobert`)** | 75.79% | 80.59% | 78.12% | 57.14% | 88.60% | **75.95%** | 46.27% | **78.57%** | 58.49% |

### 3.2 Earlier Baseline Run: 45 Annotated Sentences (5 Epochs, MPS)

| Model | WS Precision | WS Recall | WS Exact F1 | WS OOV Recall | Syllable B/I Acc | POS Token Acc | POS Macro-F1 | POS OOV Acc | End-to-End Joint F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HMM (`hmm`)** | 55.17% | 63.37% | 58.99% | 36.36% | 76.82% | **67.33%** | **52.16%** | **38.64%** | **44.24%** |
| **PhoBERT (`phobert`)** | 63.87% | **75.25%** | **69.09%** | **56.82%** | **84.11%** | 41.58% | 15.59% | 36.36% | 32.73% |
| **Hybrid (`hybrid_hmm_phobert`)** | **66.06%** | 71.29% | 68.57% | 50.00% | **84.11%** | 41.58% | 15.59% | 36.36% | 30.48% |

### 3.3 PhoBERT 10-Epoch Fine-Tuning Loss Progression (119-Sentence Dataset)

| Epoch | WS Train Loss | WS Val Loss | POS Train Loss | POS Val Loss |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.6287 | 0.5785 | 2.3287 | 2.1847 |
| **2** | 0.4986 | 0.4684 | 2.0384 | 2.0368 |
| **3** | 0.4026 | 0.3751 | 1.8840 | 1.9204 |
| **4** | 0.3172 | 0.2874 | 1.7501 | 1.7971 |
| **5** | 0.2478 | 0.2259 | 1.5903 | 1.6556 |
| **6** | 0.2045 | 0.2133 | 1.4427 | 1.5066 |
| **7** | 0.1594 | 0.1921 | 1.3018 | 1.3801 |
| **8** | 0.1348 | 0.1913 | 1.1519 | 1.2735 |
| **9** | 0.1116 | **0.1797** | 1.0498 | 1.1622 |
| **10** | **0.0893** | 0.1870 | **0.9461** | **1.0907** |

---

## 4. Key Findings & Analysis
1. **Substantial Word Segmentation Gain (+18.51 F1 points over HMM)**:
   - PhoBERT achieves **86.35% exact word-boundary F1** and **93.27% syllable `B`/`I` accuracy**, outperforming the HMM baseline (`67.84%` F1).
2. **Superior Out-of-Vocabulary (OOV) Generalization**:
   - On unseen words in the test split, PhoBERT reaches **71.43% WS OOV Recall** (more than 2x HMM's `32.86%`) and **78.57% POS OOV Accuracy** (vs. HMM's `62.86%`).
3. **Scaling Impact (45 $\rightarrow$ 119 Annotated Sentences)**:
   - Increasing the HumanSignal dataset from 45 sentences (835 words) to 119 sentences (2,393 words) improved PhoBERT Word Segmentation F1 from `69.09%` $\rightarrow$ `86.35%`, POS Token Accuracy from `41.58%` $\rightarrow$ `75.95%`, and End-to-End Joint F1 from `32.73%` $\rightarrow$ `66.40%`.
4. **Data Recommendations for Next Iteration**:
   - Rare POS tags (`P`, `X`, `Np`, `C`, `M`) currently have fewer than 10 examples in the 10% test split (with `P` and `X` appearing once each), which penalizes unweighted POS Macro-F1 (`46.27%`) despite high overall POS Token Accuracy (`75.95%`).
   - Merging duplicate punctuation labels (`CH` and `F`) and expanding the labeled dataset to **~250–300 sentences** will balance rare POS classes and push both Word Segmentation F1 (>92%) and POS Macro-F1 significantly higher.

---

## 5. Sample Inference Output
```text
Raw input:      Ẩm thực làm thăng hoa tình yêu trong cuộc sống .
Segmented:      Ẩm_thực làm thăng_hoa tình_yêu trong cuộc_sống .
POS tagged:     Ẩm_thực/N làm/V thăng_hoa/V tình_yêu/N trong/E cuộc_sống/N ./F
```

---

## 6. Reproducing the Experiment

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Convert HumanSignal / Label Studio JSON export into splits
python main.py prepare-data --label-studio data/processed/label_studio_export.json --no-silver

# 3. Train HMM and PhoBERT models (WS + POS)
python main.py train --model all --epochs 10

# 4. Evaluate on held-out test set and write logs/experiment_results.csv
python main.py evaluate

# 5. Run live demo on custom text
python main.py demo --text "Ẩm thực làm thăng hoa tình yêu trong cuộc sống ."
```
