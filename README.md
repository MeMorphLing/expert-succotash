# NLP Coursework — 22F-3177

Four assignments from a Natural Language Processing course, tracing one continuous arc: **frequency counts → dense embeddings → sequential deep nets → transformers with explainability**. Each notebook is self-contained (dataset pull, preprocessing, modeling, evaluation) but they build on each other conceptually — this repo keeps them together because the comparisons only make sense read as a set.

**Author:** 22F-3177 · FAST NUCES, Karachi

---

## Repository Structure

```
.
├── 01-customer-feedback-nlp-pipeline/
│   └── 22F_3177_NLP_A_1.ipynb
├── 02-phishing-email-detection/
│   ├── 22F_3177_NLP_A_2.ipynb          # baseline suite (Kaggle dataset)
│   ├── 22F_3177_NLP_A_2.pdf            # exported report w/ full run output
│   └── phishing_modules_1_and_2.ipynb  # extended edition (HF dataset, +BiLSTM/Attention)
├── 03-word-embedding-evolution/
│   └── 22F_3177_NLP_A_3.ipynb
├── 04-transformer-explainability/
│   └── 22F-3177_A4.ipynb
└── README.md
```

---

## 01 — Customer Feedback NLP Pipeline

**Notebook:** `22F_3177_NLP_A_1.ipynb`
**Dataset:** `mmehul/ecommerce-reviews-sentiment` (HuggingFace, 3,000 balanced reviews — 1,000 each of positive/neutral/negative)

A full classical NLP pipeline end to end:

| Stage | What it does |
|---|---|
| Preprocessing | Clean → tokenize → stopword removal → lemmatize (~40–50% token reduction/review) |
| Feature engineering | Bag-of-Words vs. TF-IDF comparison |
| Sentiment | VADER (rule-based) vs. Naïve Bayes / Logistic Regression (ML) |
| Intent classification | Keyword-heuristic 5-class tagger (Complaint, Refund, Delivery, Query, Praise) |
| Topic modeling | NMF over the TF-IDF matrix → 5 latent topics |
| Evaluation | Precision/Recall/F1, confusion matrices — Weighted F1 selected as the primary metric for the class imbalance |
| Interface | Gradio app: input a review, get sentiment + intent + topic + keywords live |

**Findings:** TF-IDF beats BoW by ~2–4%; Logistic Regression on TF-IDF is the best performer overall and clearly outperforms VADER on the neutral class; sentiment and intent turn out to be only loosely correlated (positive-sentiment reviews can still carry a Delivery or Refund intent), which is the justification for keeping intent as a separate classifier.

---

## 02 — Phishing Email Detection

Two notebooks in this folder cover the same problem at different depth.

### Baseline suite — `22F_3177_NLP_A_2.ipynb`
**Dataset:** `naserabdullahalam/phishing-email-dataset` (Kaggle, via `kagglehub`) — 82,486 emails, strict 80/10/10 stratified split (65,988 / 8,249 / 8,249), fit-on-train-only vectorizers to guarantee no leakage.

Four models trained and compared head-to-head: **Logistic Regression + TF-IDF → Feedforward NN → Vanilla RNN → LSTM.**

| Model | Accuracy | Recall (Phishing) | Notes |
|---|---|---|---|
| Logistic Regression | 98.24% | 98.37% | Fast, interpretable baseline |
| FNN | 98.31% | 98.46% | Marginal gain over LR for real added cost |
| Vanilla RNN | 62.47% | 97.65% | Collapses — vanishing gradients, precision on legit mail cratered |
| **LSTM** | **98.47%** | **98.83%** | Deployed model — best recall, the metric that matters most here |

The notebook argues explicitly for **Recall (Phishing)** as the deployment metric over raw accuracy, since a missed phishing email (false negative) is far costlier than a legitimate email flagged as spam. Module 2 adds ROC/AUC, calibration curves (reliability diagrams), qualitative error analysis on the LSTM's misclassifications, and an adversarial out-of-distribution probe (a hand-crafted email mimicking a benign internal update with a hidden malicious link).

### Extended edition — `phishing_modules_1_and_2.ipynb`
**Dataset:** `ealvaradob/phishing-dataset` (HuggingFace, ~80K samples, near-balanced classes, full email bodies).

Same Module 1 / Module 2 structure, plus a fourth architecture — **Bidirectional LSTM + Attention** — and PyTorch instead of Keras for the neural models. Same no-leakage discipline (vectorizer/tokenizer fit only on train, fixed seed, identical split reused across all models for a fair comparison).

**`22F_3177_NLP_A_2.pdf`** is the exported report of the baseline notebook with all cell outputs, plots, confusion matrices, and the calibration diagram rendered — useful as a static reference without re-running anything.

---

## 03 — Word Embedding Evolution

**Notebook:** `22F_3177_NLP_A_3.ipynb`
**Dataset:** Twitter sentiment (binary positive/negative)

A controlled comparison of three eras of word representation on the same classification task:

1. **Frequency-based** — BoW, TF-IDF
2. **Prediction-based** — Word2Vec, GloVe (+ PCA/t-SNE visualization of the resulting vector space)
3. **Contextualized** — BERT (includes a polysemy case study showing the same word getting different embeddings across contexts)

| | BoW/TF-IDF | Word2Vec | BERT |
|---|---|---|---|
| Semantic awareness | ✗ | ✓ | ✓✓✓ |
| Context sensitivity | ✗ | ✗ | ✓✓✓ |
| Polysemy handling | ✗ | ✗ | ✓✓✓ |
| Train/inference speed | ✓✓✓ | ✓✓ | ✗ |
| Interpretability | ✓✓✓ | ✓ | ✗ |

**Takeaway:** performance climbs with representation sophistication, but so does compute cost — there's no universal winner, only a right tool per constraint (data volume, latency budget, interpretability requirement).

---

## 04 — Transformer Explainability

**Notebook:** `22F-3177_A4.ipynb`
**Dataset:** Amazon Polarity (HuggingFace) · **Model:** `bert-base-uncased`, fine-tuned for binary sentiment

The deepest notebook in the set — fine-tunes BERT, then spends most of its length on *why* it predicts what it predicts:

- **Attention analysis** — extracts and visualizes attention heatmaps from layer 1 (syntactic) vs. layer 12 (semantic), across multiple heads, plus CLS-token attention aggregated across all 12 layers
- **SHAP** — Partition explainer, 20 explained predictions
- **LIME** — 20 explained predictions on the same samples
- **SHAP vs. LIME comparison** — runtime, faithfulness (sufficiency test), stability (repeated-run variance), and token-level agreement between the two methods
- **Error analysis** — misclassified samples isolated and inspected for common failure patterns
- **Methodology flowchart** — end-to-end pipeline diagram generated for the writeup

This notebook is the natural endpoint of the repo's arc: 01 shows what's possible with no deep learning at all, 04 shows a fine-tuned transformer plus the tooling needed to justify trusting (or not trusting) its predictions.

---

## Environment & Setup

Each notebook installs its own dependencies in its first cell (`pip install` / `%pip install`), so they're runnable standalone in Colab or a local Jupyter environment. Consolidated dependency list across the repo:

```
pandas numpy scikit-learn matplotlib seaborn wordcloud
tensorflow torch transformers datasets
nltk vaderSentiment gradio kagglehub
shap lime bertviz ipywidgets
```

**Notes:**
- `22F_3177_NLP_A_2.ipynb` requires a Kaggle API token (`kagglehub` auth) to pull the phishing dataset.
- `22F-3177_A4.ipynb` and the BiLSTM/Attention model in `phishing_modules_1_and_2.ipynb` are meaningfully faster on GPU (T4/A100 in Colab); everything else runs fine on CPU.
- All notebooks fix `random_state=42` / equivalent seeds for reproducibility.

## Suggested `requirements.txt`

```
pandas
numpy
scikit-learn
tensorflow
torch
transformers
datasets
nltk
vaderSentiment
gradio
kagglehub
shap
lime
bertviz
matplotlib
seaborn
wordcloud
```

---

## License

Academic coursework — shared for reference and portfolio purposes.
