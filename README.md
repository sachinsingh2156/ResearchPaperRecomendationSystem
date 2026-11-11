# Research Paper Recommendation System
### Using Vector-Space Models and Linear-Algebraic Dimensionality Reduction

A course project for **Mathematical Foundations for Computer Science** (Group 05). It represents 9,956
research papers as vectors, recommends similar papers by **cosine similarity**, and studies whether
**Singular Value Decomposition (SVD)** dimensionality reduction improves effectiveness and efficiency over a
plain **TF-IDF** representation. The final submission adds a **least-squares learned metric** on top of the
SVD space and a **deep-learning extension** that feeds pretrained transformer embeddings through the same
linear-algebra pipeline.

> **Research question:** *Does linear-algebraic dimensionality reduction using SVD improve the
> effectiveness and computational efficiency of research-paper recommendation compared with the original
> TF-IDF representation?*

**Notebooks**

- [`notebooks/ML_vs_DL_Recommendation_Analysis.ipynb`](notebooks/ML_vs_DL_Recommendation_Analysis.ipynb) —
  the whole project organised as **Part 1: machine learning** (implementation, accuracy, analysis),
  **Part 2: deep learning** (implementation, accuracy, analysis) and **Part 3: comparative analysis**.
- [`notebooks/Research_Paper_Recommendation_System_Final.ipynb`](notebooks/Research_Paper_Recommendation_System_Final.ipynb) —
  the same models with a longer treatment of the mathematics and viva-style questions and answers.

- [`notebooks/Research_Paper_Recommendation_System.ipynb`](notebooks/Research_Paper_Recommendation_System.ipynb) —
  the intermediate (checkpoint 2) version, kept for reference together with its progress report
  [`Group_05.pdf`](Group_05.pdf). Its figures (400 queries, no validation/test split) are superseded by the two
  notebooks above.

**Dashboard:** `streamlit run dashboard.py` — an interactive dashboard built on the deep-learning model
(see [Dashboard](#dashboard) below).

## Results (final submission)

Means over **6,911 held-out test queries**; relevance = the two papers share at least one OpenAlex topic.
All settings were chosen on 3,000 separate validation queries.

| Approach | Dims | P@1 | P@5 | P@10 | MRR | nDCG@10 | Hit@10 | Query time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random ranking (expected) | – | 0.075 | 0.075 | 0.075 | – | – | 0.481 | – |
| A — TF-IDF (baseline) | 31,291 | 0.795 | 0.738 | 0.699 | 0.863 | 0.720 | 0.973 | ≈ 1.9 ms |
| B — SVD, k=300 | 300 | 0.795 | 0.742 | 0.712 | 0.854 | 0.729 | 0.956 | ≈ 0.03 ms |
| B — SVD, k=1000 | 1,000 | 0.818 | 0.759 | 0.721 | 0.875 | 0.741 | 0.967 | ≈ 0.3 ms |
| B+ — hybrid of A and B | 31,591 | 0.820 | 0.764 | 0.727 | 0.876 | 0.747 | 0.968 | ≈ 2 ms |
| C — SVD + least-squares metric | 200 | 0.841 | 0.816 | 0.799 | 0.878 | 0.809 | 0.940 | ≈ 0.02 ms |
| D — fusion of C and A | 31,491 | 0.855 | 0.821 | 0.801 | 0.891 | 0.813 | 0.954 | ≈ 2 ms |
| E — transformer embeddings (no labels) | 1,920 | 0.923 | 0.882 | 0.852 | 0.948 | 0.869 | **0.988** | ≈ 1 ms |
| **F — embeddings + SVD + least squares** | **300** | 0.922 | 0.894 | 0.874 | 0.942 | 0.885 | 0.975 | **≈ 0.03 ms** |
| **G — fusion of F and E** | 2,220 | **0.929** | **0.902** | **0.881** | **0.948** | **0.892** | 0.982 | ≈ 1 ms |

Metrics: **P@K** = share of the top K that is relevant; **MRR** = mean of 1/rank of the first relevant paper;
**nDCG@10** = rank-weighted precision; **Hit@10** = share of queries with at least one relevant paper in the top 10.
(Query times are from one laptop and vary between runs; the quality figures are reproducible — all seeds are fixed.)

**Conclusions**

- **Efficiency:** SVD cuts 31,291 sparse dimensions to a few hundred dense ones and makes a query one to two
  orders of magnitude faster.
- **Effectiveness, unsupervised:** k=200 matches TF-IDF on nDCG@10; k≥300 is significantly better on precision
  and nDCG; k=1000 is better on P@K, MRR and nDCG. Very small ranks (k≤100) are worse.
- **With least squares (C, D):** a ridge least-squares map on the SVD space, compressed by a second SVD, gives a
  200-dimensional representation that is ten points more precise than TF-IDF and ≈ 80× faster to query.
- **With deep-learning features (E, F, G):** pretrained transformer embeddings reach P@10 0.852 with no labels;
  pushing them, together with the SVD vector, through the same least-squares + SVD pipeline gives the best
  system (P@10 0.881, P@1 0.929, MRR 0.948, Hit@10 0.982) and a 300-dimensional variant that is more accurate
  and 6× smaller than the raw embeddings.
- **Why not 95% Precision@10:** only the label-leaking in-sample variant gets there (0.943) — that is
  over-fitting. Honestly evaluated, precision is limited by corpus coverage: for the 22% of queries that have
  ten close neighbours the best system scores 0.987; for sparsely covered subjects it cannot.

The supervised approaches (C, D, F, G) use topic labels of *other* papers only, under a strict
pairwise-exclusion cross-fitting protocol; a leakage audit is included in the notebook.

## Project layout

```
Math-project/
├── data/
│   ├── research_papers.csv          # 9,958 papers (OpenAlex): title, abstract, authors, year, venue, topics, domain
│   └── paper_embeddings.npz         # cached transformer embeddings of the papers (3 encoders, 35 MB)
├── notebooks/
│   ├── ML_vs_DL_Recommendation_Analysis.ipynb             # FINAL: Part 1 ML, Part 2 DL, Part 3 comparison (with outputs)
│   ├── Research_Paper_Recommendation_System_Final.ipynb   # FINAL: same models, maths-first presentation (with outputs)
│   └── Research_Paper_Recommendation_System.ipynb         # intermediate (checkpoint 2) version
├── models/
│   ├── deep_recommender.npz         # saved deep-learning model (created by train_deep_model.py)
│   └── deep_metrics.json            # its held-out accuracy, shown in the dashboard
├── train_deep_model.py              # trains, evaluates and saves the deep-learning model
├── dashboard.py                     # Streamlit dashboard on the deep-learning model
├── app.py                           # simple Streamlit search page (all approaches A–G)
├── Group_05.pdf                     # checkpoint-2 (intermediate) progress report
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

```bash
# 1. create environment (Python 3.12 recommended) and install deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. open a final notebook (Run All takes two to three minutes)
jupyter notebook notebooks/ML_vs_DL_Recommendation_Analysis.ipynb

# 3. launch the deep-learning dashboard
streamlit run dashboard.py

# (optional) the simple search page with all approaches A-G
streamlit run app.py
```

**Optional deep-learning packages.** The notebook loads the cached embeddings in `data/paper_embeddings.npz`, so it
runs without PyTorch. To recompute the embeddings, or to use Approach G for *new free-text queries* in the notebook
and the app, also install:

```bash
pip install torch sentence-transformers
```

The first use downloads about 1 GB of pretrained model weights from Hugging Face. Without these packages the
notebook and the app fall back to Approach D for free-text search.

## Dashboard

`dashboard.py` is a five-tab Streamlit dashboard driven by the saved deep-learning model
(three pretrained transformer encoders → learned 300-dimensional metric → score fusion).

| Tab | What it does |
|---|---|
| Search | free-text query → ranked papers, predicted topics, domain mix, timing; filters for domain and year |
| Similar papers | pick any paper of the corpus → most similar papers, with a ✓ where they share an OpenAlex topic |
| Model performance | held-out accuracy, comparison with the classical models, accuracy by domain, what limits accuracy, leakage audit |
| Corpus map | 2-D map of the corpus in the embedding space, with your last search results marked |
| How it works | the pipeline, the encoders, training configuration and limitations |

Held-out accuracy of the dashboard's model ("Deep fusion", 6,911 test queries): **Precision@10 0.878,
Precision@1 0.930, MRR 0.950, Hit@10 0.983**. It is the pure deep-learning pipeline (no TF-IDF or SVD features),
so it is 0.003 below approach G of the notebooks, which also uses the SVD features.

```bash
pip install torch sentence-transformers     # needed for free-text search (encodes the query)
python train_deep_model.py                  # optional: retrain and re-evaluate (about 30 s); the dashboard does this itself if models/ is missing
streamlit run dashboard.py
```

The dashboard needs `data/paper_embeddings.npz`. Without PyTorch it still runs, but only the *Similar papers*,
*Model performance*, *Corpus map* and *How it works* tabs are usable.

## What the final notebook covers (mapped to the proposal objectives)

| § | Content | Objective |
|---|---|---|
| 1–2 | Dataset, exploratory analysis, preprocessing (lower-casing, tokenisation, stop-words, Porter stemming) | 1 |
| 3 | The mathematics on a toy corpus — SVD computed by hand from an eigen-decomposition, Eckart–Young verified | 2, 4 |
| 4 | TF-IDF term–document matrix `A ∈ ℝ^(9956×31291)` | 2 |
| 5 | Approach A — TF-IDF + cosine similarity | 3 |
| 6 | SVD (`A = UΣVᵀ`): randomised computation, six numerical verifications, spectrum, meaning of latent directions | 4 |
| 7 | Approach B — recommendation in the rank-k latent space | 5 |
| 8–9 | Evaluation framework (validation/test split, P@K, R@K, MRR, nDCG@10, Hit@10) and tuning of A, B, B+ | 6 |
| 10–11 | Approach C (least-squares learned metric, leakage-free cross-fitting) and Approach D (fusion) | extension |
| 12 | Deep-learning extension: pretrained transformer embeddings (Approaches E, F, G) | extension |
| 13–14 | Final test results with confidence intervals, significance tests, leakage audit, analysis of what limits precision; efficiency | 6 |
| 15–16 | Latent-space visualisation; final model, free-text search with predicted topics, live widget | 7 |
| 17, A, B | Conclusion and limitations; code map; viva-style questions and answers | – |

## Notes

- **Relevance ground truth:** the dataset has no human relevance labels, so two papers are treated as
  relevant if they share at least one of their 3 `topics`. This is a proxy, not a human judgement.
- **Why Recall@K is small:** each query paper has hundreds of topically-related papers in this corpus, so
  only a tiny fraction can appear in the top-K. Precision@K, MRR, nDCG and Hit@10 are the informative signals.
- **Change from the intermediate report:** the earlier figures used 400 queries with no validation/test split.
  With all queries, SVD at k=300 beats TF-IDF on precision and nDCG but not on MRR; it is ahead on those
  metrics only from k≈500–1000.
- **Which approaches use labels:** C, D, F and G are trained on topic labels (never those of the two papers
  being compared). A, B, B+ and E use no labels.
- **Pretrained encoders:** E, F and G rely on transformer models trained by others on large external text
  collections. They were not trained on this corpus's topic labels; they are reported separately from the
  purely linear-algebraic approaches A–D.
