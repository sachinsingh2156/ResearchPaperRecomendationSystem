"""
Train, evaluate and save the deep-learning recommender used by dashboard.py.

    python train_deep_model.py

Model (the pure deep-learning pipeline of the analysis notebook: models E, F- and their fusion):

    title + abstract
      -> three pretrained transformer encoders (MiniLM, SPECTER, BGE)          -> 1,920-d embedding   (model E)
      -> ridge least squares to topic space, SVD of the learned map            -> 300-d learned vector (model F-)
      -> fusion:  score = (1 - w) * cosine(learned vectors) + w * cosine(embeddings)

Inputs : data/research_papers.csv, data/paper_embeddings.npz (embedding cache created by the notebook)
Outputs: models/deep_recommender.npz   (learned projection, document vectors, topic map, 2-D map coordinates)
         models/deep_metrics.json      (held-out accuracy of every model, per-domain results, leakage audit)

Accuracy is measured exactly as in the notebooks: 3,000 validation queries to choose the fusion weight, 6,911 test
queries to report, and pairwise-exclusion cross-fitting so that no paper's own labels influence its own score.
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_PATH = "data/research_papers.csv"
EMB_PATH = "data/paper_embeddings.npz"
MODEL_PATH = "models/deep_recommender.npz"
METRICS_PATH = "models/deep_metrics.json"

EMB_KEYS = ["minilm", "specter", "bge"]
ENCODER_INFO = {
    "minilm": dict(label="MiniLM-L6", hf_name="sentence-transformers/all-MiniLM-L6-v2", max_len=256, dims=384,
                   layers=6, params="23 M", trained_on="over 1 billion general sentence pairs"),
    "specter": dict(label="SPECTER", hf_name="sentence-transformers/allenai-specter", max_len=384, dims=768,
                    layers=12, params="110 M", trained_on="scientific papers, with citation links as the training signal"),
    "bge": dict(label="BGE-base", hf_name="BAAI/bge-base-en-v1.5", max_len=384, dims=768,
                layers=12, params="109 M", trained_on="large-scale general retrieval pairs"),
}

SEED = 42
S_F = 2.0          # scale of the embeddings in the least-squares features (chosen on validation queries in the notebook)
LAMBDA_F = 3.0     # ridge strength (chosen on validation queries in the notebook)
R_F = 300          # rank of the learned metric (chosen on validation queries in the notebook)
K_FOLDS = 5
METRICS = ["P@1", "P@5", "P@10", "MRR", "nDCG@10", "Hit@10"]

# Best classical machine-learning model (TF-IDF -> SVD -> least squares, fused with TF-IDF), as reported on the same
# test queries in notebooks/ML_vs_DL_Recommendation_Analysis.ipynb.  Shown in the dashboard for context only.
CLASSICAL_BEST = {"P@1": 0.8552, "P@5": 0.8207, "P@10": 0.8014, "MRR": 0.8910, "nDCG@10": 0.8126, "Hit@10": 0.9540}


def build(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    t_start = time.time()

    # ------------------------------------------------------------------ data and cached embeddings
    df = pd.read_csv(DATA_PATH).dropna(subset=["abstract"]).reset_index(drop=True)
    N = len(df)
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(EMB_PATH + " is missing - run the analysis notebook once to create the embedding cache.")
    cache = np.load(EMB_PATH, allow_pickle=False)
    paper_ids = df["paper_id"].to_numpy(dtype=str)
    assert (cache["paper_id"] == paper_ids).all(), "embedding cache does not match the corpus rows"
    EMB = {k: normalize(cache[k].astype(np.float32)) for k in EMB_KEYS}
    E_raw = np.hstack([EMB[k] for k in EMB_KEYS])                      # (N, 1920); each block has unit length
    E_cat = E_raw / np.sqrt(len(EMB_KEYS))                            # unit rows; dot product = mean of the 3 cosines
    log("papers: %d   embedding matrix: %s" % (N, E_cat.shape))

    # ------------------------------------------------------------------ ground truth, splits, folds
    topic_lists = df["topics"].fillna("").map(lambda s: [t.strip().lower() for t in s.split(";") if t.strip()])
    topic_names = np.array(sorted({t for ts in topic_lists for t in ts}))
    topic_id = {t: i for i, t in enumerate(topic_names)}
    rows, cols = zip(*[(i, topic_id[t]) for i, ts in enumerate(topic_lists) for t in ts])
    T = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, len(topic_names)))
    REL = (T @ T.T).toarray() > 0
    np.fill_diagonal(REL, False)
    N_REL = REL.sum(axis=1)
    usable = np.where(N_REL > 0)[0]
    perm = np.random.default_rng(SEED).permutation(usable)
    VAL, TEST = np.sort(perm[:3000]), np.sort(perm[3000:])
    Y = T.toarray().astype(np.float64)
    has_topic = np.asarray(T.sum(axis=1)).ravel() > 0
    fold = np.random.default_rng(SEED).integers(0, K_FOLDS, N)
    fold_idx = [np.where(fold == f)[0] for f in range(K_FOLDS)]
    pos_in_fold = np.empty(N, dtype=int)
    for f in range(K_FOLDS):
        pos_in_fold[fold_idx[f]] = np.arange(len(fold_idx[f]))

    DISCOUNT = 1.0 / np.log2(np.arange(2, 12))
    IDEAL_DCG = np.array([DISCOUNT[:min(10, r)].sum() for r in N_REL])

    def evaluate(sim_fn, queries, chunk=500):
        """Per-query P@1, P@5, P@10, RR, nDCG@10, Hit@10 for a similarity function."""
        out = []
        for s in range(0, len(queries), chunk):
            q = queries[s:s + chunk]
            S = np.asarray(sim_fn(q), dtype=np.float32)
            S[np.arange(len(q)), q] = -np.inf
            top = np.argpartition(-S, 10, axis=1)[:, :10]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, axis=1), axis=1), axis=1)
            hit = REL[q[:, None], top]
            rr = np.where(hit.any(axis=1), 1.0 / (hit.argmax(axis=1) + 1), 0.0)
            out.append(np.c_[hit[:, 0], hit[:, :5].mean(axis=1), hit.mean(axis=1), rr,
                             (hit * DISCOUNT).sum(axis=1) / IDEAL_DCG[q], hit.any(axis=1)])
        return np.vstack(out)

    def summary(scores):
        return {m: float(v) for m, v in zip(METRICS, scores.mean(axis=0))}

    # ------------------------------------------------------------------ reference: TF-IDF baseline
    stemmer, token, stop = PorterStemmer(), re.compile(r"[a-z]+"), set(ENGLISH_STOP_WORDS)
    clean = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).map(
        lambda t: " ".join(stemmer.stem(w) for w in token.findall(str(t).lower()) if w not in stop and len(w) > 2))
    A = TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True).fit_transform(clean).tocsr()
    A_T = A.T.tocsr()
    test = {"TF-IDF baseline": evaluate(lambda q: (A[q] @ A_T).toarray(), TEST)}
    log("TF-IDF baseline   P@10 = %.4f" % test["TF-IDF baseline"][:, 2].mean())

    # ------------------------------------------------------------------ model E: pretrained embeddings, no labels
    for k in EMB_KEYS:
        test[ENCODER_INFO[k]["label"] + " alone"] = evaluate((lambda M: (lambda q: M[q] @ M.T))(EMB[k]), TEST)
    sim_E = lambda q: E_cat[q] @ E_cat.T
    test["Embeddings (3 encoders)"] = evaluate(sim_E, TEST)
    log("Embeddings        P@10 = %.4f" % test["Embeddings (3 encoders)"][:, 2].mean())

    # ------------------------------------------------------------------ model F-: least squares + SVD on the embeddings
    X = (S_F * E_raw).astype(np.float64)
    X32 = X.astype(np.float32)

    def fit_map(train_mask):
        """Ridge least squares  min ||X W - Y||^2 + lam ||W||^2  ->  W  (1920 x topics)."""
        Xtr = X[train_mask]
        G = Xtr.T @ Xtr
        G[np.diag_indices_from(G)] += LAMBDA_F
        return sla.solve(G, Xtr.T @ Y[train_mask], assume_a="pos")

    def projection(W):
        """SVD of the map, W = Q S P^T; the first R_F columns of Q S give the learned 300-d projection."""
        Q, S, _ = np.linalg.svd(W, full_matrices=False)
        return (Q[:, :R_F] * S[:R_F]).astype(np.float32)

    # pairwise exclusion: the model that scores (query in fold f, candidate in fold g) never saw folds f and g
    models = {(f, g): projection(fit_map(has_topic & (fold != f) & (fold != g)))
              for f in range(K_FOLDS) for g in range(f, K_FOLDS)}
    reps = {}

    def rep(key, f):
        if (key, f) not in reps:
            reps[(key, f)] = normalize(X32[fold_idx[f]] @ models[key])
        return reps[(key, f)]

    def sim_F(q):
        S = np.empty((len(q), N), dtype=np.float32)
        for f in np.unique(fold[q]):
            r_ = np.where(fold[q] == f)[0]
            for g in range(K_FOLDS):
                key = (min(f, g), max(f, g))
                S[np.ix_(r_, fold_idx[g])] = rep(key, f)[pos_in_fold[q[r_]]] @ rep(key, g).T
        return S

    test["Learned metric (300-d)"] = evaluate(sim_F, TEST)
    log("Learned metric    P@10 = %.4f" % test["Learned metric (300-d)"][:, 2].mean())

    # ------------------------------------------------------------------ fusion weight, chosen on validation queries
    grid = {}
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]:
        grid[w] = float(evaluate(lambda q: (1 - w) * sim_F(q) + w * sim_E(q), VAL)[:, 4].mean())   # nDCG@10
    W_FUSE = max(grid, key=grid.get)
    sim_fusion = lambda q: (1 - W_FUSE) * sim_F(q) + W_FUSE * sim_E(q)
    test["Deep fusion"] = evaluate(sim_fusion, TEST)
    log("Deep fusion       P@10 = %.4f   (w = %.1f)" % (test["Deep fusion"][:, 2].mean(), W_FUSE))

    # ------------------------------------------------------------------ final (deployment) model: all labelled papers
    W_all = fit_map(has_topic)
    L = projection(W_all)                                   # (1920, 300)
    R = normalize(X32 @ L)                                  # every paper as a unit 300-d vector

    # leakage audit for the learned metric
    s_in = evaluate(lambda q: R[q] @ R.T, TEST)
    P_oof = np.zeros((N, Y.shape[1]), dtype=np.float32)
    for f in range(K_FOLDS):
        P_oof[fold_idx[f]] = X[fold_idx[f]] @ fit_map(has_topic & (fold != f))
    P_oof = normalize(P_oof)
    s_oof = evaluate(lambda q: P_oof[q] @ P_oof.T, TEST)
    del P_oof

    # ------------------------------------------------------------------ breakdowns for the dashboard
    dom = df["domain"].to_numpy()
    by_domain = []
    for d in df["domain"].value_counts().index:
        m = dom[TEST] == d
        by_domain.append({"domain": d, "queries": int(m.sum()),
                          "TF-IDF baseline": float(test["TF-IDF baseline"][m, 2].mean()),
                          "Embeddings (3 encoders)": float(test["Embeddings (3 encoders)"][m, 2].mean()),
                          "Deep fusion": float(test["Deep fusion"][m, 2].mean())})

    kth = np.concatenate([np.sort(np.where(np.arange(N)[None, :] == q[:, None], -1.0, sim_E(q)), axis=1)[:, -10]
                          for q in np.array_split(TEST, 14)])          # cosine of each query's 10th nearest neighbour
    bands = [("below 0.6", -1, 0.6), ("0.6 - 0.7", 0.6, 0.7), ("0.7 - 0.8", 0.7, 0.8), ("0.8 or more", 0.8, 1.01)]
    density = [{"band": name, "share": float(((kth >= lo) & (kth < hi)).mean()),
                "TF-IDF baseline": float(test["TF-IDF baseline"][(kth >= lo) & (kth < hi), 2].mean()),
                "Deep fusion": float(test["Deep fusion"][(kth >= lo) & (kth < hi), 2].mean())}
               for name, lo, hi in bands]

    # 2-D map of the corpus: first two principal components of the embeddings
    coords = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(E_cat - E_cat.mean(axis=0)).astype(np.float32)

    # ------------------------------------------------------------------ save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    np.savez_compressed(MODEL_PATH, paper_id=paper_ids, L=L, R=R, W_topic=W_all.astype(np.float16),
                        topic_names=topic_names.astype(str), coords=coords,
                        config=np.array([S_F, LAMBDA_F, R_F, W_FUSE], dtype=np.float64))

    def row(name, family, labels, dims, scores):
        return dict(name=name, family=family, uses_labels=labels, dims=dims, **summary(scores))

    p_rand = N_REL[TEST] / (N - 1)
    metrics = {
        "n_papers": int(N), "n_val": int(len(VAL)), "n_test": int(len(TEST)), "n_topics": int(len(topic_names)),
        "config": {"S_F": S_F, "LAMBDA_F": LAMBDA_F, "R_F": R_F, "W_FUSE": W_FUSE, "fusion_grid_val_ndcg": grid},
        "encoders": {k: ENCODER_INFO[k] for k in EMB_KEYS},
        "random": {"P@10": float(p_rand.mean()), "Hit@10": float((1 - (1 - p_rand) ** 10).mean())},
        "models": [
            row("TF-IDF baseline", "classical", False, int(A.shape[1]), test["TF-IDF baseline"]),
            dict(name="Best classical ML (SVD + least squares + TF-IDF)", family="classical", uses_labels=True,
                 dims=int(A.shape[1]) + 200, **CLASSICAL_BEST),
            row("MiniLM-L6 alone", "deep learning", False, 384, test["MiniLM-L6 alone"]),
            row("SPECTER alone", "deep learning", False, 768, test["SPECTER alone"]),
            row("BGE-base alone", "deep learning", False, 768, test["BGE-base alone"]),
            row("Embeddings (3 encoders)", "deep learning", False, int(E_cat.shape[1]), test["Embeddings (3 encoders)"]),
            row("Learned metric (300-d)", "deep learning", True, R_F, test["Learned metric (300-d)"]),
            row("Deep fusion", "deep learning", True, R_F + int(E_cat.shape[1]), test["Deep fusion"]),
        ],
        "by_domain": by_domain,
        "density": density,
        "audit": [dict(protocol="In-sample (invalid: model saw both papers' labels)", **summary(s_in)),
                  dict(protocol="Ordinary out-of-fold (full rank)", **summary(s_oof)),
                  dict(protocol="Pairwise exclusion (reported)", **summary(test["Learned metric (300-d)"]))],
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "train_seconds": round(time.time() - t_start, 1),
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    log("saved %s (%.1f MB) and %s in %.0fs" % (MODEL_PATH, os.path.getsize(MODEL_PATH) / 1e6, METRICS_PATH,
                                               time.time() - t_start))
    return metrics


if __name__ == "__main__":
    m = build()
    print("\nHeld-out test accuracy (%d queries)" % m["n_test"])
    print(pd.DataFrame(m["models"]).set_index("name")[["family", "uses_labels", "dims"] + METRICS].round(4).to_string())
