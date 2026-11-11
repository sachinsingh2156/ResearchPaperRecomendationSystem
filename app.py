"""
Research Paper Recommendation System — interactive web interface (Objective 7).

Same pipeline as notebooks/Research_Paper_Recommendation_System_Final.ipynb:

    text -> TF-IDF (A)  -> SVD / LSA (B)  -> least-squares learned metric (C)  -> fusion with TF-IDF (D)
    text -> pretrained transformer embeddings (E) -> least squares on [embeddings, SVD] (F) -> fusion (G)

Approach G needs the optional packages torch + sentence-transformers and the cache data/paper_embeddings.npz
(created by the notebook). Without them the app offers approaches A-D only.

Run with:
    streamlit run app.py
"""
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp
import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Research Paper Recommender", page_icon="📚", layout="wide")

# Hyper-parameters selected on the validation queries in the final notebook (sections 9-11).
K_MAX = 1000     # rank of the SVD
K_SVD = 300      # rank used for the plain SVD mode (Approach B)
LAMBDA = 0.3     # ridge strength of the least-squares map
R_SUP = 200      # rank of the learned metric (Approach C)
BETA = 0.4       # weight of TF-IDF in the fusion (Approach D)
S_F = 2.0        # scale of the embedding block in the feature vector of Approach F
LAMBDA_F = 3.0   # ridge strength for Approach F
R_F = 300        # rank of the learned metric of Approach F
W_G = 0.4        # weight of the raw embedding score in the fusion (Approach G)

EMB_PATH = "data/paper_embeddings.npz"
EMB_MODELS = {   # key: (Hugging Face model name, maximum number of tokens read)
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 256),
    "specter": ("sentence-transformers/allenai-specter", 384),
    "bge": ("BAAI/bge-base-en-v1.5", 384),
}

DEEP_LABEL = "G: Deep fusion — transformer embeddings + least squares (best)"
METHODS = {
    "D: Fusion — learned metric + TF-IDF": "fusion",
    "C: SVD + least-squares learned metric (200 dims)": "supervised",
    "B: SVD / LSA (k = 300)": "svd",
    "A: TF-IDF baseline (31,291 dims)": "tfidf",
}

# ----------------------------------------------------------------------------- preprocessing
_stemmer = PorterStemmer()
_token = re.compile(r"[a-z]+")
_stop = set(ENGLISH_STOP_WORDS)


@st.cache_resource(show_spinner="Loading the transformer encoders…")
def load_encoders():
    """Returns {key: encoder}, or None when the optional deep-learning packages / cache are unavailable."""
    if not os.path.exists(EMB_PATH):
        return None
    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        encoders = {}
        for key, (name, max_len) in EMB_MODELS.items():
            try:
                enc = SentenceTransformer(name, device=device, local_files_only=True)
            except Exception:
                enc = SentenceTransformer(name, device=device)
            enc.max_seq_length = max_len
            encoders[key] = enc
        return encoders
    except Exception:
        return None


def preprocess(text: str) -> str:
    """lowercase -> tokenise -> remove stop-words / short tokens -> Porter stemming."""
    return " ".join(
        _stemmer.stem(w) for w in _token.findall(str(text).lower())
        if w not in _stop and len(w) > 2
    )


# ----------------------------------------------------------------------------- model (cached)
@st.cache_resource(show_spinner="Building the vector-space model (about 30 seconds, once)…")
def build_model():
    df = pd.read_csv("data/research_papers.csv").dropna(subset=["abstract"]).reset_index(drop=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    clean = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).map(preprocess)

    # Approach A: TF-IDF term-document matrix (rows are unit vectors, so cosine = dot product)
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True)
    A = vectorizer.fit_transform(clean).tocsr()

    # Approach B: rank-1000 SVD, Z = U S = A V
    svd = TruncatedSVD(n_components=K_MAX, n_iter=7, random_state=42)
    Z = svd.fit_transform(A)
    Z_svd = normalize(Z[:, :K_SVD]).astype(np.float32)

    # Approach C: ridge least squares from the latent space to topic space, then an SVD of the map
    topic_lists = df["topics"].fillna("").map(lambda s: [t.strip().lower() for t in s.split(";") if t.strip()])
    topic_names = np.array(sorted({t for ts in topic_lists for t in ts}))
    topic_id = {t: i for i, t in enumerate(topic_names)}
    rows, cols = zip(*[(i, topic_id[t]) for i, ts in enumerate(topic_lists) for t in ts])
    Y = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(topic_names))).toarray()
    labelled = Y.sum(axis=1) > 0
    Xl = Z[labelled]
    G = Xl.T @ Xl
    G[np.diag_indices_from(G)] += LAMBDA
    W = sla.solve(G, Xl.T @ Y[labelled], assume_a="pos")          # normal equations (Z^T Z + lam I) W = Z^T Y
    Q, S, _ = np.linalg.svd(W, full_matrices=False)               # W = Q S P^T
    L = (Q[:, :R_SUP] * S[:R_SUP]).astype(np.float32)             # k x r learned projection
    R = normalize(Z.astype(np.float32) @ L)                       # every paper as a unit r-vector

    M = dict(df=df, vectorizer=vectorizer, A=A, svd=svd, Z_svd=Z_svd, W=W.astype(np.float32), L=L, R=R,
             topic_names=topic_names, deep=False)

    # Approaches E-G: cached transformer embeddings + the same least-squares pipeline on [embeddings, SVD]
    if os.path.exists(EMB_PATH):
        cache = np.load(EMB_PATH, allow_pickle=False)
        if (cache["paper_id"] == df["paper_id"].to_numpy(dtype=str)).all():
            emb = [normalize(cache[k].astype(np.float32)) for k in EMB_MODELS]
            E_cat = np.hstack(emb) / np.sqrt(len(emb))            # unit rows; dot = mean of the three cosines
            X = np.hstack([S_F * e for e in emb] + [Z])
            Xl = X[labelled]
            G = Xl.T @ Xl
            G[np.diag_indices_from(G)] += LAMBDA_F
            W_F = sla.solve(G, Xl.T @ Y[labelled], assume_a="pos")
            Qf, Sf, _ = np.linalg.svd(W_F, full_matrices=False)
            L_F = (Qf[:, :R_F] * Sf[:R_F]).astype(np.float32)
            M.update(deep=True, E_cat=E_cat, L_F=L_F, R_F=normalize(X.astype(np.float32) @ L_F))
    return M


def recommend(query, M, method, k, encoders=None):
    q = M["vectorizer"].transform([preprocess(query)])            # 1 x n, unit length
    t0 = time.perf_counter()
    zq = M["svd"].transform(q).astype(np.float32)                 # fold-in: z_q = q V_k
    s_tfidf = (M["A"] @ q.T).toarray().ravel()
    if method == "deep":
        parts = [encoders[key].encode([query], convert_to_numpy=True, normalize_embeddings=True,
                                      show_progress_bar=False).astype(np.float32) for key in EMB_MODELS]
        e_q = np.hstack(parts).ravel() / np.sqrt(len(parts))
        x_q = normalize(np.hstack([S_F * p for p in parts] + [zq]) @ M["L_F"]).ravel()
        sims = (1 - W_G) * (M["R_F"] @ x_q) + W_G * (M["E_cat"] @ e_q)
    elif method == "tfidf":
        sims = s_tfidf
    elif method == "svd":
        sims = M["Z_svd"] @ normalize(zq[:, :K_SVD]).ravel()
    else:
        s_sup = M["R"] @ normalize(zq @ M["L"]).ravel()
        sims = s_sup if method == "supervised" else (1 - BETA) * s_sup + BETA * s_tfidf
    dt = (time.perf_counter() - t0) * 1000
    top = np.argsort(-sims)[:k]
    res = M["df"].loc[top, ["title", "authors", "year", "venue", "domain", "abstract"]].copy()
    res.insert(0, "similarity", np.round(sims[top], 3))
    topic_scores = (zq @ M["W"]).ravel()
    best = np.argsort(-topic_scores)[:3]
    topics = [(M["topic_names"][i], float(topic_scores[i])) for i in best]
    return res.reset_index(drop=True), dt, topics


# ----------------------------------------------------------------------------- UI
st.title("📚 Research Paper Recommendation System")
st.caption("Vector-space models + linear-algebraic dimensionality reduction (SVD / LSA) and least squares — "
           "Mathematical Foundations for Computer Science")

M = build_model()
encoders = load_encoders() if M["deep"] else None
methods = ({DEEP_LABEL: "deep"} if encoders else {}) | METHODS

with st.sidebar:
    st.header("Settings")
    label = st.radio("Representation", list(methods), index=0,
                     help="A = full TF-IDF space. B = rank-300 SVD latent space. "
                          "C = 200-dimensional learned metric on the SVD space. D = C fused with A. "
                          "G = pretrained transformer embeddings + SVD features through the same least-squares map.")
    topk = st.slider("Number of recommendations", 3, 20, 5)
    st.markdown("---")
    st.metric("Papers in corpus", f"{len(M['df']):,}")
    st.metric("Domains", M["df"]["domain"].nunique())
    st.caption("Test-set Precision@10: A 0.699 · B 0.712 · C 0.799 · D 0.801 · G 0.881")
    if not encoders:
        st.caption("Approach G is hidden: it needs torch, sentence-transformers and data/paper_embeddings.npz.")

query = st.text_input("Enter a research query", value="adversarial attacks on image classifiers")

if query.strip():
    res, dt, topics = recommend(query, M, methods[label], topk, encoders)
    st.success(f"Top {topk} papers • {label} • {dt:.1f} ms")
    st.caption("Predicted topics for this query: " + " · ".join(f"{t} ({s:.2f})" for t, s in topics))
    for _, row in res.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{row['title']}**")
            c2.metric("similarity", f"{row['similarity']:.3f}")
            meta = []
            if pd.notna(row["authors"]):
                meta.append(str(row["authors"])[:120])
            if pd.notna(row["year"]):
                meta.append(str(int(row["year"])))
            if pd.notna(row["venue"]):
                meta.append(str(row["venue"]))
            st.caption(" · ".join(meta) + f"  ·  _{row['domain']}_")
            with st.expander("Abstract"):
                st.write(row["abstract"])
