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

st.set_page_config(
    page_title="Research Paper Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

DEEP_LABEL = "G — Deep fusion (embeddings + least squares)"
METHODS = {
    "D — Fusion (learned metric + TF-IDF)": "fusion",
    "C — Learned metric on SVD (200-d)": "supervised",
    "B — SVD / LSA (k = 300)": "svd",
    "A — TF-IDF baseline": "tfidf",
}

METHOD_BLURB = {
    "deep": "Pretrained transformer embeddings fused with a least-squares metric on SVD features.",
    "fusion": "Learned 200-d metric fused with full TF-IDF cosine similarity.",
    "supervised": "Ridge least-squares map on the SVD space, reduced to 200 dimensions.",
    "svd": "Rank-300 latent semantic space via truncated SVD.",
    "tfidf": "Full sparse TF-IDF term space (cosine = dot product on unit rows).",
}

PRECISION = {
    "deep": 0.881,
    "fusion": 0.801,
    "supervised": 0.799,
    "svd": 0.712,
    "tfidf": 0.699,
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
def inject_styles():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap');

:root {
  --ink: #15202b;
  --muted: #5c6b7a;
  --line: #d7dee6;
  --paper: #f4f7fa;
  --surface: #ffffff;
  --accent: #1f6f78;
  --accent-soft: #e6f2f3;
  --rank: #8a96a3;
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 500px at 8% -10%, #e8eef4 0%, transparent 55%),
    radial-gradient(900px 420px at 100% 0%, #e7f1f1 0%, transparent 50%),
    linear-gradient(180deg, #f7f9fb 0%, var(--paper) 40%, #eef2f6 100%);
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #fbfcfd 0%, #f1f5f8 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { font-family: "IBM Plex Sans", sans-serif !important; }

.block-container {
  padding-top: 2.25rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 1080px;
}

#MainMenu, footer { visibility: hidden; }

.brand {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
  margin-bottom: 1.75rem;
  padding-bottom: 1.35rem;
  border-bottom: 1px solid var(--line);
}
.brand-kicker {
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 600;
}
.brand h1 {
  font-family: "Source Serif 4", Georgia, serif !important;
  font-size: 2.15rem !important;
  font-weight: 600 !important;
  line-height: 1.15 !important;
  letter-spacing: -0.02em;
  color: var(--ink) !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
}
.brand p {
  margin: 0;
  max-width: 42rem;
  color: var(--muted);
  font-size: 0.98rem;
  line-height: 1.55;
}

.corpus-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.85rem;
  margin: 0 0 1.5rem 0;
}
.corpus-cell {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 2px;
  padding: 0.9rem 1rem;
}
.corpus-cell .label {
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
}
.corpus-cell .value {
  margin-top: 0.35rem;
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 1.45rem;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.02em;
}

.search-panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 2px;
  padding: 1.15rem 1.2rem 0.35rem 1.2rem;
  margin-bottom: 0.35rem;
}
.search-panel .panel-title {
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
  margin-bottom: 0.55rem;
}

.result-bar {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
  flex-wrap: wrap;
  margin: 1.4rem 0 0.85rem 0;
  padding-bottom: 0.7rem;
  border-bottom: 1px solid var(--line);
}
.result-bar .summary {
  font-size: 0.92rem;
  color: var(--ink);
  font-weight: 500;
}
.result-bar .meta {
  font-size: 0.82rem;
  color: var(--muted);
}
.topics {
  margin: 0 0 1.1rem 0;
  font-size: 0.86rem;
  color: var(--muted);
  line-height: 1.5;
}
.topics strong {
  color: var(--ink);
  font-weight: 600;
}

.paper {
  display: grid;
  grid-template-columns: 2.4rem 1fr auto;
  gap: 0.85rem 1rem;
  align-items: start;
  padding: 1.05rem 0;
  border-bottom: 1px solid var(--line);
}
.paper:last-child { border-bottom: none; }
.rank {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 1.05rem;
  color: var(--rank);
  font-weight: 600;
  padding-top: 0.12rem;
}
.paper-title {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 1.08rem;
  font-weight: 600;
  line-height: 1.35;
  color: var(--ink);
  margin: 0 0 0.35rem 0;
}
.paper-meta {
  font-size: 0.82rem;
  color: var(--muted);
  line-height: 1.45;
  margin: 0;
}
.domain {
  display: inline-block;
  margin-top: 0.45rem;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--accent);
  background: var(--accent-soft);
  border: 1px solid #c9e0e2;
  padding: 0.18rem 0.45rem;
  border-radius: 2px;
  font-weight: 600;
}
.score {
  text-align: right;
  min-width: 4.5rem;
}
.score .num {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.02em;
}
.score .lbl {
  font-size: 0.66rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 0.15rem;
}

div[data-testid="stTextInput"] label {
  font-size: 0.78rem !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted) !important;
  font-weight: 600 !important;
}
div[data-testid="stTextInput"] input {
  border-radius: 2px !important;
  border: 1px solid var(--line) !important;
  background: #fbfcfd !important;
  font-size: 0.98rem !important;
  padding: 0.7rem 0.85rem !important;
  color: var(--ink) !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent) !important;
}

[data-testid="stExpander"] {
  border: none !important;
  background: transparent !important;
  margin-top: 0.35rem;
}
[data-testid="stExpander"] details {
  border: 1px solid var(--line);
  border-radius: 2px;
  background: #fafbfc;
}
[data-testid="stExpander"] summary {
  font-size: 0.8rem !important;
  color: var(--muted) !important;
  font-weight: 500 !important;
}
[data-testid="stExpander"] p {
  font-size: 0.9rem;
  line-height: 1.55;
  color: #334155;
}

.sidebar-section {
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
  margin: 0.4rem 0 0.75rem 0;
}
.sidebar-note {
  font-size: 0.78rem;
  color: var(--muted);
  line-height: 1.45;
  margin-top: 0.85rem;
}
.method-note {
  font-size: 0.8rem;
  color: var(--muted);
  line-height: 1.45;
  margin: 0.35rem 0 1rem 0;
  padding: 0.65rem 0.75rem;
  background: var(--accent-soft);
  border-left: 2px solid var(--accent);
}

div[data-testid="stMetricValue"] {
  font-family: "Source Serif 4", Georgia, serif !important;
  font-size: 1.35rem !important;
}
div[data-testid="stMetricLabel"] {
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.68rem !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def esc(text) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


inject_styles()

st.markdown(
    """
<div class="brand">
  <div class="brand-kicker">Mathematical Foundations for Computer Science · Group 05</div>
  <h1>Research Paper Recommendation System</h1>
  <p>Retrieve related work from a curated OpenAlex corpus using vector-space models,
  SVD / LSA dimensionality reduction, and least-squares learned metrics.</p>
</div>
    """,
    unsafe_allow_html=True,
)

M = build_model()
encoders = load_encoders() if M["deep"] else None
methods = ({DEEP_LABEL: "deep"} if encoders else {}) | METHODS

n_papers = len(M["df"])
n_domains = int(M["df"]["domain"].nunique())
n_years = int(M["df"]["year"].dropna().nunique()) if "year" in M["df"].columns else 0

st.markdown(
    f"""
<div class="corpus-strip">
  <div class="corpus-cell"><div class="label">Papers indexed</div><div class="value">{n_papers:,}</div></div>
  <div class="corpus-cell"><div class="label">Research domains</div><div class="value">{n_domains}</div></div>
  <div class="corpus-cell"><div class="label">Publication years</div><div class="value">{n_years}</div></div>
</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-section">Retrieval settings</div>', unsafe_allow_html=True)
    label = st.radio(
        "Representation",
        list(methods),
        index=0,
        help=(
            "A = full TF-IDF space. B = rank-300 SVD latent space. "
            "C = 200-dimensional learned metric on the SVD space. D = C fused with A. "
            "G = pretrained transformer embeddings + SVD features through the same least-squares map."
        ),
    )
    method_key = methods[label]
    st.markdown(
        f'<div class="method-note">{METHOD_BLURB[method_key]}'
        f"<br><br>Held-out Precision@10: <strong>{PRECISION[method_key]:.3f}</strong></div>",
        unsafe_allow_html=True,
    )
    topk = st.slider("Recommendations to return", 3, 20, 5)
    st.markdown("---")
    st.markdown('<div class="sidebar-section">Corpus</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Papers", f"{n_papers:,}")
    c2.metric("Domains", n_domains)
    st.markdown(
        '<div class="sidebar-note">Test-set Precision@10 — '
        "A 0.699 · B 0.712 · C 0.799 · D 0.801 · G 0.881</div>",
        unsafe_allow_html=True,
    )
    if not encoders:
        st.markdown(
            '<div class="sidebar-note">Approach G is unavailable until torch, '
            "sentence-transformers, and data/paper_embeddings.npz are present.</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="search-panel"><div class="panel-title">Query</div></div>',
    unsafe_allow_html=True,
)
query = st.text_input(
    "Research query",
    value="adversarial attacks on image classifiers",
    label_visibility="collapsed",
    placeholder="Describe the topic, method, or problem you want papers about",
)

if query.strip():
    res, dt, topics = recommend(query, M, method_key, topk, encoders)
    topic_str = " · ".join(f"{esc(t)} ({s:.2f})" for t, s in topics)
    st.markdown(
        f"""
<div class="result-bar">
  <div class="summary">{topk} recommendations</div>
  <div class="meta">{esc(label)} · {dt:.1f} ms</div>
</div>
<div class="topics"><strong>Predicted topics</strong> — {topic_str}</div>
        """,
        unsafe_allow_html=True,
    )

    for i, row in res.iterrows():
        authors = esc(str(row["authors"])[:110]) if pd.notna(row["authors"]) else ""
        year = str(int(row["year"])) if pd.notna(row["year"]) else ""
        venue = esc(str(row["venue"])) if pd.notna(row["venue"]) else ""
        domain = esc(row["domain"]) if pd.notna(row["domain"]) else ""
        meta_bits = [b for b in (authors, year, venue) if b]
        meta = " · ".join(meta_bits)
        st.markdown(
            f"""
<div class="paper">
  <div class="rank">{i + 1:02d}</div>
  <div>
    <div class="paper-title">{esc(row["title"])}</div>
    <p class="paper-meta">{meta}</p>
    <span class="domain">{domain}</span>
  </div>
  <div class="score">
    <div class="num">{row["similarity"]:.3f}</div>
    <div class="lbl">Similarity</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Abstract"):
            st.write(row["abstract"] if pd.notna(row["abstract"]) else "No abstract available.")
elif not query.strip():
    st.markdown(
        '<div class="topics">Enter a research query above to retrieve ranked papers from the corpus.</div>',
        unsafe_allow_html=True,
    )
