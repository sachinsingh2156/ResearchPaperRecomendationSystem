"""
Research Paper Recommender — deep-learning dashboard.

    streamlit run dashboard.py

Uses the deep-learning model trained and saved by train_deep_model.py:

    text -> three pretrained transformer encoders -> 1,920-d embedding
         -> learned 300-d projection (ridge least squares + SVD)
         -> fusion of the two cosine scores

Tabs: Search · Similar papers · Model performance · Corpus map · How it works
"""
import json
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import normalize

import train_deep_model as tdm

st.set_page_config(page_title="Research Paper Recommender — Deep Learning", page_icon="📚", layout="wide")
alt.data_transformers.disable_max_rows()

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#7a7a75"
METRICS = tdm.METRICS
MODES = {
    "Deep fusion (best)": "fusion",
    "Learned metric (300-d)": "metric",
    "Raw embeddings (no labels)": "raw",
}
MODE_TO_MODEL = {"fusion": "Deep fusion", "metric": "Learned metric (300-d)", "raw": "Embeddings (3 encoders)"}
EXAMPLES = ["adversarial attacks on image classifiers",
            "transformer models for machine translation",
            "detecting software vulnerabilities with machine learning",
            "explainable AI in healthcare"]


# ============================================================================================ loading
@st.cache_resource(show_spinner="Loading the deep-learning model…")
def load_model():
    """Corpus, saved model artefacts and held-out metrics. Trains the model first if it has not been saved yet."""
    if not (os.path.exists(tdm.MODEL_PATH) and os.path.exists(tdm.METRICS_PATH)):
        tdm.build(verbose=False)
    df = pd.read_csv(tdm.DATA_PATH).dropna(subset=["abstract"]).reset_index(drop=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    art = np.load(tdm.MODEL_PATH, allow_pickle=False)
    ids = df["paper_id"].to_numpy(dtype=str)
    if not (art["paper_id"] == ids).all():                       # corpus changed since training -> retrain once
        tdm.build(verbose=False)
        art = np.load(tdm.MODEL_PATH, allow_pickle=False)
    cache = np.load(tdm.EMB_PATH, allow_pickle=False)
    E_raw = np.hstack([normalize(cache[k].astype(np.float32)) for k in tdm.EMB_KEYS])
    s_f, _, _, w = art["config"]
    with open(tdm.METRICS_PATH) as fh:
        metrics = json.load(fh)
    topics = df["topics"].fillna("").map(lambda s: [t.strip() for t in s.split(";") if t.strip()]).tolist()
    return dict(df=df, E_cat=E_raw / np.sqrt(len(tdm.EMB_KEYS)), L=art["L"], R=art["R"],
                W_topic=art["W_topic"].astype(np.float32), topic_names=art["topic_names"], coords=art["coords"],
                S_F=float(s_f), W=float(w), metrics=metrics, topics=topics,
                topic_sets=[{t.lower() for t in ts} for ts in topics],
                titles=df["title"].fillna("").tolist(), domain=df["domain"].to_numpy(),
                year=df["year"].to_numpy(dtype=float))


@st.cache_resource(show_spinner="Loading the transformer encoders (first time only)…")
def load_encoders():
    """The three pretrained encoders, or (None, reason) when torch / sentence-transformers are not installed."""
    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        encoders = {}
        for key in tdm.EMB_KEYS:
            info = tdm.ENCODER_INFO[key]
            try:
                enc = SentenceTransformer(info["hf_name"], device=device, local_files_only=True)
            except Exception:
                enc = SentenceTransformer(info["hf_name"], device=device)
            enc.max_seq_length = info["max_len"]
            encoders[key] = enc
        return encoders, device
    except Exception as e:                                        # noqa: BLE001 - shown to the user
        return None, "%s: %s" % (type(e).__name__, e)


@st.cache_data(show_spinner=False, max_entries=200)
def encode_query(text):
    """Free text -> (1,920-d vector with three unit-length blocks, encoding time in ms)."""
    encoders, _ = load_encoders()
    t0 = time.perf_counter()
    parts = [encoders[k].encode([text], convert_to_numpy=True, normalize_embeddings=True,
                                show_progress_bar=False).astype(np.float32) for k in tdm.EMB_KEYS]
    return np.hstack(parts).ravel(), (time.perf_counter() - t0) * 1000


# ============================================================================================ scoring
def combine(s_metric, s_raw, mode):
    if mode == "metric":
        return s_metric
    if mode == "raw":
        return s_raw
    return (1 - M["W"]) * s_metric + M["W"] * s_raw


def score_text(raw_vec, mode):
    """Similarity of a free-text query to every paper, and the topic scores predicted for the query."""
    feats = M["S_F"] * raw_vec
    x_q = feats @ M["L"]
    x_q = x_q / (np.linalg.norm(x_q) + 1e-12)
    sims = combine(M["R"] @ x_q, M["E_cat"] @ (raw_vec / np.sqrt(len(tdm.EMB_KEYS))), mode)
    return sims, feats @ M["W_topic"]


def score_paper(idx, mode):
    sims = combine(M["R"] @ M["R"][idx], M["E_cat"] @ M["E_cat"][idx], mode).astype(np.float32)
    sims[idx] = -np.inf
    return sims


def top_k(sims, k, mask):
    s = np.where(mask, sims, -np.inf)
    k = int(min(k, mask.sum()))
    if k == 0:
        return np.array([], dtype=int)
    top = np.argpartition(-s, k - 1)[:k]
    return top[np.argsort(-s[top])]


def paper_card(i, sim=None, shared=None):
    """One result card. `shared` = set of topics shared with the query paper (None when there is no query paper)."""
    row = M["df"].iloc[i]
    with st.container(border=True):
        left = st.container()
        left.markdown("**%s**" % row["title"])
        meta = []
        if pd.notna(row["authors"]):
            meta.append(str(row["authors"])[:110] + ("…" if len(str(row["authors"])) > 110 else ""))
        if pd.notna(row["year"]):
            meta.append(str(int(row["year"])))
        if pd.notna(row["venue"]):
            meta.append(str(row["venue"]))
        left.caption(" · ".join(meta + [row["domain"]]))
        badges = [":blue-badge[similarity %.3f]" % sim] if sim is not None else []
        if shared is not None:
            badges.append(":green-badge[✓ shares a topic]  " + " · ".join(sorted(shared)) if shared
                          else ":orange-badge[✗ no shared topic]")
        if badges:
            left.markdown("  ".join(badges))
        with st.expander("Abstract and topics"):
            st.write(row["abstract"])
            if M["topics"][i]:
                st.caption("OpenAlex topics: " + " · ".join(M["topics"][i]))


def value_labels(chart_data, x, y, fmt=".3f", dx=9):
    return alt.Chart(chart_data).mark_text(align="left", dx=dx, color=INK, fontSize=12).encode(
        x=x, y=y, text=alt.Text(x.split(":")[0] + ":Q", format=fmt))


# ============================================================================================ app
M = load_model()
MET = M["metrics"]
models_df = pd.DataFrame(MET["models"])
by_name = models_df.set_index("name")
encoders, enc_info = load_encoders()
domains_all = list(M["df"]["domain"].value_counts().index)
yr_min, yr_max = int(np.nanmin(M["year"])), int(np.nanmax(M["year"]))

with st.sidebar:
    st.header("Recommender settings")
    mode_label = st.radio("Model", list(MODES), index=0,
                          help="Deep fusion combines the learned 300-d metric with the raw embedding similarity. "
                               "Raw embeddings use no topic labels at all.")
    mode = MODES[mode_label]
    top_n = st.slider("Number of recommendations", 3, 20, 8)
    st.divider()
    st.subheader("Filter the results")
    sel_domains = st.multiselect("Domains", domains_all, default=[], placeholder="All domains")
    yr_from, yr_to = st.slider("Publication year", yr_min, yr_max, (yr_min, yr_max))
    keep_undated = st.checkbox("Include papers without a year", value=True)
    st.divider()
    sel = by_name.loc[MODE_TO_MODEL[mode]]
    st.caption("Held-out accuracy of the selected model")
    c1, c2 = st.columns(2)
    c1.metric("Precision@10", "%.1f%%" % (100 * sel["P@10"]))
    c2.metric("MRR", "%.3f" % sel["MRR"])
    st.caption("%s papers · %s test queries" % (f"{MET['n_papers']:,}", f"{MET['n_test']:,}"))

mask = np.ones(len(M["df"]), dtype=bool)
if sel_domains:
    mask &= np.isin(M["domain"], sel_domains)
in_years = (M["year"] >= yr_from) & (M["year"] <= yr_to)
mask &= (in_years | np.isnan(M["year"])) if keep_undated else in_years

st.title("📚 Research Paper Recommender")
st.caption("Deep-learning model: three pretrained transformer encoders → learned 300-dimensional metric → score fusion. "
           "Mathematical Foundations for Computer Science · Group 05")

tab_search, tab_similar, tab_perf, tab_map, tab_how = st.tabs(
    ["🔎 Search", "📄 Similar papers", "📊 Model performance", "🗺️ Corpus map", "ℹ️ How it works"])

# -------------------------------------------------------------------------------------------- Search
with tab_search:
    if encoders is None:
        st.warning("Free-text search needs the optional packages **torch** and **sentence-transformers** to encode "
                   "the query (`pip install torch sentence-transformers`). The *Similar papers* tab works without "
                   "them.\n\nDetails: `%s`" % enc_info)
    else:
        if "query" not in st.session_state:
            st.session_state["query"] = EXAMPLES[0]
        picked = st.pills("Try an example", EXAMPLES, selection_mode="single", key="example_pill")
        if picked and picked != st.session_state.get("last_pill"):
            st.session_state["query"] = picked
        st.session_state["last_pill"] = picked
        query = st.text_input("Describe the research you are looking for", key="query",
                              placeholder="e.g. graph neural networks for fraud detection")
        if query.strip():
            raw_vec, enc_ms = encode_query(query.strip())
            t0 = time.perf_counter()
            sims, topic_scores = score_text(raw_vec, mode)
            top = top_k(sims, top_n, mask)
            score_ms = (time.perf_counter() - t0) * 1000
            st.session_state["last_results"] = top.tolist()
            st.session_state["last_query"] = query.strip()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Papers searched", f"{int(mask.sum()):,}")
            k2.metric("Best similarity", "%.3f" % sims[top[0]] if len(top) else "–")
            k3.metric("Encode query", "%.0f ms" % enc_ms, help="Three transformer forward passes (cached per query).")
            k4.metric("Score all papers", "%.1f ms" % score_ms)

            res_col, side_col = st.columns([1.7, 1])
            with res_col:
                st.subheader("Top %d recommendations" % len(top))
                if len(top) == 0:
                    st.info("No paper matches the current filters.")
                for i in top:
                    paper_card(int(i), float(sims[i]))
            with side_col:
                st.subheader("What the model thinks this is about")
                best = np.argsort(-topic_scores)[:6]
                tp = pd.DataFrame({"topic": [str(M["topic_names"][j]).capitalize() for j in best],
                                   "score": np.clip(topic_scores[best], 0, None)})
                st.altair_chart(
                    alt.Chart(tp).mark_bar(color=BLUE, cornerRadiusEnd=4, height=14).encode(
                        x=alt.X("score:Q", title="score"),
                        y=alt.Y("topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
                        tooltip=["topic", alt.Tooltip("score:Q", format=".2f")]).properties(height=210),
                    width="stretch")
                st.caption("Topic scores come from the least-squares map inside the model; they are a by-product, "
                           "not a classifier trained for this purpose.")
                if len(top):
                    st.subheader("Where the results come from")
                    mix = pd.Series(M["domain"][top]).value_counts().rename_axis("domain").reset_index(name="papers")
                    st.altair_chart(
                        alt.Chart(mix).mark_bar(color=BLUE, cornerRadiusEnd=4, height=14).encode(
                            x=alt.X("papers:Q", title="recommended papers", axis=alt.Axis(tickMinStep=1)),
                            y=alt.Y("domain:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
                            tooltip=["domain", "papers"]).properties(height=40 + 26 * len(mix)),
                        width="stretch")
                    st.caption("Open the **Corpus map** tab to see these papers on the map.")

# -------------------------------------------------------------------------------------------- Similar papers
with tab_similar:
    st.markdown("Pick a paper from the corpus and get the papers most similar to it. No encoder is needed here: "
                "every paper's vectors are stored in the model.")
    a, b = st.columns([5, 1])
    if b.button("🎲 Random paper", width="stretch"):
        st.session_state["paper_idx"] = int(np.random.default_rng().integers(len(M["df"])))
    idx = a.selectbox("Paper", range(len(M["df"])), key="paper_idx", label_visibility="collapsed",
                      format_func=lambda i: M["titles"][i], placeholder="Type to search 9,956 titles")
    if idx is not None:
        st.subheader("Selected paper")
        paper_card(int(idx))
        sims = score_paper(int(idx), mode)
        top = top_k(sims, top_n, mask)
        q_topics = M["topic_sets"][int(idx)]
        shared = [{t for t in M["topics"][int(i)] if t.lower() in q_topics} for i in top]
        n_hit = sum(bool(s) for s in shared)
        m1, m2, m3 = st.columns(3)
        m1.metric("Recommendations", len(top))
        if q_topics:
            m2.metric("Share a topic with the selected paper", "%d of %d" % (n_hit, len(top)))
        m3.metric("Best similarity", "%.3f" % sims[top[0]] if len(top) else "–")
        st.caption("The ✓ / ✗ marks compare OpenAlex topics, the ground truth used for evaluation. The dashboard model "
                   "was trained on all labelled papers, so these live marks are illustrative; the held-out accuracy "
                   "is on the **Model performance** tab.")
        st.subheader("Most similar papers")
        for i, s_ in zip(top, shared):
            paper_card(int(i), float(sims[i]), s_ if q_topics else None)

# -------------------------------------------------------------------------------------------- Model performance
with tab_perf:
    fusion, base_row, classic = by_name.loc["Deep fusion"], by_name.loc["TF-IDF baseline"], by_name.iloc[1]
    st.subheader("Held-out accuracy of the deep-learning model")
    st.caption("Deep fusion on %s test queries that were never used to choose any setting. "
               "Change shown against the TF-IDF baseline." % f"{MET['n_test']:,}")
    k = st.columns(5)
    for col, (m_, lab, pct) in zip(k, [("P@10", "Precision@10", True), ("P@5", "Precision@5", True),
                                       ("P@1", "Precision@1", True), ("MRR", "MRR", False), ("Hit@10", "Hit@10", True)]):
        val, d = fusion[m_], fusion[m_] - base_row[m_]
        col.metric(lab, ("%.1f%%" % (100 * val)) if pct else "%.3f" % val,
                   ("%+.1f pts" % (100 * d)) if pct else "%+.3f" % d)
    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Precision@K** — share of the top K recommendations that are relevant (share an OpenAlex topic with the query paper).\n"
            "- **MRR** — mean of 1 / rank of the first relevant recommendation; 1.0 means the first result is always relevant.\n"
            "- **nDCG@10** — precision that rewards putting relevant papers higher in the list.\n"
            "- **Hit@10** — share of queries with at least one relevant paper in the top 10.\n"
            "- A random ranking scores Precision@10 = %.3f and Hit@10 = %.3f."
            % (MET["random"]["P@10"], MET["random"]["Hit@10"]))

    st.divider()
    st.subheader("Model comparison")
    metric = st.segmented_control("Metric", METRICS, default="P@10", key="cmp_metric") or "P@10"
    cmp_df = models_df.assign(value=models_df[metric],
                              labels=np.where(models_df["uses_labels"], "trained on topic labels", "no labels"))
    order = cmp_df["name"].tolist()
    x_enc = alt.X("value:Q", title=metric, scale=alt.Scale(zero=False, padding=28))
    y_enc = alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    pts = alt.Chart(cmp_df).mark_circle(size=170, opacity=1).encode(
        x=x_enc, y=y_enc,
        color=alt.Color("family:N", title=None, scale=alt.Scale(domain=["classical", "deep learning"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("name:N", title="model"), alt.Tooltip("family:N"), alt.Tooltip("labels:N", title="training"),
                 alt.Tooltip("dims:Q", title="dimensions", format=","), alt.Tooltip("value:Q", title=metric, format=".4f")])
    st.altair_chart((pts + value_labels(cmp_df, "value:Q", y_enc, dx=11)).properties(height=320), width="stretch")
    st.caption("Blue = classical machine learning (bag of words), orange = deep learning (transformer embeddings). "
               "Classical figures: TF-IDF is re-evaluated by the training script; the best classical model is the "
               "figure reported in the analysis notebook on the same test queries.")
    show_tab = models_df.rename(columns={"name": "model", "uses_labels": "trained on labels", "dims": "dimensions"})
    st.dataframe(show_tab[["model", "family", "trained on labels", "dimensions"] + METRICS], hide_index=True,
                 width="stretch",
                 column_config={m_: st.column_config.NumberColumn(format="%.3f") for m_ in METRICS} |
                               {"dimensions": st.column_config.NumberColumn(format="localized")})

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Precision@10 by research domain")
        dom_df = pd.DataFrame(MET["by_domain"])
        long = dom_df.melt(id_vars=["domain", "queries"], value_vars=["TF-IDF baseline", "Deep fusion"],
                           var_name="model", value_name="P@10").rename(columns={"P@10": "p10"})
        d_order = dom_df.sort_values("Deep fusion", ascending=False)["domain"].tolist()
        y_d = alt.Y("domain:N", sort=d_order, title=None, axis=alt.Axis(labelLimit=220))
        rule = alt.Chart(dom_df).mark_rule(color="#c9c8c2", strokeWidth=2).encode(
            y=y_d, x=alt.X("TF-IDF baseline:Q", scale=alt.Scale(zero=False, padding=26), title="Precision@10",
                           axis=alt.Axis(values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
            x2="Deep fusion:Q")
        dots = alt.Chart(long).mark_circle(size=150, opacity=1).encode(
            y=y_d, x="p10:Q",
            color=alt.Color("model:N", title=None, scale=alt.Scale(domain=["TF-IDF baseline", "Deep fusion"], range=[GREY, ORANGE]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["domain", "model", alt.Tooltip("p10:Q", title="Precision@10", format=".3f"),
                     alt.Tooltip("queries:Q", title="test queries", format=",")])
        lab = alt.Chart(dom_df).mark_text(align="left", dx=11, color=INK, fontSize=12).encode(
            y=y_d, x="Deep fusion:Q", text=alt.Text("Deep fusion:Q", format=".3f"))
        st.altair_chart((rule + dots + lab).properties(height=280), width="stretch")
        best_d, worst_d = dom_df.loc[dom_df["Deep fusion"].idxmax()], dom_df.loc[dom_df["Deep fusion"].idxmin()]
        st.caption("The deep model is ahead in every domain. Easiest: %s (%.3f); hardest: %s (%.3f)."
                   % (best_d["domain"], best_d["Deep fusion"], worst_d["domain"], worst_d["Deep fusion"]))
    with right:
        st.subheader("What limits accuracy: corpus coverage")
        den = pd.DataFrame(MET["density"])
        den["label"] = den.apply(lambda r: "%.3f  (%s of queries)" % (
            r["Deep fusion"], "under 1%" if r["share"] < 0.01 else "%.0f%%" % (100 * r["share"])), axis=1)
        den["band"] = "cosine " + den["band"]
        band_order = den["band"].tolist()
        y_b = alt.Y("band:N", sort=band_order, title=None, axis=alt.Axis(labelLimit=160))
        bars = alt.Chart(den).mark_bar(color=ORANGE, cornerRadiusEnd=4, height=22).encode(
            y=y_b, x=alt.X("Deep fusion:Q", title="Precision@10 of the deep fusion model", scale=alt.Scale(domain=[0, 1.5]),
                           axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
            tooltip=["band", alt.Tooltip("share:Q", title="share of test queries", format=".1%"),
                     alt.Tooltip("Deep fusion:Q", title="deep fusion P@10", format=".3f"),
                     alt.Tooltip("TF-IDF baseline:Q", title="TF-IDF P@10", format=".3f")])
        blab = alt.Chart(den).mark_text(align="left", dx=6, color=INK, fontSize=12).encode(
            y=y_b, x="Deep fusion:Q", text="label:N")
        st.altair_chart((bars + blab).properties(height=280), width="stretch")
        dense = den.iloc[-1]
        st.caption("Test queries grouped by the cosine similarity of their 10th nearest paper in the corpus. "
                   "When a query has ten close neighbours (%.0f%% of queries) precision is %.1f%%. "
                   "It falls where the corpus covers the subject thinly — a limit of a 10,000-paper corpus, not of the model."
                   % (100 * dense["share"], 100 * dense["Deep fusion"]))

    st.divider()
    st.subheader("Leakage audit")
    st.markdown("The learned metric is trained on the same kind of topic labels that define relevance, so it is "
                "evaluated with **pairwise exclusion**: the model that scores a (query, candidate) pair never saw the "
                "labels of either paper. The table shows the same model under three protocols.")
    aud = pd.DataFrame(MET["audit"])
    st.dataframe(aud[["protocol"] + METRICS], hide_index=True, width="stretch",
                 column_config={m_: st.column_config.NumberColumn(format="%.3f") for m_ in METRICS})
    st.caption("The in-sample figure (%.3f) is what over-fitting looks like; the reported figure (%.3f) matches ordinary "
               "out-of-fold prediction (%.3f), so it reflects genuine generalisation."
               % (aud.loc[0, "P@10"], aud.loc[2, "P@10"], aud.loc[1, "P@10"]))

# -------------------------------------------------------------------------------------------- Corpus map
with tab_map:
    st.subheader("Map of the corpus in the embedding space")
    st.caption("Each dot is a paper, placed by the first two principal components of its transformer embedding. "
               "Papers on similar subjects sit close together. The model was never given the domain labels.")
    c1, c2 = st.columns([2, 3])
    highlight = c1.selectbox("Highlight a domain", domains_all, index=0)
    last = st.session_state.get("last_results", [])
    show_last = c2.checkbox("Show my last search results (%d)" % len(last), value=bool(last), disabled=not last)
    rng = np.random.default_rng(tdm.SEED)
    samp = np.sort(rng.choice(len(M["df"]), 5000, replace=False))
    mp = pd.DataFrame({"x": M["coords"][samp, 0], "y": M["coords"][samp, 1], "title": np.array(M["titles"])[samp],
                       "domain": M["domain"][samp], "year": M["year"][samp]})
    mp["group"] = np.where(mp["domain"] == highlight, highlight, "other domains")
    ax_x = alt.X("x:Q", axis=None)
    ax_y = alt.Y("y:Q", axis=None)
    tips = [alt.Tooltip("title:N"), alt.Tooltip("domain:N"), alt.Tooltip("year:Q", format=".0f")]
    back = alt.Chart(mp[mp["group"] == "other domains"]).mark_circle(size=14, color="#b9b8b2", opacity=0.45).encode(
        x=ax_x, y=ax_y, tooltip=tips)
    front = alt.Chart(mp[mp["group"] == highlight]).mark_circle(size=22, opacity=0.8).encode(
        x=ax_x, y=ax_y, tooltip=tips,
        color=alt.Color("group:N", title=None, scale=alt.Scale(domain=[highlight], range=[BLUE]),
                        legend=alt.Legend(orient="bottom")))
    layers = back + front
    if show_last and last:
        lr = pd.DataFrame({"x": M["coords"][last, 0], "y": M["coords"][last, 1], "title": np.array(M["titles"])[last],
                           "domain": M["domain"][last], "year": M["year"][last],
                           "group": "results for: " + st.session_state.get("last_query", "")[:50]})
        layers = layers + alt.Chart(lr).mark_point(size=190, filled=True, opacity=1, stroke="white", strokeWidth=1.5).encode(
            x=ax_x, y=ax_y, tooltip=tips,
            color=alt.Color("group:N", title=None, scale=alt.Scale(range=[ORANGE]),
                            legend=alt.Legend(orient="bottom", labelLimit=420)))
    st.altair_chart(layers.resolve_scale(color="independent").properties(height=520), width="stretch")
    st.caption("A random sample of 5,000 papers is drawn. Hover over a dot for its title, domain and year.")

    st.divider()
    l2, r2 = st.columns(2)
    with l2:
        st.subheader("Papers per domain")
        dc = M["df"]["domain"].value_counts().rename_axis("domain").reset_index(name="papers")
        y_dc = alt.Y("domain:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220))
        st.altair_chart(
            (alt.Chart(dc).mark_bar(color=BLUE, cornerRadiusEnd=4, height=18).encode(
                x=alt.X("papers:Q", title="papers", scale=alt.Scale(domain=[0, dc["papers"].max() * 1.15])), y=y_dc,
                tooltip=["domain", alt.Tooltip("papers:Q", format=",")])
             + alt.Chart(dc).mark_text(align="left", dx=6, color=INK, fontSize=12).encode(
                x="papers:Q", y=y_dc, text=alt.Text("papers:Q", format=","))).properties(height=250),
            width="stretch")
    with r2:
        st.subheader("Papers per publication year")
        yc = pd.Series(M["year"]).dropna().astype(int)
        yc = yc[yc >= 1990].value_counts().sort_index().rename_axis("year").reset_index(name="papers")
        st.altair_chart(
            alt.Chart(yc).mark_bar(color=BLUE, cornerRadiusEnd=2).encode(
                x=alt.X("year:O", title="year (1990 onwards)", axis=alt.Axis(values=list(range(1990, 2027, 5)), labelAngle=0)),
                y=alt.Y("papers:Q", title="papers"),
                tooltip=["year", alt.Tooltip("papers:Q", format=",")]).properties(height=250),
            width="stretch")

# -------------------------------------------------------------------------------------------- How it works
with tab_how:
    cfg = MET["config"]
    st.subheader("The pipeline")
    st.markdown(
        """
1. **Read the text.** The title and abstract are given, unprocessed, to three pretrained **transformer encoders**.
   Each one tokenises the text into word pieces, passes them through 6–12 self-attention layers so that every token
   is interpreted in its context, and pools the result into one vector of unit length.
2. **Embedding (1,920 dimensions).** The three vectors are joined. The cosine similarity of two papers in this space
   is the average of the three encoders' similarities. This step uses no labels — it is the *Raw embeddings* model.
3. **Learned metric (300 dimensions).** A ridge-regularised **least-squares** map from the embedding to the 1,846
   OpenAlex topics is fitted, $W = (X^\\top X + \\lambda I)^{-1} X^\\top Y$, and factorised with an **SVD**,
   $W = Q S P^\\top$. Keeping the 300 strongest directions gives a compact vector $x = e\\,Q_{300}S_{300}$ in which
   distance reflects topical relatedness. This is the *Learned metric* model.
4. **Fusion.** The final score is $(1-w)\\cdot$ learned-metric cosine $+\\; w\\cdot$ embedding cosine. This is the
   *Deep fusion* model.
5. **Recommend.** The papers with the highest score are returned. A free-text query goes through exactly the same
   steps as a paper.
""")
    st.subheader("The pretrained encoders")
    enc_tab = pd.DataFrame([{"encoder": v["label"], "Hugging Face model": v["hf_name"], "layers": v["layers"],
                             "parameters": v["params"], "output dims": v["dims"], "pretrained on": v["trained_on"]}
                            for v in MET["encoders"].values()])
    st.dataframe(enc_tab, hide_index=True, width="stretch")
    st.caption("The encoders are used as published; no transformer is trained or fine-tuned in this project, and none "
               "of them was trained on this corpus's topic labels.")

    a, b = st.columns(2)
    with a:
        st.subheader("Training configuration")
        st.markdown(
            "- Embedding scale in the least-squares features: **%g**\n"
            "- Ridge strength λ: **%g**\n"
            "- Rank of the learned metric: **%d**\n"
            "- Fusion weight *w* on the raw embeddings: **%g** (chosen on %s validation queries)\n"
            "- Evaluation: %s held-out test queries, pairwise-exclusion cross-fitting with 5 folds\n"
            "- Last trained: %s (%.0f s)"
            % (cfg["S_F"], cfg["LAMBDA_F"], cfg["R_F"], cfg["W_FUSE"], f"{MET['n_val']:,}", f"{MET['n_test']:,}",
               MET["trained_at"], MET["train_seconds"]))
    with b:
        st.subheader("Files")
        st.markdown(
            "- `data/research_papers.csv` — the corpus (%s papers)\n"
            "- `data/paper_embeddings.npz` — cached transformer embeddings of every paper\n"
            "- `train_deep_model.py` — trains, evaluates and saves the model\n"
            "- `models/deep_recommender.npz` — the saved model (projection, paper vectors, topic map, map coordinates)\n"
            "- `models/deep_metrics.json` — the held-out results shown on the *Model performance* tab\n"
            "- `dashboard.py` — this dashboard" % f"{MET['n_papers']:,}")
        if encoders is not None:
            st.caption("Encoders are running on: **%s**" % enc_info)
    st.subheader("Limits to keep in mind")
    st.markdown(
        "- **Relevance is a proxy.** Two papers count as related if they share an automatically assigned OpenAlex topic; "
        "this is not a human judgement.\n"
        "- **The dashboard model has seen every label.** It is trained on all labelled papers for the best live results; "
        "the accuracy figures come from separate models that never saw the papers they were scored on.\n"
        "- **Pretrained encoders bring outside knowledge.** They learned from very large public text collections and may "
        "have seen some of these abstracts as plain text (never their topic labels).\n"
        "- **Coverage.** A 10,000-paper corpus cannot contain ten close matches for every subject; see the *Model "
        "performance* tab.")
