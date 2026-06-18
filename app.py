

# STREAMLIT APP: app.py
# BookMatch AI — Goodreads Recommendation System


import os
import json
import pickle
import html

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# APP CONFIGURATION


ARTIFACT_DIR = "artifacts"

st.set_page_config(
    page_title="BookMatch AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)



# CUSTOM CSS


st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(79, 70, 229, 0.18), transparent 35%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.16), transparent 30%),
            linear-gradient(135deg, #020617 0%, #0F172A 45%, #111827 100%);
        color: #F9FAFB;
    }

    /* Hero card */
    .hero {
        padding: 34px 38px;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.92)),
            radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.35), transparent 35%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.38);
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #F8FAFC;
        font-size: 3.2rem;
        margin-bottom: 0.35rem;
        letter-spacing: -1px;
    }

    .hero p {
        color: #CBD5E1;
        font-size: 1.08rem;
        max-width: 950px;
        line-height: 1.55;
    }

    .tag {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(96, 165, 250, 0.30);
        color: #BFDBFE;
        font-size: 0.82rem;
        margin-right: 8px;
        margin-top: 8px;
    }

    /* Metric cards */
    .metric-card {
        padding: 20px 22px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
        min-height: 112px;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.05;
    }

    .metric-note {
        color: #64748B;
        font-size: 0.82rem;
        margin-top: 8px;
    }

    /* Recommendation cards */
    .book-card {
        padding: 22px;
        border-radius: 22px;
        background:
            linear-gradient(135deg, rgba(30, 41, 59, 0.94), rgba(15, 23, 42, 0.94));
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
        margin-bottom: 16px;
    }

    .book-card h3 {
        color: #F9FAFB;
        margin-bottom: 8px;
        font-size: 1.18rem;
    }

    .book-card p {
        color: #CBD5E1;
        font-size: 0.98rem;
        line-height: 1.42;
        margin-bottom: 5px;
    }

    .score-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.14);
        border: 1px solid rgba(34, 197, 94, 0.28);
        color: #BBF7D0;
        font-size: 0.82rem;
        margin-top: 7px;
    }

    .ai-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(192, 132, 252, 0.30);
        color: #E9D5FF;
        font-size: 0.82rem;
        margin-top: 7px;
        margin-right: 7px;
    }

    .section-note {
        color: #94A3B8;
        font-size: 0.98rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #64748B;
        font-size: 0.86rem;
    }

    /* Streamlit default spacing tweaks */
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
    }

    div[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.94);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True
)



# LOAD DATA AND MODEL ARTIFACTS

@st.cache_data
def load_data():
    ratings = pd.read_csv(os.path.join(ARTIFACT_DIR, "ratings_clean.csv"))
    book_meta = pd.read_csv(os.path.join(ARTIFACT_DIR, "book_meta.csv"))
    user_activity = pd.read_csv(os.path.join(ARTIFACT_DIR, "user_activity.csv"))
    model_results = pd.read_csv(os.path.join(ARTIFACT_DIR, "model_results.csv"))

    with open(os.path.join(ARTIFACT_DIR, "config.json"), "r") as f:
        config = json.load(f)

    book_id_col = config["book_id_col"]
    user_col = config["user_col"]
    rating_book_col = config["rating_book_col"]

    ratings[user_col] = ratings[user_col].astype(str)
    ratings[rating_book_col] = ratings[rating_book_col].astype(str)
    book_meta[book_id_col] = book_meta[book_id_col].astype(str)

    return ratings, book_meta, user_activity, model_results, config


@st.cache_resource
def load_model():
    model_path = os.path.join(ARTIFACT_DIR, "selected_model.pkl")

    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        st.error(
            "The selected Surprise model could not be loaded. "
            "This usually means scikit-surprise is missing from requirements.txt."
        )
        st.code("Add this to requirements.txt:\nscikit-surprise==1.1.4", language="text")
        st.exception(e)
        st.stop()


ratings, book_meta, user_activity, model_results, config = load_data()
model = load_model()

book_id_col = config["book_id_col"]
title_col = config["title_col"]
author_col = config["author_col"]
year_col = config["year_col"]
avg_col = config["avg_col"]
language_col = config.get("language_col")
user_col = config["user_col"]
rating_book_col = config["rating_book_col"]
rating_col = config["rating_col"]
selected_model_name = config["selected_model_name"]


# HELPER FUNCTIONS


def safe_text(value, default="Unknown"):
    if pd.isna(value):
        return default
    return html.escape(str(value))


def safe_float(value, decimals=3, default="Unavailable"):
    try:
        if pd.isna(value):
            return default
        return f"{float(value):.{decimals}f}"
    except Exception:
        return default


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_text(label)}</div>
            <div class="metric-value">{safe_text(value)}</div>
            <div class="metric-note">{safe_text(note)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_top_n_for_user(
    model,
    ratings_df,
    book_meta_df,
    user_id,
    n=10,
    candidate_pool=1000
):
    user_id = str(user_id)

    rated_books = set(
        ratings_df.loc[
            ratings_df[user_col].astype(str) == user_id,
            rating_book_col
        ].astype(str)
    )

    popularity_ranked_books = (
        ratings_df[rating_book_col]
        .astype(str)
        .value_counts()
        .index
        .tolist()
    )

    candidates = [
        book_id
        for book_id in popularity_ranked_books
        if book_id not in rated_books
    ][:candidate_pool]

    if len(candidates) == 0:
        return pd.DataFrame()

    predictions = []

    for book_id in candidates:
        pred = model.predict(user_id, str(book_id))
        predictions.append((str(book_id), pred.est))

    recs = (
        pd.DataFrame(predictions, columns=[book_id_col, "cf_predicted_rating"])
        .sort_values("cf_predicted_rating", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    recs.insert(0, "cf_rank", range(1, len(recs) + 1))
    recs[book_id_col] = recs[book_id_col].astype(str)

    final_recs = recs.merge(
        book_meta_df,
        on=book_id_col,
        how="left"
    )

    return final_recs


def get_user_profile(user_id, ratings_df, book_meta_df, top_n=10):
    user_id = str(user_id)

    user_ratings = ratings_df[
        ratings_df[user_col].astype(str) == user_id
    ].copy()

    user_ratings = user_ratings.rename(
        columns={rating_book_col: book_id_col}
    )

    user_ratings[book_id_col] = user_ratings[book_id_col].astype(str)

    user_books = user_ratings.merge(
        book_meta_df,
        on=book_id_col,
        how="left"
    )

    profile = {
        "ratings_given": len(user_books),
        "average_rating": user_books[rating_col].mean(),
        "rating_std": user_books[rating_col].std()
    }

    favorite_books = (
        user_books
        .sort_values(rating_col, ascending=False)
        .head(top_n)
    )

    return profile, favorite_books, user_books


def format_catalog_for_prompt(recs):
    lines = []

    for i, row in recs.reset_index(drop=True).iterrows():
        title = row.get(title_col, "Unknown title")

        author = (
            row.get(author_col, "Unknown author")
            if author_col is not None
            else "Unknown author"
        )

        year = (
            row.get(year_col, "")
            if year_col is not None
            else ""
        )

        avg_rating = (
            row.get(avg_col, "")
            if avg_col is not None
            else ""
        )

        cf_score = row.get("cf_predicted_rating", "")
        cf_score_text = safe_float(cf_score, decimals=3)
        cf_rank = row.get("cf_rank", i + 1)

        lines.append(
            f"{i+1}. "
            f"book_id={row[book_id_col]}; "
            f"original_cf_rank={cf_rank}; "
            f"title={title}; "
            f"author={author}; "
            f"year={year}; "
            f"catalog_avg_rating={avg_rating}; "
            f"cf_predicted_rating={cf_score_text}"
        )

    return "\n".join(lines)


def build_rerank_prompt(recs, reader_preference):
    catalog = format_catalog_for_prompt(recs)

    prompt = f"""
You are an AI reading concierge.

Task:
Re-rank the candidate book recommendations according to the user's stated preference.

Rules:
1. You may ONLY choose from the candidate books listed below.
2. Do NOT invent new books.
3. Keep the original book_id values exactly as provided.
4. Return valid JSON only.
5. Rank the best personalized fit first.
6. Provide a short, specific explanation for each recommendation.
7. Include a match_score from 1 to 100.
8. Use the collaborative-filtering score as useful signal, but prioritize the user's stated preference.
9. Keep all recommendations grounded in the provided book metadata.

User preference:
{reader_preference}

Candidate books from the collaborative-filtering model:
{catalog}

Return JSON only in this exact format:
[
  {{
    "rank": 1,
    "book_id": "...",
    "title": "...",
    "match_score": 95,
    "reason": "Short personalized reason."
  }}
]
"""

    return prompt.strip()


def rerank_with_gemini(recs, reader_preference, model_name="gemini-2.5-flash-lite"):
    try:
        from google import genai
    except ImportError:
        st.error("Missing package: install with `pip install google-genai`.")
        st.stop()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

    if not api_key:
        st.error(
            "Missing GEMINI_API_KEY. Add it in Streamlit Cloud under "
            "Manage app > Settings > Secrets."
        )
        st.stop()

    client = genai.Client(api_key=api_key)
    prompt = build_rerank_prompt(recs, reader_preference)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    raw_text = response.text.strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        st.error("Gemini returned a response that could not be parsed as JSON.")
        st.text_area("Raw Gemini response", raw_text, height=250)
        st.stop()

    reranked_df = pd.DataFrame(parsed)

    required_cols = {"rank", "book_id", "title", "match_score", "reason"}
    missing_cols = required_cols - set(reranked_df.columns)

    if missing_cols:
        st.error(f"Gemini response is missing required columns: {missing_cols}")
        st.dataframe(reranked_df)
        st.stop()

    reranked_df[book_id_col] = reranked_df["book_id"].astype(str)

    final = reranked_df.merge(
        recs,
        on=book_id_col,
        how="left",
        suffixes=("_llm", "_cf")
    )

    return final



# CHART FUNCTIONS

def chart_model_results(model_results_df):
    df = model_results_df.copy()

    numeric_cols = ["RMSE", "Precision@10", "Recall@10"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    metric = st.radio(
        "Choose model metric",
        ["RMSE", "Precision@10", "Recall@10"],
        horizontal=True
    )

    if metric not in df.columns:
        st.info(f"{metric} is not available in model_results.csv.")
        return

    plot_df = df.dropna(subset=[metric]).copy()

    if plot_df.empty:
        st.info(f"No valid values available for {metric}.")
        return

    ascending = True if metric == "RMSE" else False
    plot_df = plot_df.sort_values(metric, ascending=ascending)

    fig = px.bar(
        plot_df,
        x=metric,
        y="Model",
        orientation="h",
        title=f"Model comparison by {metric}",
        text=metric,
        template="plotly_dark",
        color=metric,
        color_continuous_scale="Blues"
    )

    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=40, t=60, b=20),
        yaxis_title="",
        xaxis_title=metric,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, width="stretch")


def chart_rating_distribution():
    dist = (
        ratings[rating_col]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    dist.columns = ["Rating", "Count"]

    fig = px.bar(
        dist,
        x="Rating",
        y="Count",
        title="Rating distribution",
        template="plotly_dark",
        text="Count",
        color="Count",
        color_continuous_scale="Purples"
    )

    fig.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, width="stretch")


def chart_user_rating_distribution(user_books):
    user_dist = (
        user_books[rating_col]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    user_dist.columns = ["Rating", "Count"]

    fig = px.bar(
        user_dist,
        x="Rating",
        y="Count",
        title="Selected user's rating distribution",
        template="plotly_dark",
        text="Count",
        color="Count",
        color_continuous_scale="Teal"
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, width="stretch")


def chart_recommendation_scatter(cf_recs):
    if cf_recs.empty:
        return

    plot_df = cf_recs.copy()

    if avg_col is not None and avg_col in plot_df.columns:
        plot_df[avg_col] = pd.to_numeric(plot_df[avg_col], errors="coerce")
    else:
        plot_df["catalog_average_rating"] = np.nan

    plot_df["cf_predicted_rating"] = pd.to_numeric(
        plot_df["cf_predicted_rating"],
        errors="coerce"
    )

    hover_cols = [title_col]
    if author_col is not None and author_col in plot_df.columns:
        hover_cols.append(author_col)
    if year_col is not None and year_col in plot_df.columns:
        hover_cols.append(year_col)

    y_col = avg_col if avg_col is not None and avg_col in plot_df.columns else "catalog_average_rating"

    fig = px.scatter(
        plot_df,
        x="cf_predicted_rating",
        y=y_col,
        size="cf_predicted_rating",
        color="cf_rank",
        hover_data=hover_cols,
        title="Recommendation map: CF score vs catalog rating",
        template="plotly_dark",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="CF predicted rating",
        yaxis_title="Catalog average rating"
    )

    st.plotly_chart(fig, width="stretch")


def chart_top_recs_bar(cf_recs):
    if cf_recs.empty:
        return

    plot_df = cf_recs.copy()
    plot_df["label"] = plot_df[title_col].astype(str).str.slice(0, 38)
    plot_df = plot_df.sort_values("cf_predicted_rating", ascending=True)

    fig = px.bar(
        plot_df,
        x="cf_predicted_rating",
        y="label",
        orientation="h",
        title="Top recommendations by collaborative-filtering score",
        template="plotly_dark",
        text="cf_predicted_rating",
        color="cf_predicted_rating",
        color_continuous_scale="Blues"
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        height=max(400, 42 * len(plot_df)),
        margin=dict(l=20, r=40, t=60, b=20),
        yaxis_title="",
        xaxis_title="Predicted rating",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, width="stretch")


def chart_ai_match_scores(ai_recs):
    if ai_recs.empty:
        return

    plot_df = ai_recs.copy()
    plot_df["match_score"] = pd.to_numeric(plot_df["match_score"], errors="coerce")

    title_source = "title_llm" if "title_llm" in plot_df.columns else "title"
    if title_source not in plot_df.columns:
        title_source = title_col

    plot_df["label"] = plot_df[title_source].astype(str).str.slice(0, 38)
    plot_df = plot_df.sort_values("match_score", ascending=True)

    fig = px.bar(
        plot_df,
        x="match_score",
        y="label",
        orientation="h",
        title="AI personalized match scores",
        template="plotly_dark",
        text="match_score",
        color="match_score",
        color_continuous_scale="Plasma"
    )

    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(
        height=max(400, 42 * len(plot_df)),
        margin=dict(l=20, r=40, t=60, b=20),
        yaxis_title="",
        xaxis_title="AI match score",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, width="stretch")


# SIDEBAR CONTROLS

st.sidebar.markdown("## ⚙️ Recommendation Controls")

user_options = sorted(ratings[user_col].astype(str).unique())

selected_user = st.sidebar.selectbox(
    "Select Goodreads user",
    user_options
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

candidate_pool = st.sidebar.slider(
    "Candidate pool size",
    min_value=250,
    max_value=3000,
    value=1000,
    step=250,
    help="Larger pools give the model more candidates to score, but may run slower."
)

reader_preference = st.sidebar.text_area(
    "What are you in the mood for?",
    value="I want something emotionally engaging, thoughtful, and not too dark.",
    height=120
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **How it works**

    1. Select a Goodreads user  
    2. Generate CF recommendations  
    3. Optionally re-rank with Gemini  
    4. Compare model scores and AI explanations  
    """
)



# HERO SECTION


st.markdown(
    """
    <div class="hero">
        <h1>📚 BookMatch AI</h1>
        <p>
            A hybrid Goodreads recommendation system that combines collaborative filtering,
            model evaluation, interactive analytics, and Gemini-powered personalization.
            The recommender first generates grounded candidate books, then the AI layer
            re-ranks them based on the reader's current mood or preference.
        </p>
        <span class="tag">Collaborative Filtering</span>
        <span class="tag">Top-N Recommender</span>
        <span class="tag">Gemini Re-Ranking</span>
        <span class="tag">Interactive Analytics</span>
    </div>
    """,
    unsafe_allow_html=True
)


# PROFILE CALCULATION


profile, favorite_books, user_books = get_user_profile(
    user_id=selected_user,
    ratings_df=ratings,
    book_meta_df=book_meta,
    top_n=10
)


# TOP METRICS


m1, m2, m3, m4 = st.columns(4)

with m1:
    metric_card(
        "Selected model",
        selected_model_name,
        "Best fitted model selected from notebook evaluation"
    )

with m2:
    metric_card(
        "Users",
        f"{ratings[user_col].nunique():,}",
        "Unique Goodreads users"
    )

with m3:
    metric_card(
        "Ratings",
        f"{len(ratings):,}",
        "Cleaned explicit rating records"
    )

with m4:
    metric_card(
        "Selected user avg",
        safe_float(profile["average_rating"], decimals=2),
        f"{profile['ratings_given']:,} ratings given"
    )


# TABS


tab_overview, tab_reader, tab_recs, tab_ai = st.tabs(
    [
        "📊 Overview",
        "🧬 Reader Profile",
        "🎯 CF Recommendations",
        "✨ AI Re-Ranking"
    ]
)


# TAB 1: OVERVIEW


with tab_overview:
    st.markdown("### System overview")
    st.markdown(
        """
        <div class="section-note">
        This dashboard summarizes the recommendation system, model results, and rating behavior.
        RMSE measures rating-prediction accuracy, while Precision@10 and Recall@10 evaluate
        Top-N recommendation usefulness.
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([1.1, 0.9])

    with left:
        chart_model_results(model_results)

    with right:
        chart_rating_distribution()

    with st.expander("Raw model evaluation results"):
        st.dataframe(model_results, width="stretch")


# TAB 2: READER PROFILE


with tab_reader:
    st.markdown("### Reader DNA")
    st.markdown(
        """
        <div class="section-note">
        This section summarizes the selected user's historical rating behavior.
        The recommender uses this behavioral pattern to estimate which unread books
        are most likely to fit the user.
        </div>
        """,
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Ratings given", f"{profile['ratings_given']:,}")

    with r2:
        st.metric("Average rating", safe_float(profile["average_rating"], decimals=2))

    with r3:
        st.metric("Rating std dev", safe_float(profile["rating_std"], decimals=2))

    left, right = st.columns([0.9, 1.1])

    with left:
        chart_user_rating_distribution(user_books)

    with right:
        st.markdown("#### Highest-rated books")
        favorite_display_cols = [book_id_col, title_col]

        if author_col is not None:
            favorite_display_cols.append(author_col)

        if year_col is not None:
            favorite_display_cols.append(year_col)

        favorite_display_cols.append(rating_col)

        st.dataframe(
            favorite_books[favorite_display_cols],
            width="stretch",
            height=360
        )



# TAB 3: COLLABORATIVE FILTERING RECOMMENDATIONS


with tab_recs:
    st.markdown("### Collaborative-filtering Top-N recommendations")
    st.markdown(
        """
        <div class="section-note">
        The selected Surprise model scores candidate books the user has not already rated.
        The highest predicted ratings become the Top-N recommendation list.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚀 Generate collaborative-filtering recommendations", type="primary"):
        cf_recs = get_top_n_for_user(
            model=model,
            ratings_df=ratings,
            book_meta_df=book_meta,
            user_id=selected_user,
            n=top_n,
            candidate_pool=candidate_pool
        )

        if cf_recs.empty:
            st.warning("No recommendations could be generated for this user.")
        else:
            st.session_state["cf_recs"] = cf_recs

            if "ai_recs" in st.session_state:
                del st.session_state["ai_recs"]

    if "cf_recs" not in st.session_state:
        st.info("Click the button above to generate collaborative-filtering recommendations.")
    else:
        cf_recs = st.session_state["cf_recs"]

        c_left, c_right = st.columns([1.05, 0.95])

        with c_left:
            chart_top_recs_bar(cf_recs)

        with c_right:
            chart_recommendation_scatter(cf_recs)

        st.markdown("#### Recommendation cards")

        for idx, row in cf_recs.reset_index(drop=True).iterrows():
            title = safe_text(row.get(title_col, "Unknown Title"), "Unknown Title")

            author = (
                safe_text(row.get(author_col, "Unknown Author"), "Unknown Author")
                if author_col is not None
                else "Unknown Author"
            )

            year = (
                safe_text(row.get(year_col, ""), "")
                if year_col is not None
                else ""
            )

            score = safe_float(row.get("cf_predicted_rating", ""), decimals=3)

            avg_rating = (
                safe_float(row.get(avg_col, ""), decimals=2)
                if avg_col is not None
                else "Unavailable"
            )

            year_line = f"<p><b>Year:</b> {year}</p>" if year else ""

            st.markdown(
                f"""
                <div class="book-card">
                    <h3>#{idx + 1} — {title}</h3>
                    <p><b>Author:</b> {author}</p>
                    {year_line}
                    <p><b>Catalog average rating:</b> {avg_rating}</p>
                    <span class="score-pill">CF predicted rating: {score}</span>
                </div>
                """,
                unsafe_allow_html=True
            )



# TAB 4: AI RE-RANKING


with tab_ai:
    st.markdown("### Gemini AI re-ranking")
    st.markdown(
        """
        <div class="section-note">
        The AI layer does not invent new books. It receives the collaborative-filtering
        candidate list and re-ranks only those books based on the user's stated preference.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "cf_recs" not in st.session_state:
        st.info("Generate collaborative-filtering recommendations first, then return to this tab.")
    else:
        cf_recs = st.session_state["cf_recs"]

        st.markdown("#### Current personalization request")
        st.info(reader_preference)

        if st.button("✨ AI re-rank with Gemini", type="primary"):
            with st.spinner("Gemini is re-ranking the collaborative-filtering candidates..."):
                ai_recs = rerank_with_gemini(
                    recs=cf_recs,
                    reader_preference=reader_preference
                )

            st.session_state["ai_recs"] = ai_recs

        if "ai_recs" in st.session_state:
            ai_recs = st.session_state["ai_recs"]

            chart_ai_match_scores(ai_recs)

            st.markdown("#### AI-personalized recommendation cards")

            for _, row in ai_recs.sort_values("rank").iterrows():
                title = row.get(
                    "title_llm",
                    row.get(
                        "title",
                        row.get(title_col, "Unknown Title")
                    )
                )

                title = safe_text(title, "Unknown Title")
                reason = safe_text(row.get("reason", ""), "")
                match_score = safe_text(row.get("match_score", ""), "")
                cf_score = safe_float(row.get("cf_predicted_rating", ""), decimals=3)

                rank = int(row["rank"])

                st.markdown(
                    f"""
                    <div class="book-card">
                        <h3>#{rank} — {title}</h3>
                        <span class="ai-pill">AI match score: {match_score}/100</span>
                        <span class="score-pill">CF predicted rating: {cf_score}</span>
                        <p style="margin-top: 12px;"><b>Why this fits:</b> {reason}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with st.expander("View LLM prompt sent to Gemini"):
            st.code(build_rerank_prompt(cf_recs, reader_preference), language="text")
