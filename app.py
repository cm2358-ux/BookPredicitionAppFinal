###############################################################
# STREAMLIT APP: app.py
###############################################################

import os
import json
import pickle
import html

import pandas as pd
import streamlit as st


###############################################################
# APP CONFIGURATION
###############################################################

ARTIFACT_DIR = "artifacts"

st.set_page_config(
    page_title="BookMatch AI",
    page_icon="📚",
    layout="wide"
)

st.markdown(
    """
    <style>
    .book-card {
        padding: 18px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #374151;
        margin-bottom: 15px;
    }
    .book-card h3 {
        color: #F9FAFB;
    }
    .book-card p {
        color: #D1D5DB;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


###############################################################
# LOAD DATA AND MODEL ARTIFACTS
###############################################################

@st.cache_data
def load_data():
    ratings = pd.read_csv(os.path.join(ARTIFACT_DIR, "ratings_clean.csv"))
    book_meta = pd.read_csv(os.path.join(ARTIFACT_DIR, "book_meta.csv"))
    user_activity = pd.read_csv(os.path.join(ARTIFACT_DIR, "user_activity.csv"))
    model_results = pd.read_csv(os.path.join(ARTIFACT_DIR, "model_results.csv"))

    with open(os.path.join(ARTIFACT_DIR, "config.json"), "r") as f:
        config = json.load(f)

    # Pull key column names from config
    book_id_col = config["book_id_col"]
    user_col = config["user_col"]
    rating_book_col = config["rating_book_col"]

    # Force IDs to string so merges and Surprise predictions work correctly
    ratings[user_col] = ratings[user_col].astype(str)
    ratings[rating_book_col] = ratings[rating_book_col].astype(str)
    book_meta[book_id_col] = book_meta[book_id_col].astype(str)

    return ratings, book_meta, user_activity, model_results, config


@st.cache_resource
def load_model():
    with open(os.path.join(ARTIFACT_DIR, "selected_model.pkl"), "rb") as f:
        return pickle.load(f)


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


###############################################################
# HELPER FUNCTIONS
###############################################################

def safe_text(value, default="Unknown"):
    """
    Safely formats text for HTML display.
    """
    if pd.isna(value):
        return default
    return html.escape(str(value))


def safe_float(value, decimals=3, default="Unavailable"):
    """
    Safely formats numeric values.
    """
    try:
        if pd.isna(value):
            return default
        return f"{float(value):.{decimals}f}"
    except Exception:
        return default


def get_top_n_for_user(
    model,
    ratings_df,
    book_meta_df,
    user_id,
    n=10,
    candidate_pool=1000
):
    """
    Generates Top-N collaborative-filtering recommendations for a selected user.

    Logic:
    - Identify books the user has already rated.
    - Create a candidate pool of popular books the user has not rated.
    - Use the selected Surprise model to predict the user's rating for each candidate.
    - Return the highest-scoring recommendations with book metadata.
    """

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
    """
    Builds a simple reader profile for the selected user.
    """

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

    return profile, favorite_books


def format_catalog_for_prompt(recs):
    """
    Converts collaborative-filtering recommendations into prompt context.

    The LLM may only re-rank these candidate books.
    It may not invent new books.
    """

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
    """
    Builds a strict prompt that forces the LLM to re-rank only the CF candidates.
    """

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


def get_gemini_api_key():
    """
    Reads Gemini API key from local environment variable or Streamlit Cloud secrets.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    try:
        return st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        return None


def rerank_with_gemini(recs, reader_preference, model_name="gemini-2.5-flash-lite"):
    """
    Uses Gemini to re-rank collaborative-filtering recommendations.

    The API key should be stored as:
    - Local environment variable: GEMINI_API_KEY
    - Or Streamlit Cloud secret: GEMINI_API_KEY
    """

    try:
        from google import genai
    except ImportError:
        st.error("Missing package: install with `pip install google-genai`.")
        st.stop()

    api_key = get_gemini_api_key()

    if not api_key:
        st.error(
            "Missing GEMINI_API_KEY. Add it as an environment variable locally "
            "or in Streamlit Cloud under Manage app > Settings > Secrets."
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


###############################################################
# STREAMLIT USER INTERFACE
###############################################################

st.title("📚 BookMatch AI")
st.caption("Collaborative filtering meets mood-aware AI personalization.")

st.sidebar.header("Recommendation Controls")

user_options = sorted(ratings[user_col].astype(str).unique())

selected_user = st.sidebar.selectbox(
    "Select Goodreads User",
    user_options
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

reader_preference = st.sidebar.text_area(
    "What are you in the mood for?",
    value="I want something emotionally engaging, thoughtful, and not too dark."
)


###############################################################
# MODEL SUMMARY SECTION
###############################################################

st.subheader("Model Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Model", selected_model_name)

with col2:
    st.metric("Total Users", f"{ratings[user_col].nunique():,}")

with col3:
    st.metric("Total Ratings", f"{len(ratings):,}")

with st.expander("Model Evaluation Results"):
    st.dataframe(model_results, width="stretch")


###############################################################
# USER PROFILE SECTION
###############################################################

st.subheader("Reader DNA")

profile, favorite_books = get_user_profile(
    user_id=selected_user,
    ratings_df=ratings,
    book_meta_df=book_meta,
    top_n=10
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Ratings Given", f"{profile['ratings_given']:,}")

with c2:
    st.metric("Average Rating", safe_float(profile["average_rating"], decimals=2))

with c3:
    st.metric("Rating Std Dev", safe_float(profile["rating_std"], decimals=2))

with st.expander("User's Highest-Rated Books"):
    favorite_display_cols = [book_id_col, title_col]

    if author_col is not None:
        favorite_display_cols.append(author_col)

    if year_col is not None:
        favorite_display_cols.append(year_col)

    favorite_display_cols.append(rating_col)

    st.dataframe(
        favorite_books[favorite_display_cols],
        width="stretch"
    )


###############################################################
# COLLABORATIVE FILTERING RECOMMENDATIONS
###############################################################

if st.button("Generate Collaborative Filtering Recommendations"):
    cf_recs = get_top_n_for_user(
        model=model,
        ratings_df=ratings,
        book_meta_df=book_meta,
        user_id=selected_user,
        n=top_n
    )

    if cf_recs.empty:
        st.warning("No recommendations could be generated for this user.")
    else:
        st.session_state["cf_recs"] = cf_recs

        # Clear prior AI results when new CF recommendations are generated
        if "ai_recs" in st.session_state:
            del st.session_state["ai_recs"]


if "cf_recs" in st.session_state:
    st.subheader("Collaborative Filtering Top-N Recommendations")

    cf_recs = st.session_state["cf_recs"]

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

        year_line = f"<p><b>Year:</b> {year}</p>" if year else ""

        st.markdown(
            f"""
            <div class="book-card">
                <h3>#{idx + 1} — {title}</h3>
                <p><b>Author:</b> {author}</p>
                {year_line}
                <p><b>CF Predicted Rating:</b> {score}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("AI Re-Rank With Gemini"):
        ai_recs = rerank_with_gemini(
            recs=cf_recs,
            reader_preference=reader_preference
        )

        st.session_state["ai_recs"] = ai_recs


###############################################################
# AI RE-RANKED RECOMMENDATIONS
###############################################################

if "ai_recs" in st.session_state:
    st.subheader("AI-Personalized Ranking")

    ai_recs = st.session_state["ai_recs"]

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
                <p><b>AI Match Score:</b> {match_score}/100</p>
                <p><b>CF Predicted Rating:</b> {cf_score}</p>
                <p><b>Why this fits:</b> {reason}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
