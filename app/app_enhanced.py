"""Enhanced Book Recommendation Streamlit App.

Features:
- Hybrid recommender (collaborative + content-based)
- Tunable weights for hybrid approach
- Better fuzzy matching
- Evaluation metrics
- Visualizations
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fuzzywuzzy import process

from src.book_ids import is_untitled_fallback
from src.hybrid_recommender import HybridRecommender
from src.matching import matches_language_filter


@st.cache_resource
def load_hybrid_recommender(collab_weight=0.6, content_weight=0.4):
    """Load and cache the hybrid recommender model."""
    try:
        recommender = HybridRecommender(
            collaborative_weight=collab_weight, content_weight=content_weight, n_neighbors=30
        )
        return recommender
    except Exception as e:
        st.error(f"Error loading recommender: {e}")
        st.info("Make sure you've run the data preparation scripts.")
        return None


@st.cache_data
def load_book_metadata_extended():
    """Load extended book metadata including genres and authors."""
    try:
        metadata_df = pl.read_parquet("data/book_metadata.parquet")
        return metadata_df
    except Exception as e:
        st.warning(f"Could not load extended metadata: {e}")
        return None


def smart_search(query, available_titles, threshold=70, max_results=10):
    """Improved fuzzy search with better ranking."""
    if not query:
        return []

    # Use fuzzywuzzy's process.extract for better fuzzy matching
    matches = process.extract(query, available_titles, limit=max_results)

    # Filter by threshold and return
    filtered_matches = [(title, score) for title, score in matches if score >= threshold]

    return filtered_matches


def calculate_diversity_score(recommendations, metadata_dict, recommender):
    """Calculate diversity of recommendations based on genres."""
    if not recommendations or not metadata_dict:
        return 0.0

    genres_lists = []
    for idx, _, _, _ in recommendations:
        book_id = recommender.collab_book_ids[idx]
        if book_id in metadata_dict:
            meta = metadata_dict[book_id]
            shelves = meta.get("shelves", "")
            if shelves:
                genres = set(shelves.split(",")[:5])  # Top 5 genres
                genres_lists.append(genres)

    if not genres_lists:
        return 0.0

    # Calculate diversity as average pairwise distance
    total_pairs = 0
    total_difference = 0

    for i in range(len(genres_lists)):
        for j in range(i + 1, len(genres_lists)):
            intersection = len(genres_lists[i] & genres_lists[j])
            union = len(genres_lists[i] | genres_lists[j])
            if union > 0:
                jaccard_similarity = intersection / union
                total_difference += 1 - jaccard_similarity
                total_pairs += 1

    if total_pairs == 0:
        return 0.0

    diversity = (total_difference / total_pairs) * 100
    return diversity


def plot_score_breakdown(recommendations, method="hybrid"):
    """Create a plotly chart showing score breakdown."""
    if method != "hybrid" or not recommendations:
        return None

    titles = [f"Book {i+1}" for i in range(len(recommendations))]
    collab_scores = [collab * 100 for _, _, collab, content in recommendations]
    content_scores = [content * 100 for _, _, collab, content in recommendations]

    fig = go.Figure()

    fig.add_trace(go.Bar(name="Collaborative", x=titles, y=collab_scores, marker_color="#1f77b4"))

    fig.add_trace(go.Bar(name="Content", x=titles, y=content_scores, marker_color="#ff7f0e"))

    fig.update_layout(
        title="Recommendation Score Breakdown",
        xaxis_title="Recommendations",
        yaxis_title="Score (%)",
        barmode="stack",
        height=400,
        showlegend=True,
    )

    return fig


def plot_genre_distribution(recommendations, metadata_dict, recommender):
    """Create a chart showing genre distribution of recommendations."""
    if not recommendations or not metadata_dict:
        return None

    genre_counts = {}

    for idx, _, _, _ in recommendations:
        book_id = recommender.collab_book_ids[idx]
        if book_id in metadata_dict:
            meta = metadata_dict[book_id]
            shelves = meta.get("shelves", "")
            if shelves:
                for genre in shelves.split(",")[:5]:  # Top 5 genres per book
                    genre = genre.strip()
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

    if not genre_counts:
        return None

    # Sort and get top 10
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    genres, counts = zip(*sorted_genres)

    fig = px.bar(
        x=list(counts),
        y=list(genres),
        orientation="h",
        title="Top Genres in Recommendations",
        labels={"x": "Count", "y": "Genre"},
        color=list(counts),
        color_continuous_scale="Blues",
    )

    fig.update_layout(height=400, showlegend=False)

    return fig


def main():
    st.set_page_config(page_title="Enhanced Book Recommender", page_icon="📚", layout="wide")

    # Header
    st.title("📚 Enhanced Book Recommendation System")
    st.markdown("Hybrid collaborative filtering + content-based recommendations with advanced features")

    # Sidebar - Model Settings
    st.sidebar.header("🎛️ Model Settings")

    recommendation_method = st.sidebar.radio(
        "Recommendation Method",
        ["Hybrid", "Collaborative Only", "Content Only"],
        index=0,
        help="Hybrid combines user interactions with book features (genres, authors)",
    )

    # Hybrid weight tuning
    collab_weight = 0.5
    content_weight = 0.5

    if recommendation_method == "Hybrid":
        st.sidebar.subheader("Hybrid Weights")
        collab_weight = st.sidebar.slider(
            "Collaborative Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Higher = more weight on user interaction data",
        )
        content_weight = 1.0 - collab_weight
        st.sidebar.caption(f"Content Weight: {content_weight:.2f}")

        # Visual indicator
        weights_df = pd.DataFrame({"Type": ["Collaborative", "Content"], "Weight": [collab_weight, content_weight]})
        fig_weights = px.pie(
            weights_df,
            values="Weight",
            names="Type",
            title="Current Weights",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"],
        )
        fig_weights.update_layout(height=250, showlegend=True)
        st.sidebar.plotly_chart(fig_weights, width="stretch")

    # Load data
    with st.spinner("Loading recommendation model..."):
        recommender = load_hybrid_recommender(collab_weight, content_weight)
        metadata_df = load_book_metadata_extended()

        if metadata_df is not None and not metadata_df.is_empty():
            metadata_dict = {row["book_id"]: row for row in metadata_df.iter_rows(named=True)}
        else:
            metadata_dict = {}

    if recommender is None:
        st.error("Failed to load recommender. Please check the data files.")
        return

    # Sidebar - Recommendation Settings
    st.sidebar.header("⚙️ Settings")

    n_recommendations = st.sidebar.slider("Number of recommendations", min_value=5, max_value=50, value=10, step=5)

    fuzzy_threshold = st.sidebar.slider(
        "Search threshold", min_value=50, max_value=100, value=70, help="Lower = more flexible matching"
    )

    # Sidebar - Filters
    st.sidebar.header("🔍 Filters")

    min_rating = None
    min_year = None
    max_year = None

    if metadata_dict:
        filter_by_rating = st.sidebar.checkbox("Filter by minimum rating")
        if filter_by_rating:
            min_rating = st.sidebar.slider("Minimum rating", 0.0, 5.0, 3.5, 0.1)

        filter_by_year = st.sidebar.checkbox("Filter by publication year")
        if filter_by_year:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                min_year = st.number_input("From", 1800, 2030, 1990)
            with col2:
                max_year = st.number_input("To", 1800, 2030, 2024)

    # Main content - Search
    st.header("🔍 Find a Book")

    # Top-line language setting (English-first for the demo audience)
    lang_col, search_col, btn_col = st.columns([1.3, 3.2, 1])

    with lang_col:
        language_filter = st.selectbox(
            "Language",
            options=["English", "All languages"],
            index=0,
            help=(
                "Limit search and recommendations to English editions. "
                "Translated titles (e.g. Harry Potter in Turkish/Italian/German) "
                "are excluded. Books with no language tag are kept by default."
            ),
        )

    with search_col:
        search_query = st.text_input(
            "Search for a book:",
            placeholder="Start typing... (e.g., 'Harry Potter', '1984', 'Lord of the Rings')",
            key="search",
        )

    with btn_col:
        st.markdown("<div style='height: 1.7rem'></div>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary", width="stretch")

    include_unknown_language = True
    if language_filter == "English" and metadata_dict:
        include_unknown_language = st.checkbox(
            "Include books with unknown language",
            value=True,
            help=(
                "About half the catalog has no language_code. Most unlabeled "
                "editions are English; uncheck to require an explicit eng/en-* tag."
            ),
        )

    # Title → language from metadata (via collab matrix book_id)
    title_to_language = {}
    for title, idx in recommender.title_to_idx.items():
        book_id = recommender.collab_book_ids[idx]
        title_to_language[title] = metadata_dict.get(book_id, {}).get("language_code", "")

    all_titles = sorted(recommender.title_to_idx.keys())
    available_titles = [
        title
        for title in all_titles
        if matches_language_filter(
            title_to_language.get(title, ""),
            language_filter,
            include_unknown=include_unknown_language,
        )
    ]

    if language_filter == "English":
        st.caption(
            f"Searching {len(available_titles):,} English"
            f"{' + unlabeled' if include_unknown_language else ''} titles "
            f"(of {len(all_titles):,} total). Switch to All languages for translations."
        )

    # Show search results
    if search_query:
        matches = smart_search(search_query, available_titles, threshold=fuzzy_threshold)

        if matches:
            st.info(f"Found {len(matches)} matches")
            selected_book = st.selectbox(
                "Select a book:",
                options=[title for title, score in matches],
                format_func=lambda x: f"{x} ({[s for t, s in matches if t == x][0]}% match)",
            )
        else:
            st.warning(
                "No matches found. Try lowering the search threshold"
                + (", switching Language to All languages," if language_filter == "English" else "")
                + " or a different query."
            )
            selected_book = None
    else:
        selected_book = None

    # Get recommendations
    if selected_book or search_button:
        if not selected_book:
            st.warning("Please search for and select a book first.")
            return

        with st.spinner(f"Generating recommendations for '{selected_book}'..."):
            # Over-fetch when language-filtering so translated neighbors don't
            # shrink the final list below the requested count.
            fetch_n = n_recommendations
            if language_filter == "English":
                fetch_n = min(max(n_recommendations * 5, n_recommendations), 100)

            # Get recommendations based on method
            if recommendation_method == "Hybrid":
                recommendations = recommender.recommend_hybrid(recommender.title_to_idx[selected_book], fetch_n)
            elif recommendation_method == "Collaborative Only":
                recs = recommender.recommend_collaborative(recommender.title_to_idx[selected_book], fetch_n)
                # Convert to hybrid format (idx, combined_score, collab, content)
                recommendations = [(idx, 1 - dist, 1 - dist, 0) for idx, dist in recs]
            else:  # Content Only
                recs = recommender.recommend_content(recommender.title_to_idx[selected_book], fetch_n)
                recommendations = [(idx, 1 - dist, 0, 1 - dist) for idx, dist in recs]

        if not recommendations:
            st.error("No recommendations found.")
            return

        # Filter recommendations
        filtered_recs = []
        for idx, combined, collab, content in recommendations:
            book_id = recommender.collab_book_ids[idx]
            title = recommender.idx_to_title.get(idx, "")
            book_info = metadata_dict.get(book_id, {})

            # Skip matrix rows with no resolvable title / metadata body.
            if is_untitled_fallback(title) or not book_info:
                continue

            # Apply filters
            if not matches_language_filter(
                book_info.get("language_code", ""),
                language_filter,
                include_unknown=include_unknown_language,
            ):
                continue

            if min_rating and book_info.get("average_rating"):
                if book_info["average_rating"] < min_rating:
                    continue

            if min_year and book_info.get("publication_year"):
                if book_info["publication_year"] < min_year:
                    continue

            if max_year and book_info.get("publication_year"):
                if book_info["publication_year"] > max_year:
                    continue

            filtered_recs.append((idx, combined, collab, content))
            if len(filtered_recs) >= n_recommendations:
                break

        if not filtered_recs:
            st.warning("No recommendations match your filters.")
            return

        # Display results
        st.success(f"Found {len(filtered_recs)} recommendations!")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "📊 Analytics", "🎯 Metrics"])

        with tab1:
            st.markdown(f"### Top {len(filtered_recs)} Recommendations")

            for rank, (idx, combined, collab, content) in enumerate(filtered_recs, 1):
                book_id = recommender.collab_book_ids[idx]
                title = recommender.idx_to_title[idx]
                book_info = metadata_dict.get(book_id, {})

                with st.expander(f"**{rank}. {title}**", expanded=(rank <= 3)):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        # Book details
                        if book_info:
                            authors = book_info.get("authors", "N/A")
                            st.markdown(f"**Authors:** {authors[:100]}")

                            rating = book_info.get("average_rating")
                            ratings_count = book_info.get("ratings_count")
                            if rating:
                                st.markdown(f"**Rating:** ⭐ {rating:.2f} ({ratings_count:,} ratings)")

                            year = book_info.get("publication_year")
                            pages = book_info.get("num_pages")
                            lang = book_info.get("language_code") or "unknown"
                            if year:
                                st.markdown(f"**Published:** {int(year)}")
                            if pages:
                                st.markdown(f"**Pages:** {int(pages)}")
                            st.markdown(f"**Language:** {lang}")

                            shelves = book_info.get("shelves", "")
                            if shelves:
                                top_genres = shelves.split(",")[:5]
                                st.markdown(f"**Genres:** {', '.join(top_genres)}")

                    with col2:
                        # Scores
                        st.metric("Combined", f"{combined*100:.1f}%")
                        if recommendation_method == "Hybrid":
                            st.caption(f"Collab: {collab*100:.0f}%")
                            st.caption(f"Content: {content*100:.0f}%")

        with tab2:
            st.markdown("### Recommendation Analytics")

            # Score breakdown chart
            if recommendation_method == "Hybrid":
                fig_scores = plot_score_breakdown(filtered_recs[:10], method="hybrid")
                if fig_scores:
                    st.plotly_chart(fig_scores, width="stretch")

            # Genre distribution
            fig_genres = plot_genre_distribution(filtered_recs, metadata_dict, recommender)
            if fig_genres:
                st.plotly_chart(fig_genres, width="stretch")

        with tab3:
            st.markdown("### Recommendation Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                avg_score = sum(combined for _, combined, _, _ in filtered_recs) / len(filtered_recs)
                st.metric("Average Score", f"{avg_score*100:.1f}%")

            with col2:
                diversity = calculate_diversity_score(filtered_recs, metadata_dict, recommender)
                st.metric("Diversity", f"{diversity:.1f}%", help="Higher = more genre variety")

            with col3:
                coverage = (len(filtered_recs) / n_recommendations) * 100
                st.metric("Filter Pass Rate", f"{coverage:.0f}%")

            # Distribution of scores
            scores = [combined * 100 for _, combined, _, _ in filtered_recs]
            fig_hist = px.histogram(
                x=scores,
                nbins=20,
                title="Distribution of Recommendation Scores",
                labels={"x": "Score (%)", "y": "Count"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, width="stretch")


if __name__ == "__main__":
    main()
