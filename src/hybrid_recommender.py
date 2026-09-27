"""Hybrid recommendation system combining collaborative and content-based filtering.

This recommender combines:
1. Collaborative filtering: User-item interactions (KNN on sparse matrix)
2. Content-based filtering: Book metadata (genres, authors, ratings)

The hybrid approach provides:
- Better recommendations for books with sparse interaction data
- More diverse and explainable recommendations
- Cold-start handling for books with metadata but few ratings
"""

import time

import polars as pl
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

from src.knn_recommender_sparse import create_title_mapping
from src.matching import fuzzy_title_matches


class HybridRecommender:
    """Hybrid recommender combining collaborative and content-based filtering."""

    def __init__(
        self,
        interaction_matrix_path="data/book_user_matrix_sparse.npz",
        content_features_path="data/content_features.npz",
        book_mapping_path="data/sparse_matrix_book_mapping.parquet",
        content_mapping_path="data/content_features_mapping.parquet",
        metadata_path="data/book_metadata.parquet",
        collaborative_weight=0.6,
        content_weight=0.4,
        n_neighbors=20,
    ):
        """Initialize hybrid recommender.

        Args:
            interaction_matrix_path: Path to interaction sparse matrix
            content_features_path: Path to content features matrix
            book_mapping_path: Path to book ID mapping for interaction matrix
            content_mapping_path: Path to book ID mapping for content features
            metadata_path: Path to book metadata
            collaborative_weight: Weight for collaborative filtering (0-1)
            content_weight: Weight for content-based filtering (0-1)
            n_neighbors: Number of neighbors for KNN
        """
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.n_neighbors = n_neighbors

        # Load data
        print("Loading interaction matrix...")
        self.interaction_matrix = load_npz(interaction_matrix_path)

        print("Loading content features...")
        self.content_features = load_npz(content_features_path)

        print("Loading book mappings...")
        # Collaborative filtering book IDs (interaction matrix rows)
        collab_mapping = pl.read_parquet(book_mapping_path)
        self.collab_book_ids = collab_mapping.sort("matrix_idx")["book_id"].to_list()

        # Content features book IDs (content matrix rows)
        content_mapping = pl.read_parquet(content_mapping_path)
        self.content_book_ids = content_mapping.sort("matrix_idx")["book_id"].to_list()

        # Create bidirectional mappings between collaborative and content indices
        self.collab_to_content_idx = {}  # collab_idx -> content_idx
        self.content_to_collab_idx = {}  # content_idx -> collab_idx

        content_book_to_idx = {book_id: idx for idx, book_id in enumerate(self.content_book_ids)}

        for collab_idx, book_id in enumerate(self.collab_book_ids):
            if book_id in content_book_to_idx:
                content_idx = content_book_to_idx[book_id]
                self.collab_to_content_idx[collab_idx] = content_idx
                self.content_to_collab_idx[content_idx] = collab_idx

        print("Loading metadata...")
        metadata = pl.read_parquet(metadata_path)
        self.metadata_dict = {row["book_id"]: row for row in metadata.iter_rows(named=True)}

        # Titles via CSV→JSON bridge (titles.snap), not metadata alone.
        # Metadata used to be keyed by JSON id while the matrix uses CSV ids,
        # which produced mass "Book ID: …" placeholders and wrong titles.
        print("Creating title mappings...")
        self.title_to_idx, self.idx_to_title, _ = create_title_mapping(self.collab_book_ids)

        print(f"Initialized with {len(self.collab_book_ids):,} books")
        print(f"  Collaborative: {self.interaction_matrix.shape}")
        print(f"  Content: {self.content_features.shape}")
        print(
            f"  Index mapping coverage: {len(self.collab_to_content_idx):,}/{len(self.collab_book_ids):,} "
            f"({100 * len(self.collab_to_content_idx) / len(self.collab_book_ids):.1f}%)"
        )
        print(
            f"  Metadata coverage: {len(self.metadata_dict):,} ({100 * len(self.metadata_dict) / len(self.collab_book_ids):.1f}%)"
        )
        titled = sum(1 for t in self.idx_to_title.values() if not t.startswith("Book ID:"))
        print(f"  Title coverage: {titled:,} ({100 * titled / len(self.collab_book_ids):.1f}%)")

        # Initialize KNN models
        print("\nFitting collaborative KNN...")
        self.collaborative_knn = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1
        )
        self.collaborative_knn.fit(self.interaction_matrix)

        print("Fitting content-based KNN...")
        self.content_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1)
        self.content_knn.fit(self.content_features)

        print("Hybrid recommender ready!\n")

    def fuzzy_search(self, query, threshold=60, max_results=5):
        """Find books using fuzzy string matching.

        Backed by the shared :func:`src.matching.fuzzy_title_matches` helper so
        the hybrid app and the sparse-KNN CLI use identical matching semantics
        (substring-friendly via ``partial_ratio``).

        Args:
            query: Search query.
            threshold: Minimum fuzzy match score (0-100).
            max_results: Maximum number of results to return.

        Returns:
            ``[(title, idx, score), ...]`` sorted by descending score.
        """
        return fuzzy_title_matches(self.title_to_idx, query, threshold=threshold, max_results=max_results)

    def recommend_collaborative(self, book_idx, n_recommendations=5):
        """Get recommendations using collaborative filtering only.

        Args:
            book_idx: Index of the query book
            n_recommendations: Number of recommendations

        Returns:
            list: [(idx, distance), ...]
        """
        book_vector = self.interaction_matrix[book_idx]
        distances, indices = self.collaborative_knn.kneighbors(book_vector, n_neighbors=n_recommendations + 1)

        # Skip first result (query book itself)
        recommendations = [(indices[0][i], distances[0][i]) for i in range(1, len(indices[0]))]

        return recommendations

    def recommend_content(self, book_idx, n_recommendations=5):
        """Get recommendations using content-based filtering only.

        Args:
            book_idx: Index of the query book (in collaborative matrix space)
            n_recommendations: Number of recommendations

        Returns:
            list: [(collab_idx, distance), ...] - returns collaborative indices
        """
        # Map collaborative index to content index
        if book_idx not in self.collab_to_content_idx:
            # Book doesn't have content features, return empty
            return []

        content_idx = self.collab_to_content_idx[book_idx]
        book_vector = self.content_features[content_idx]
        distances, indices = self.content_knn.kneighbors(book_vector, n_neighbors=n_recommendations + 1)

        # Skip first result (query book itself) and map back to collaborative indices
        recommendations = []
        for i in range(1, len(indices[0])):
            content_neighbor_idx = indices[0][i]
            # Map content index back to collaborative index
            if content_neighbor_idx in self.content_to_collab_idx:
                collab_neighbor_idx = self.content_to_collab_idx[content_neighbor_idx]
                recommendations.append((collab_neighbor_idx, distances[0][i]))

        return recommendations

    def recommend_hybrid(self, book_idx, n_recommendations=10):
        """Get recommendations using hybrid approach.

        Combines collaborative and content-based scores with weighted average.

        Args:
            book_idx: Index of the query book
            n_recommendations: Number of recommendations

        Returns:
            list: [(idx, combined_score, collab_score, content_score), ...]
        """
        # Get more candidates from each method
        n_candidates = n_recommendations * 3

        # Collaborative filtering
        collab_recs = self.recommend_collaborative(book_idx, n_candidates)
        collab_scores = {idx: 1 - dist for idx, dist in collab_recs}  # Convert distance to similarity

        # Content-based filtering
        content_recs = self.recommend_content(book_idx, n_candidates)
        content_scores = {idx: 1 - dist for idx, dist in content_recs}

        # Combine scores
        all_indices = set(collab_scores.keys()) | set(content_scores.keys())
        all_indices.discard(book_idx)  # Remove query book

        combined_recommendations = []
        for idx in all_indices:
            collab_score = collab_scores.get(idx, 0)
            content_score = content_scores.get(idx, 0)

            # Weighted combination
            combined_score = self.collaborative_weight * collab_score + self.content_weight * content_score

            combined_recommendations.append((idx, combined_score, collab_score, content_score))

        # Sort by combined score
        combined_recommendations.sort(key=lambda x: x[1], reverse=True)

        return combined_recommendations[:n_recommendations]

    def print_recommendations(self, query, n_recommendations=5, method="hybrid", threshold=60, show_details=True):
        """Print recommendations in a readable format.

        Args:
            query: Book title to search for
            n_recommendations: Number of recommendations
            method: 'hybrid', 'collaborative', or 'content'
            threshold: Fuzzy match threshold
            show_details: Show recommendation details
        """
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print(f"Method: {method.upper()}")
        print(f"{'=' * 70}")

        # Find book
        matches = self.fuzzy_search(query, threshold=threshold)

        if not matches:
            print(f"No matches found for '{query}' (threshold={threshold})")
            return

        if len(matches) > 1:
            print(f"\nFound {len(matches)} matches:")
            for title, idx, score in matches[:5]:
                print(f"  {score}% - {title}")
            print()

        # Use best match
        best_title, book_idx, match_score = matches[0]
        print(f"Using: {best_title} (match: {match_score}%)")

        # Show book details
        book_id = self.collab_book_ids[book_idx]
        if book_id in self.metadata_dict:
            meta = self.metadata_dict[book_id]
            print(f"  Authors: {meta.get('authors', 'N/A')}")
            print(f"  Rating: {meta.get('average_rating', 0):.2f} ({meta.get('ratings_count', 0):,} ratings)")
            shelves = meta.get("shelves", "")
            if shelves:
                top_shelves = shelves.split(",")[:5]
                print(f"  Genres: {', '.join(top_shelves)}")

        # Get recommendations
        print(f"\nTop {n_recommendations} Recommendations:")
        print("-" * 70)

        if method == "hybrid":
            recommendations = self.recommend_hybrid(book_idx, n_recommendations)
            for i, (idx, combined, collab, content) in enumerate(recommendations, 1):
                title = self.idx_to_title.get(idx, f"Book ID: {self.collab_book_ids[idx]}")
                print(f"\n{i}. {title[:60]}")
                print(f"   Score: {combined:.3f} (collab: {collab:.3f}, content: {content:.3f})")

                if show_details and self.collab_book_ids[idx] in self.metadata_dict:
                    meta = self.metadata_dict[self.collab_book_ids[idx]]
                    print(f"   Authors: {meta.get('authors', 'N/A')[:50]}")
                    shelves = meta.get("shelves", "").split(",")[:3]
                    if shelves and shelves[0]:
                        print(f"   Genres: {', '.join(shelves)}")

        elif method == "collaborative":
            recommendations = self.recommend_collaborative(book_idx, n_recommendations)
            for i, (idx, dist) in enumerate(recommendations, 1):
                title = self.idx_to_title.get(idx, f"Book ID: {self.collab_book_ids[idx]}")
                print(f"{i}. {title[:60]}")
                print(f"   Similarity: {1 - dist:.3f}")

        elif method == "content":
            recommendations = self.recommend_content(book_idx, n_recommendations)
            for i, (idx, dist) in enumerate(recommendations, 1):
                title = self.idx_to_title.get(idx, f"Book ID: {self.collab_book_ids[idx]}")
                print(f"{i}. {title[:60]}")
                print(f"   Similarity: {1 - dist:.3f}")

        print("=" * 70)


def main():
    """Demo the hybrid recommender."""
    print("\n" + "=" * 70)
    print("HYBRID BOOK RECOMMENDATION SYSTEM")
    print("Collaborative Filtering + Content-Based Filtering")
    print("=" * 70 + "\n")

    start = time.time()

    # Initialize recommender
    recommender = HybridRecommender(collaborative_weight=0.6, content_weight=0.4, n_neighbors=30)

    init_time = time.time() - start
    print(f"Initialization time: {init_time:.2f}s\n")

    # Test queries
    test_queries = [
        "Harry Potter",
        "Lord of the Rings",
        "1984",
    ]

    for query in test_queries:
        recommender.print_recommendations(query, n_recommendations=5, method="hybrid", show_details=True)
        print()

    total_time = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"Total execution time: {total_time:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
