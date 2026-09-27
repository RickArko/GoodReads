"""Extract book metadata from goodreads_books.json.gz for content-based filtering.

This script extracts:
- Authors
- Popular shelves (user-generated genres/tags)
- Other useful features

For the books in our sparse matrix.
"""

import gzip
import json

import polars as pl
from loguru import logger

from src.book_ids import load_matrix_id_bridges


def _safe_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_metadata(
    json_path="data/goodreads_books.json.gz",
    matrix_mapping_path="data/sparse_matrix_book_mapping.parquet",
    id_map_path="data/book_id_map.csv",
    output_path="data/book_metadata.parquet",
    top_n_shelves=10,
):
    """Extract metadata for books in the matrix.

    Sparse-matrix ``book_id`` values are interaction CSV ids. Book JSON and
    titles use a separate JSON id space, so this joins through
    ``book_id_map.csv`` and writes metadata keyed by the CSV id (matrix row id).

    Args:
        json_path: Path to goodreads_books.json.gz
        matrix_mapping_path: Path to sparse matrix book mapping
        id_map_path: Path to CSV↔JSON book id map
        output_path: Output path for extracted metadata
        top_n_shelves: Number of top shelves to keep per book
    """
    logger.info("Loading matrix book IDs and CSV↔JSON bridge...")
    mapping = pl.read_parquet(matrix_mapping_path)
    matrix_book_ids = set(mapping["book_id"].to_list())
    json_to_csv, _ = load_matrix_id_bridges(matrix_mapping_path, id_map_path)
    logger.info(
        f"Looking for metadata for {len(matrix_book_ids):,} matrix books "
        f"({len(json_to_csv):,} have a JSON id mapping)"
    )

    # Collect metadata
    metadata_records = []
    total_processed = 0
    found_count = 0

    logger.info(f"Scanning {json_path}...")
    with gzip.open(json_path, "rt", encoding="utf-8") as f:
        for line in f:
            total_processed += 1

            if total_processed % 500000 == 0:
                logger.info(f"Processed {total_processed:,} books, found {found_count:,} matches")

            book = json.loads(line)
            json_book_id = int(book["book_id"])

            csv_book_id = json_to_csv.get(json_book_id)
            if csv_book_id is None:
                continue

            found_count += 1

            # Extract authors
            authors = book.get("authors", [])
            author_ids = [a.get("author_id", "") for a in authors if a.get("author_id")]
            author_names = ", ".join([a.get("name", "") for a in authors if a.get("name", "")])[:500]

            # Extract popular shelves
            shelves = book.get("popular_shelves", [])

            # Sort by count and take top N
            shelves_sorted = sorted(shelves, key=lambda x: int(x.get("count", 0)), reverse=True)[:top_n_shelves]

            shelf_names = [s["name"] for s in shelves_sorted if s.get("name")]
            shelf_counts = [int(s["count"]) for s in shelves_sorted if s.get("count")]

            # Key by CSV id so joins to the sparse matrix succeed.
            record = {
                "book_id": csv_book_id,
                "book_id_json": json_book_id,
                "title": book.get("title", "")[:500],
                "authors": author_names,
                "author_ids": ",".join(author_ids),
                "num_authors": len(author_ids),
                "shelves": ",".join(shelf_names),
                "shelf_counts": ",".join(map(str, shelf_counts)),
                "num_shelves": len(shelf_names),
                "average_rating": _safe_float(book.get("average_rating", 0)),
                "ratings_count": _safe_int(book.get("ratings_count", 0)),
                "publication_year": _safe_int(book.get("publication_year", 0)),
                "num_pages": _safe_int(book.get("num_pages", 0)),
                "language_code": (book.get("language_code") or "")[:10],
            }

            metadata_records.append(record)

    logger.info(f"Total books processed: {total_processed:,}")
    logger.info(
        f"Metadata extracted for: {found_count:,} / {len(matrix_book_ids):,} books "
        f"({100 * found_count / len(matrix_book_ids):.1f}%)"
    )

    # Convert to DataFrame and save
    logger.info("Creating DataFrame...")
    df = pl.DataFrame(metadata_records)

    logger.info(f"Saving to {output_path}...")
    df.write_parquet(output_path)

    logger.info(f"Saved {len(df)} book metadata records")

    # Show statistics
    logger.info("\nMetadata Statistics:")
    logger.info(
        f"  Books with authors: {df.filter(pl.col('num_authors') > 0).height:,} ({100 * df.filter(pl.col('num_authors') > 0).height / len(df):.1f}%)"
    )
    logger.info(
        f"  Books with shelves: {df.filter(pl.col('num_shelves') > 0).height:,} ({100 * df.filter(pl.col('num_shelves') > 0).height / len(df):.1f}%)"
    )
    logger.info(f"  Average shelves per book: {df['num_shelves'].mean():.2f}")
    logger.info(f"  Average authors per book: {df['num_authors'].mean():.2f}")

    # Show top shelves
    all_shelves = []
    for shelves_str in df["shelves"].to_list():
        if shelves_str:
            all_shelves.extend(shelves_str.split(","))

    from collections import Counter

    shelf_counts = Counter(all_shelves)
    logger.info("\nTop 20 most common shelves:")
    for shelf, count in shelf_counts.most_common(20):
        logger.info(f"  {shelf:30s} {count:6,} books")

    return df


if __name__ == "__main__":
    extract_metadata()
