"""Helpers for the UCSD Goodreads dual book-id scheme.

The interaction CSV uses ``book_id_csv``; the books JSON / titles parquet use
``book_id`` (JSON id). Sparse-matrix rows are keyed by CSV ids, so any join to
titles or ``goodreads_books.json.gz`` must go through ``book_id_map.csv``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_matrix_id_bridges(
    matrix_mapping_path: str | Path = "data/sparse_matrix_book_mapping.parquet",
    id_map_path: str | Path = "data/book_id_map.csv",
) -> tuple[dict[int, int], dict[int, int]]:
    """Build CSV↔JSON id maps restricted to books present in the sparse matrix.

    Returns:
        ``(json_to_csv, csv_to_json)`` for matrix books that appear in the id map.
    """
    mapping = pl.read_parquet(matrix_mapping_path)
    matrix_csv_ids = mapping["book_id"].unique()

    id_map = pl.read_csv(id_map_path).select(
        pl.col("book_id_csv").cast(pl.Int64),
        pl.col("book_id").cast(pl.Int64).alias("book_id_json"),
    )
    bridged = id_map.join(
        matrix_csv_ids.to_frame("book_id_csv"),
        on="book_id_csv",
        how="inner",
    )

    json_ids = bridged["book_id_json"].to_list()
    csv_ids = bridged["book_id_csv"].to_list()
    json_to_csv = {int(j): int(c) for j, c in zip(json_ids, csv_ids)}
    csv_to_json = {int(c): int(j) for j, c in zip(json_ids, csv_ids)}
    return json_to_csv, csv_to_json


def is_untitled_fallback(title: str | None) -> bool:
    """True for the ``Book ID: {csv_id}`` placeholder used when title lookup fails."""
    if not title:
        return True
    return title.startswith("Book ID:")
