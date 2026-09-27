"""Tests for CSV↔JSON book-id bridging."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.book_ids import is_untitled_fallback, load_matrix_id_bridges


def test_is_untitled_fallback():
    assert is_untitled_fallback("Book ID: 11145")
    assert is_untitled_fallback("")
    assert is_untitled_fallback(None)
    assert not is_untitled_fallback("Mere Christianity")


def test_load_matrix_id_bridges(tmp_path: Path):
    mapping_path = tmp_path / "sparse_matrix_book_mapping.parquet"
    id_map_path = tmp_path / "book_id_map.csv"

    pl.DataFrame({"matrix_idx": [0, 1, 2], "book_id": [10, 20, 30]}).write_parquet(mapping_path)
    pl.DataFrame(
        {
            "book_id_csv": [10, 20, 99],
            "book_id": [1000, 2000, 9900],
        }
    ).write_csv(id_map_path)

    json_to_csv, csv_to_json = load_matrix_id_bridges(mapping_path, id_map_path)

    assert json_to_csv == {1000: 10, 2000: 20}
    assert csv_to_json == {10: 1000, 20: 2000}
    assert 30 not in csv_to_json  # matrix book missing from id map
    assert 99 not in csv_to_json  # id-map book not in matrix
