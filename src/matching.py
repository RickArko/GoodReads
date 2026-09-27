"""Fuzzy title-matching helpers shared across recommenders.

Both ``knn_recommender_sparse`` and ``hybrid_recommender`` previously
shipped their own copies of this logic, with one subtle but
user-facing difference: the sparse KNN matcher used
``fuzz.partial_ratio`` (substring-friendly: typing "Harry Potter"
matches "Harry Potter and the Sorcerer's Stone"), while the hybrid
matcher used ``fuzz.ratio`` (full-string similarity, which would not
match the same partial query).

Consolidating here on ``partial_ratio`` as the default keeps the
recommenders consistent and matches what most users expect when they
type a partial title into a search box. Callers that want the stricter
full-string semantics can still pass ``scorer="ratio"`` explicitly.
"""

from __future__ import annotations

from typing import Callable, Literal

from fuzzywuzzy import fuzz

Scorer = Literal["partial_ratio", "ratio"]
ScoreFn = Callable[[str, str], int]


def _score_fn(scorer: Scorer) -> ScoreFn:
    if scorer == "partial_ratio":
        return fuzz.partial_ratio
    if scorer == "ratio":
        return fuzz.ratio
    raise ValueError(f"Unknown scorer: {scorer!r}; expected 'partial_ratio' or 'ratio'.")


def fuzzy_title_matches(
    title_to_idx: dict[str, int],
    query: str,
    *,
    threshold: int = 60,
    max_results: int | None = None,
    scorer: Scorer = "partial_ratio",
) -> list[tuple[str, int, int]]:
    """Return ``(title, idx, score)`` tuples sorted by descending score.

    Args:
        title_to_idx: Mapping from book title to its matrix index.
        query: User-provided search string.
        threshold: Minimum fuzzy score (0-100) to include in results.
        max_results: If set, cap the result list at this length.
        scorer: ``"partial_ratio"`` for substring-friendly matching
            (recommended for free-text search), or ``"ratio"`` for full-string
            similarity.

    Returns:
        Matches sorted by score, descending. Empty list if nothing meets
        ``threshold``.
    """
    score_fn = _score_fn(scorer)
    query_lower = query.lower()
    matches: list[tuple[str, int, int]] = []
    for title, idx in title_to_idx.items():
        score = score_fn(title.lower(), query_lower)
        if score >= threshold:
            matches.append((title, int(idx), int(score)))
    matches.sort(key=lambda m: m[2], reverse=True)
    if max_results is not None:
        return matches[:max_results]
    return matches


def best_match_idx(
    title_to_idx: dict[str, int],
    query: str,
    *,
    threshold: int = 60,
    scorer: Scorer = "partial_ratio",
) -> int | None:
    """Return the matrix index of the best match, or ``None`` if no match clears ``threshold``."""
    matches = fuzzy_title_matches(title_to_idx, query, threshold=threshold, max_results=1, scorer=scorer)
    return matches[0][1] if matches else None


# Goodreads ``language_code`` values treated as modern English editions.
# Intentionally excludes ``enm`` (Middle English).
ENGLISH_LANGUAGE_CODES = frozenset(
    {
        "eng",
        "en",
        "en-us",
        "en-gb",
        "en-ca",
        "en-au",
    }
)


def normalize_language_code(code: str | None) -> str:
    """Normalize a Goodreads language code for comparison."""
    if code is None:
        return ""
    return str(code).strip().lower()


def is_english_language(code: str | None) -> bool:
    """Return True if ``code`` is a modern English Goodreads language code."""
    return normalize_language_code(code) in ENGLISH_LANGUAGE_CODES


def is_unknown_language(code: str | None) -> bool:
    """Return True when the catalog has no language code for the book."""
    return normalize_language_code(code) == ""


def matches_language_filter(
    code: str | None,
    language_filter: str,
    *,
    include_unknown: bool = True,
) -> bool:
    """Whether a book language passes the UI language filter.

    Args:
        code: Raw Goodreads ``language_code`` (may be empty/None).
        language_filter: ``\"English\"`` or ``\"All languages\"``.
        include_unknown: When filtering to English, keep books with blank
            language codes. Roughly half the UCSD catalog is unlabeled; most
            of those editions are English, while translated titles usually
            carry an explicit non-English code.
    """
    if language_filter == "All languages":
        return True
    if language_filter != "English":
        raise ValueError(f"Unknown language_filter: {language_filter!r}")
    if is_english_language(code):
        return True
    return include_unknown and is_unknown_language(code)
