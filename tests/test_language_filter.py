"""Tests for English-first language filtering used by the Streamlit search UI."""

from __future__ import annotations

import pytest

from src.matching import (
    is_english_language,
    is_unknown_language,
    matches_language_filter,
    normalize_language_code,
)


class TestNormalizeLanguageCode:
    def test_strips_and_lowercases(self):
        assert normalize_language_code(" en-US ") == "en-us"

    def test_none_and_empty(self):
        assert normalize_language_code(None) == ""
        assert normalize_language_code("") == ""


class TestEnglishDetection:
    @pytest.mark.parametrize("code", ["eng", "en", "en-US", "en-GB", "en-CA", "en-AU", "ENG"])
    def test_modern_english_codes(self, code):
        assert is_english_language(code)

    @pytest.mark.parametrize("code", ["spa", "ger", "ita", "tur", "enm", "fre", ""])
    def test_non_english_codes(self, code):
        assert not is_english_language(code)

    def test_unknown(self):
        assert is_unknown_language("")
        assert is_unknown_language(None)
        assert not is_unknown_language("eng")


class TestMatchesLanguageFilter:
    def test_all_languages_passes_everything(self):
        assert matches_language_filter("spa", "All languages")
        assert matches_language_filter("", "All languages")
        assert matches_language_filter("eng", "All languages")

    def test_english_keeps_english_codes(self):
        assert matches_language_filter("eng", "English")
        assert matches_language_filter("en-US", "English")

    def test_english_drops_translations(self):
        assert not matches_language_filter("spa", "English")
        assert not matches_language_filter("ita", "English", include_unknown=True)

    def test_english_include_unknown(self):
        assert matches_language_filter("", "English", include_unknown=True)
        assert not matches_language_filter("", "English", include_unknown=False)

    def test_invalid_filter_raises(self):
        with pytest.raises(ValueError):
            matches_language_filter("eng", "Klingon")
