"""
Tests for the PreModelPipeline.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import PreModelPipeline


class TestNormalization:
    """Test the Roman normalization layer."""

    def setup_method(self):
        self.pipeline = PreModelPipeline(corpus_paths=None)

    def test_double_vowel_compression(self):
        assert self.pipeline.normalize_roman("aaa") == "a"
        assert self.pipeline.normalize_roman("eee") == "i"
        assert self.pipeline.normalize_roman("ooo") == "u"

    def test_consonant_normalization(self):
        assert self.pipeline.normalize_roman("phone") == "fone"

    def test_lowercase_strip(self):
        assert self.pipeline.normalize_roman("  NAMASTE  ") == "namaste"

    def test_special_chars_removed(self):
        assert self.pipeline.normalize_roman("hello123!") == "hello"

    def test_mixed_normalization(self):
        assert self.pipeline.normalize_roman("phool") == "ful"


class TestLanguageDetection:
    """Test the English word bypass."""

    def setup_method(self):
        self.pipeline = PreModelPipeline(corpus_paths=None)

    def test_english_word_detected(self):
        assert self.pipeline.is_english("phone") is True
        assert self.pipeline.is_english("laptop") is True
        assert self.pipeline.is_english("internet") is True

    def test_short_words_not_detected(self):
        # Words <= 3 chars are not bypassed
        assert self.pipeline.is_english("hi") is False
        assert self.pipeline.is_english("an") is False

    def test_hindi_words_not_detected(self):
        assert self.pipeline.is_english("namaste") is False
        assert self.pipeline.is_english("kisan") is False
