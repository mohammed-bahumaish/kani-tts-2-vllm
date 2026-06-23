"""Tests for tts_server/chunking.py — pure unit tests, no mocking needed."""

from tts_server.chunking import estimate_duration, split_into_sentences


class TestEstimateDuration:
    def test_empty(self):
        assert estimate_duration("") == 0.0

    def test_known_text(self):
        # "hello" = 5 chars / 15 chars_per_sec ≈ 0.333
        result = estimate_duration("hello")
        assert abs(result - 5 / 15) < 1e-6

    def test_custom_rate(self):
        result = estimate_duration("hello", chars_per_second=5.0)
        assert abs(result - 1.0) < 1e-6


class TestSplitIntoSentences:
    def test_single_sentence(self):
        text = "Hello world."
        result = split_into_sentences(text, max_duration_seconds=30)
        assert result == ["Hello world."]

    def test_sentence_split(self):
        text = "First sentence. Second sentence."
        result = split_into_sentences(text, max_duration_seconds=30)
        # Both fit in one chunk
        assert len(result) >= 1
        combined = " ".join(result)
        assert "First sentence." in combined
        assert "Second sentence." in combined

    def test_sentence_split_forced(self):
        # Two sentences that are each within limit but together exceed it
        # max_chars = 2 * 15 = 30
        s1 = "A" * 20 + "."
        s2 = "B" * 20 + "."
        text = f"{s1} {s2}"
        result = split_into_sentences(text, max_duration_seconds=2)
        assert len(result) >= 2

    def test_long_sentence_word_split(self):
        # One long sentence exceeding max_chars — must split on word boundaries
        words = ["word"] * 100
        text = " ".join(words)
        result = split_into_sentences(text, max_duration_seconds=2)
        # max_chars = 30, should produce multiple chunks
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 30 or len(chunk.split()) == 1  # single long word OK

    def test_short_sentences_grouped(self):
        # Many tiny sentences should be grouped together
        text = "A. B. C. D. E."
        result = split_into_sentences(text, max_duration_seconds=30)
        # All fit in one chunk
        assert len(result) == 1

    def test_empty_input(self):
        result = split_into_sentences("")
        assert result == []

    def test_preserves_content(self):
        text = "Hello world. This is a test. Final sentence."
        result = split_into_sentences(text, max_duration_seconds=30)
        recombined = " ".join(result)
        assert "Hello world." in recombined
        assert "This is a test." in recombined
        assert "Final sentence." in recombined
