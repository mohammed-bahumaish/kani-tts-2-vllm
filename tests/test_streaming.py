"""Tests for tts_server/streaming.py — StreamingAudioWriter with mock codec."""

from unittest.mock import patch

import numpy as np
import pytest

from tts_server.streaming import StreamingAudioWriter


class TestStreamingAudioWriter:
    def test_basic_decode(self, mock_codec):
        """Feed 40 tokens (10 frames at chunk_size=10) → on_chunk called."""
        chunks = []
        writer = StreamingAudioWriter(
            mock_codec, chunk_size=10, lookback_frames=15,
            on_chunk=lambda c: chunks.append(c),
        )
        writer.start()

        # 40 audio tokens = 10 frames (triggers decode at chunk_size=10)
        for _ in range(40):
            writer.add_token(64410)

        audio = writer.finalize()

        assert len(chunks) > 0
        assert all(isinstance(c, np.ndarray) for c in chunks)
        assert all(c.dtype == np.float32 for c in chunks)
        assert audio is not None
        assert len(audio) > 0

    def test_multiple_cycles(self, mock_codec):
        """Feed 80 tokens (20 frames) — should trigger multiple decode cycles."""
        chunks = []
        writer = StreamingAudioWriter(
            mock_codec, chunk_size=10, lookback_frames=15,
            on_chunk=lambda c: chunks.append(c),
        )
        writer.start()

        for _ in range(80):
            writer.add_token(64410)

        writer.finalize()
        # 20 frames at chunk_size=10 → at least 2 decodes
        assert len(chunks) >= 2

    def test_finalize_flushes_remaining(self, mock_codec):
        """Feed < chunk_size frames + END_OF_SPEECH → final decode triggered."""
        chunks = []
        writer = StreamingAudioWriter(
            mock_codec, chunk_size=10, lookback_frames=15,
            on_chunk=lambda c: chunks.append(c),
        )
        writer.start()

        # Only 20 tokens = 5 frames (< chunk_size=10)
        for _ in range(20):
            writer.add_token(64410)
        # END_OF_SPEECH triggers _decode(is_final=True) for remaining frames
        writer.add_token(mock_codec.end_of_speech)

        audio = writer.finalize()
        assert audio is not None
        assert len(audio) > 0

    def test_end_of_speech(self, mock_codec):
        """END_OF_SPEECH token triggers final decode."""
        chunks = []
        writer = StreamingAudioWriter(
            mock_codec, chunk_size=10, lookback_frames=15,
            on_chunk=lambda c: chunks.append(c),
        )
        writer.start()

        # Feed some tokens then END_OF_SPEECH
        for _ in range(20):
            writer.add_token(64410)
        writer.add_token(mock_codec.end_of_speech)

        writer.finalize()
        assert len(chunks) > 0

    def test_running_false_exits(self, mock_codec):
        """Setting running=False causes thread to exit without crash."""
        writer = StreamingAudioWriter(
            mock_codec, chunk_size=10, lookback_frames=15,
        )
        writer.start()
        writer.running = False
        result = writer.finalize()
        # No tokens fed, should return None
        assert result is None

    def test_no_tokens(self, mock_codec):
        """Start + finalize immediately with no tokens."""
        writer = StreamingAudioWriter(
            mock_codec, chunk_size=10, lookback_frames=15,
        )
        writer.start()
        result = writer.finalize()
        assert result is None

    def test_finalize_join_timeout(self, mock_codec):
        """finalize() calls join(timeout=30) and warns if thread hangs."""
        writer = StreamingAudioWriter(
            mock_codec, chunk_size=10, lookback_frames=15,
        )
        writer.start()
        # Feed tokens so thread is active
        for _ in range(40):
            writer.add_token(64410)

        with patch.object(writer._decoder_thread, "join", wraps=writer._decoder_thread.join) as mock_join:
            writer.finalize()
            mock_join.assert_called_once_with(timeout=30)
