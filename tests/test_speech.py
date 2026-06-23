"""Tests for POST /v1/audio/speech — all 4 modes + error handling."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
class TestFullWav:
    """Non-streaming mode (no stream_format) — returns WAV or PCM."""

    async def test_wav_response(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "voice": "random"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content[:4] == b"RIFF"
        assert len(resp.content) > 44  # WAV header is 44 bytes

    async def test_pcm_response(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "voice": "random", "response_format": "pcm"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["x-sample-rate"] == "22050"
        assert resp.headers["x-channels"] == "1"
        assert resp.headers["x-bit-depth"] == "16"
        assert len(resp.content) > 0

    async def test_random_voice_succeeds(self, client):
        """voice='random' skips speaker lookup entirely."""
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "voice": "random"},
        )
        assert resp.status_code == 200

    async def test_unknown_voice_400(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Test", "voice": "nonexistent"},
        )
        assert resp.status_code == 400
        assert "Unknown voice" in resp.json()["detail"]

    async def test_known_voice_succeeds(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "voice": "speaker_1"},
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
class TestSSE:
    """SSE streaming mode (stream_format=sse)."""

    async def test_sse_response_headers(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "voice": "random", "stream_format": "sse"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"

    async def test_sse_events(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "voice": "random", "stream_format": "sse"},
        )
        assert resp.status_code == 200

        # Parse SSE events from response body
        body = resp.text
        events = []
        for line in body.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        # Should have at least one audio delta and one done event
        types = [e["type"] for e in events]
        assert "speech.audio.done" in types

        # Done event should have usage stats
        done_events = [e for e in events if e["type"] == "speech.audio.done"]
        assert len(done_events) == 1
        usage = done_events[0].get("usage", {})
        assert "audio_tokens" in usage
        assert "audio_duration" in usage
        assert "generation_time" in usage

    async def test_sse_audio_delta_has_base64(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "voice": "random", "stream_format": "sse"},
        )
        body = resp.text
        events = []
        for line in body.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        deltas = [e for e in events if e["type"] == "speech.audio.delta"]
        if deltas:
            # Each delta should have valid base64 audio
            for delta in deltas:
                audio_bytes = base64.b64decode(delta["audio"])
                assert len(audio_bytes) > 0


@pytest.mark.asyncio
class TestPCMStream:
    """Raw PCM streaming mode (stream_format=audio)."""

    async def test_pcm_stream_response(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "voice": "random", "stream_format": "audio"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["x-sample-rate"] == "22050"
        assert resp.headers["x-channels"] == "1"
        assert resp.headers["x-bit-depth"] == "16"

    async def test_pcm_stream_content(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "voice": "random", "stream_format": "audio"},
        )
        assert resp.status_code == 200
        # PCM int16 = 2 bytes per sample, length should be even
        assert len(resp.content) % 2 == 0


@pytest.mark.asyncio
class TestSpeechErrors:
    async def test_engine_none_503(self, client):
        from tts_server.server import app
        original = app.state.engine
        app.state.engine = None
        try:
            resp = await client.post(
                "/v1/audio/speech",
                json={"input": "Hello", "voice": "random"},
            )
            assert resp.status_code == 503
        finally:
            app.state.engine = original

    async def test_input_too_long_422(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "x" * 10001, "voice": "random"},
        )
        assert resp.status_code == 422

    async def test_engine_error_500(self, client, mock_engine):
        """Engine raising RuntimeError in full mode → 500."""
        from tts_server.server import app

        async def _failing_gen(text, speaker_emb=None, speaker_proj_weight=None, sampling_params=None):
            raise RuntimeError("GPU exploded")
            # Make it an async generator
            yield  # pragma: no cover

        mock_engine.generate_stream = _failing_gen
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "voice": "random"},
        )
        assert resp.status_code == 500

    async def test_generation_timeout_500(self, client, mock_engine):
        """Engine that hangs → timeout → 500."""
        async def _hanging_gen(text, speaker_emb=None, speaker_proj_weight=None, sampling_params=None):
            await asyncio.sleep(10)
            yield 64410  # pragma: no cover

        mock_engine.generate_stream = _hanging_gen
        with patch("tts_server.server.GENERATION_TIMEOUT", 0.5):
            resp = await client.post(
                "/v1/audio/speech",
                json={"input": "Hello", "voice": "random"},
            )
        assert resp.status_code == 500
        assert "timeout" in resp.json()["detail"].lower()
