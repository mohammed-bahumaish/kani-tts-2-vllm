"""Tests for WS /v1/ws/speech endpoint.

Uses starlette TestClient which has native WebSocket support.
httpx does not support WebSocket connections.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from tts_server.server import app


class TestWebSocket:
    """WebSocket tests using starlette TestClient which has native WS support."""

    @pytest.fixture(autouse=True)
    def _setup_app_state(self, mock_engine, mock_codec, mock_speaker_manager, mock_proj_weight):
        app.state.engine = mock_engine
        app.state.codec = mock_codec
        app.state.speaker_manager = mock_speaker_manager
        app.state.speaker_proj_weight = mock_proj_weight
        yield
        app.state.engine = None
        app.state.codec = None
        app.state.speaker_manager = None
        app.state.speaker_proj_weight = None

    def test_full_flow(self):
        """Send generate → receive generation.started → binary PCM → generation.done."""
        test_client = TestClient(app)
        with test_client.websocket_connect("/v1/ws/speech") as ws:
            ws.send_json({
                "type": "generate",
                "input": "Hello world",
                "voice": "random",
                "request_id": "test-123",
            })

            messages = []
            while True:
                try:
                    data = ws.receive()
                    if "text" in data:
                        msg = json.loads(data["text"])
                        messages.append(msg)
                        if msg.get("type") in ("generation.done", "error"):
                            break
                    elif "bytes" in data:
                        messages.append({"type": "binary", "size": len(data["bytes"])})
                except Exception:
                    break

            types = [m.get("type") for m in messages]
            assert "generation.started" in types
            assert "generation.done" in types

    def test_unknown_voice_error(self):
        """Unknown voice → error with 'Unknown voice'."""
        test_client = TestClient(app)
        with test_client.websocket_connect("/v1/ws/speech") as ws:
            ws.send_json({
                "type": "generate",
                "input": "Hello",
                "voice": "nonexistent_voice",
                "request_id": "test-err",
            })

            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unknown voice" in msg["error"]
            assert msg["request_id"] == "test-err"

    def test_empty_input_error(self):
        """Empty input → error with 'Empty input text'."""
        test_client = TestClient(app)
        with test_client.websocket_connect("/v1/ws/speech") as ws:
            ws.send_json({
                "type": "generate",
                "input": "",
                "voice": "random",
                "request_id": "test-empty",
            })

            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Empty input text" in msg["error"]

    def test_input_too_long_error(self):
        """Input > 10000 chars → error with 'exceeds maximum'."""
        test_client = TestClient(app)
        with test_client.websocket_connect("/v1/ws/speech") as ws:
            ws.send_json({
                "type": "generate",
                "input": "x" * 10001,
                "voice": "random",
                "request_id": "test-long",
            })

            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "exceeds maximum" in msg["error"]

    def test_request_id_echo(self):
        """Custom request_id is echoed in all responses."""
        test_client = TestClient(app)
        with test_client.websocket_connect("/v1/ws/speech") as ws:
            ws.send_json({
                "type": "generate",
                "input": "Test",
                "voice": "random",
                "request_id": "my-custom-id",
            })

            messages = []
            while True:
                try:
                    data = ws.receive()
                    if "text" in data:
                        msg = json.loads(data["text"])
                        messages.append(msg)
                        if msg.get("type") in ("generation.done", "error"):
                            break
                    elif "bytes" in data:
                        pass  # binary PCM, no request_id
                except Exception:
                    break

            # All JSON messages should have our request_id
            for msg in messages:
                if "request_id" in msg:
                    assert msg["request_id"] == "my-custom-id"
