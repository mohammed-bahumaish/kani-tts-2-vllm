"""Tests for GET /health and GET / endpoints."""

from __future__ import annotations

import pytest

from tts_server.server import app


@pytest.mark.asyncio
class TestHealth:
    async def test_healthy(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["engine_ready"] is True
        assert data["codec_ready"] is True
        assert data["speaker_projection_loaded"] is True
        assert isinstance(data["speakers"], list)
        assert "speaker_1" in data["speakers"]

    async def test_engine_none(self, client):
        original = app.state.engine
        app.state.engine = None
        try:
            resp = await client.get("/health")
            data = resp.json()
            assert data["engine_ready"] is False
        finally:
            app.state.engine = original

    async def test_no_proj_weight(self, client):
        original = app.state.speaker_proj_weight
        app.state.speaker_proj_weight = None
        try:
            resp = await client.get("/health")
            data = resp.json()
            assert data["speaker_projection_loaded"] is False
        finally:
            app.state.speaker_proj_weight = original


@pytest.mark.asyncio
class TestRoot:
    async def test_root(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "KaniTTS-2 Server"
        assert data["version"] == "1.0.0"
        assert "/v1/audio/speech" in data["endpoints"]
        assert "/v1/ws/speech" in data["endpoints"]
        assert "/health" in data["endpoints"]
