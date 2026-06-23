"""Shared fixtures for TTS server tests.

vllm is not installable on CPU, so we stub it before any tts_server import.
Everything else (torch CPU, transformers, einops, etc.) is real.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# ── vllm stub (must happen before any tts_server import) ────────────────────

_vllm_mod = types.ModuleType("vllm")
_vllm_mod.AsyncEngineArgs = MagicMock  # type: ignore[attr-defined]
_vllm_mod.AsyncLLMEngine = MagicMock  # type: ignore[attr-defined]
_vllm_mod.SamplingParams = MagicMock  # type: ignore[attr-defined]
sys.modules["vllm"] = _vllm_mod

# ── Add kanitts2-vllm to PYTHONPATH (pure constants, safe) ──────────────────

_kanitts2_path = str(Path(__file__).resolve().parent.parent / "kanitts2-vllm")
if _kanitts2_path not in sys.path:
    sys.path.insert(0, _kanitts2_path)

# ── Now safe to import tts_server ───────────────────────────────────────────

import numpy as np
import torch

from tts_server.server import app  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_codec():
    """Mock NemoAudioCodec — decode_frames returns silence of correct length."""
    codec = MagicMock()
    codec.start_of_speech = 64401
    codec.end_of_speech = 64402
    codec.audio_tokens_start = 64410
    codec.sample_rate = 22050
    codec.codebook_size = 4032

    def _decode_frames(codes):
        num_frames = len(codes)
        return np.zeros(num_frames * 1764, dtype=np.float32)

    codec.decode_frames = MagicMock(side_effect=_decode_frames)
    return codec


@pytest.fixture()
def mock_engine():
    """Mock TTSEngine — generate_stream yields 40 audio tokens (10 frames) + EOS."""
    engine = MagicMock()
    engine.engine = True  # truthy for health check
    engine.model_name = "test-model"

    async def _generate_stream(text, speaker_emb=None, speaker_proj_weight=None, sampling_params=None):
        # 40 audio tokens (10 frames) starting at AUDIO_TOKENS_START (64410),
        # then END_OF_SPEECH (64402) so the streaming writer flushes a final
        # decode regardless of CHUNK_SIZE (10 frames < the default chunk size).
        for _ in range(40):
            yield 64410
        yield 64402

    engine.generate_stream = _generate_stream
    return engine


@pytest.fixture()
def mock_speaker_manager():
    """Mock SpeakerManager with two speakers."""
    mgr = MagicMock()

    def _get(name):
        if name == "speaker_1":
            return torch.randn(1, 128)
        return None

    mgr.get = MagicMock(side_effect=_get)
    mgr.list_names = MagicMock(return_value=["speaker_1", "speaker_2"])
    return mgr


@pytest.fixture()
def mock_proj_weight():
    """Mock speaker projection weight [2048, 128]."""
    return torch.randn(2048, 128)


@pytest_asyncio.fixture()
async def client(mock_engine, mock_codec, mock_speaker_manager, mock_proj_weight):
    """Async httpx client with mocked app.state — does NOT trigger lifespan."""
    import httpx

    app.state.engine = mock_engine
    app.state.codec = mock_codec
    app.state.speaker_manager = mock_speaker_manager
    app.state.speaker_proj_weight = mock_proj_weight

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    # Clean up app.state
    app.state.engine = None
    app.state.codec = None
    app.state.speaker_manager = None
    app.state.speaker_proj_weight = None
