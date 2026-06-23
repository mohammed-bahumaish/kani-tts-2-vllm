"""FastAPI TTS server with SSE, PCM, and WebSocket streaming.

Endpoints:
  POST /v1/audio/speech  — OpenAI-compatible TTS (streaming SSE or full WAV/PCM)
  WS   /v1/ws/speech     — WebSocket TTS (PCM streaming, cancel support)
  GET  /health           — Readiness probe
"""

from __future__ import annotations

# Tell vLLM to load the kanitts2 plugin (registers KaniTTS2ForCausalLM in all processes).
import os
os.environ["VLLM_PLUGINS"] = "kanitts2"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from scipy.io.wavfile import write as wav_write

from .chunking import estimate_duration, split_into_sentences
from .codec import NemoAudioCodec
from .config import (
    AUDIO_TOKENS_START,
    CHUNK_SIZE,
    GENERATION_TIMEOUT,
    LONG_FORM_CHUNK_DURATION,
    LONG_FORM_SILENCE_DURATION,
    LONG_FORM_THRESHOLD_SECONDS,
    LOOKBACK_FRAMES,
    SAMPLE_RATE,
)
from .engine import TTSEngine
from .speakers import SpeakerManager
from .streaming import StreamingAudioWriter
from .utils import load_safetensor_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


SPEAKERS_DIR = Path(__file__).parent / "speakers"


# ── Request schema ──────────────────────────────────────────────────────────


class OpenAISpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=10000, description="Text to convert to speech")

    @field_validator("input")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("input must not be empty or whitespace-only")
        return v
    model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] = Field(
        default="tts-1", description="TTS model (ignored — always uses KaniTTS-2)"
    )
    voice: str = Field(
        default="speaker_1",
        description="Speaker name matching a .pt file in /speakers",
    )
    response_format: Literal["wav", "pcm"] = Field(
        default="wav", description="Audio format"
    )
    stream_format: Literal["sse", "audio"] | None = Field(
        default=None, description="'sse' for SSE events, 'audio' for raw chunked PCM bytes"
    )
    speed: float | None = Field(
        default=None, ge=0.25, le=4.0, description="Speed (accepted for compatibility, ignored)"
    )
    instructions: str | None = Field(
        default=None, description="Instructions (accepted for compatibility, ignored)"
    )
    enable_long_form: bool | None = Field(
        default=True, description="Auto-detect long-form and chunk"
    )
    max_chunk_duration: float | None = Field(
        default=None, description="Max seconds per chunk"
    )
    silence_duration: float | None = Field(
        default=None, description="Silence between chunks (seconds)"
    )
    chunk_size: int | None = Field(
        default=None, description="Frames per streaming decoder iteration (default: server CHUNK_SIZE)"
    )
    lookback_frames: int | None = Field(
        default=None, description="Lookback frames for sliding window continuity (default: server LOOKBACK_FRAMES)"
    )


# ── Lifecycle ───────────────────────────────────────────────────────────────


def _extract_speaker_proj_weight(model_name: str) -> torch.Tensor | None:
    """Load speaker_emb_projection.weight from safetensors checkpoint."""
    try:
        log.info("Loading speaker projection weight from safetensors...")
        weight = load_safetensor_weight(
            model_name, "model.speaker_emb_projection.weight"
        )
        log.info("Speaker projection weight loaded: %s", weight.shape)
        return weight
    except Exception:
        log.exception("Could not load speaker projection weight — using default voice")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    log.info("Starting up...")

    app.state.speaker_manager = SpeakerManager(SPEAKERS_DIR)
    app.state.codec = NemoAudioCodec()

    app.state.engine = TTSEngine()
    await app.state.engine.initialize()

    app.state.speaker_proj_weight = _extract_speaker_proj_weight(
        app.state.engine.model_name
    )

    # Warmup: exercise vLLM + codec CUDA kernels before accepting requests
    log.info("Warming up engine (first-request CUDA compilation)...")
    try:
        warmup_speaker = app.state.speaker_manager.get("speaker_1")
        warmup_tokens: list[int] = []
        async for token_id in app.state.engine.generate_stream(
            "warmup",
            speaker_emb=warmup_speaker,
            speaker_proj_weight=app.state.speaker_proj_weight,
        ):
            if token_id >= AUDIO_TOKENS_START:
                warmup_tokens.append(token_id)
                if len(warmup_tokens) >= 20:
                    break

        # Warm up codec HiFiGAN decoder
        if len(warmup_tokens) >= 4:
            codes = np.array(
                warmup_tokens[: len(warmup_tokens) // 4 * 4]
            ).reshape(-1, 4)
            app.state.codec.decode_frames(codes)
        log.info("Warmup complete (%d audio tokens)", len(warmup_tokens))
    except Exception:
        log.warning("Warmup failed (non-fatal)", exc_info=True)

    log.info("Ready")
    yield
    log.info("Shutting down...")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="KaniTTS-2 Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _get_text_chunks(request: OpenAISpeechRequest) -> list[str]:
    """Split request text into chunks for long-form, or return as single chunk."""
    estimated = estimate_duration(request.input)
    use_long = request.enable_long_form and estimated > LONG_FORM_THRESHOLD_SECONDS

    if use_long:
        return split_into_sentences(
            request.input,
            max_duration_seconds=request.max_chunk_duration or LONG_FORM_CHUNK_DURATION,
        )
    return [request.input]


async def _timeout_stream(aiter, timeout: float):
    """Wrap an async iterator with an overall deadline timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    ait = aiter.__aiter__()
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"generation exceeded {timeout}s timeout")
        try:
            item = await asyncio.wait_for(ait.__anext__(), timeout=remaining)
            yield item
        except StopAsyncIteration:
            break
        except TimeoutError:
            raise TimeoutError(f"generation exceeded {timeout}s timeout")


async def _generate_to_queue(
    chunk_q: asyncio.Queue,
    request: OpenAISpeechRequest,
    engine: TTSEngine,
    codec: NemoAudioCodec,
    speaker_emb: torch.Tensor | None,
    speaker_proj_weight: torch.Tensor | None,
    writer_ref: list[StreamingAudioWriter | None] | None = None,
) -> None:
    """Shared generation pipeline — produces ("chunk", audio) / ("done", stats) / ("error", msg)."""
    chunks = _get_text_chunks(request)
    total_audio_tokens = 0
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    def _safe_put(item):
        try:
            chunk_q.put_nowait(item)
        except asyncio.QueueFull:
            log.warning("chunk queue full, dropping audio chunk")

    def _on_chunk(audio: np.ndarray) -> None:
        loop.call_soon_threadsafe(_safe_put, ("chunk", audio))

    try:
        for i, text_chunk in enumerate(chunks):
            writer = StreamingAudioWriter(
                codec, chunk_size=request.chunk_size or CHUNK_SIZE,
                lookback_frames=request.lookback_frames or LOOKBACK_FRAMES, on_chunk=_on_chunk,
            )
            if writer_ref is not None:
                writer_ref[0] = writer
            writer.start()
            try:
                async for token_id in _timeout_stream(
                    engine.generate_stream(
                        text_chunk, speaker_emb=speaker_emb,
                        speaker_proj_weight=speaker_proj_weight,
                    ),
                    GENERATION_TIMEOUT,
                ):
                    writer.add_token(token_id)
                    if token_id >= AUDIO_TOKENS_START:
                        total_audio_tokens += 1
            finally:
                await asyncio.to_thread(writer.finalize)

            if i < len(chunks) - 1:
                silence_samples = int(
                    (request.silence_duration or LONG_FORM_SILENCE_DURATION) * SAMPLE_RATE
                )
                await chunk_q.put(("chunk", np.zeros(silence_samples, dtype=np.float32)))

        elapsed = time.monotonic() - t0
        frames = total_audio_tokens // 4
        audio_dur = frames / 12.5
        rtf = elapsed / audio_dur if audio_dur > 0 else 0
        log.info(
            "generation done: audio_tokens=%d frames=%d duration=%.1fs "
            "gen_time=%.2fs rtf=%.3f chunks=%d",
            total_audio_tokens, frames, audio_dur, elapsed, rtf, len(chunks),
        )
        stats = {
            "audio_tokens": total_audio_tokens,
            "audio_duration": round(audio_dur, 2),
            "generation_time": round(elapsed, 2),
        }
        chunk_q.put_nowait(("done", stats))
    except asyncio.CancelledError:
        chunk_q.put_nowait(("cancelled", None))
    except Exception as exc:
        log.exception("generation error")
        chunk_q.put_nowait(("error", str(exc)))


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check(request: Request):
    engine: TTSEngine | None = request.app.state.engine
    codec: NemoAudioCodec | None = request.app.state.codec
    speaker_manager: SpeakerManager | None = request.app.state.speaker_manager
    return {
        "status": "healthy",
        "engine_ready": engine is not None and engine.engine is not None,
        "codec_ready": codec is not None,
        "speaker_projection_loaded": request.app.state.speaker_proj_weight is not None,
        "speakers": speaker_manager.list_names() if speaker_manager else [],
    }


@app.post("/v1/audio/speech")
async def speech(body: OpenAISpeechRequest, request: Request):
    engine: TTSEngine = request.app.state.engine
    codec: NemoAudioCodec = request.app.state.codec
    speaker_manager: SpeakerManager = request.app.state.speaker_manager
    speaker_proj_weight: torch.Tensor | None = request.app.state.speaker_proj_weight

    if engine is None or codec is None:
        raise HTTPException(503, "Models not initialised")

    # Resolve speaker embedding.
    # When no speaker embeddings are installed, every request uses the default
    # token-ID path (model's built-in voice) — so the server works out of the
    # box. Validation only applies once speaker .pt files are present.
    speaker_emb = None
    if (
        body.voice != "random"
        and speaker_manager is not None
        and speaker_manager.list_names()
    ):
        speaker_emb = speaker_manager.get(body.voice)
        if speaker_emb is None:
            raise HTTPException(
                400,
                f"Unknown voice '{body.voice}'. "
                f"Available: {speaker_manager.list_names()}",
            )

    # ── SSE streaming mode ──────────────────────────────────────────────
    if body.stream_format == "sse":
        return _sse_response(body, engine, codec, speaker_emb, speaker_proj_weight)

    # ── Raw PCM streaming mode (LiveKit / OpenAI SDK compatible) ──────
    if body.stream_format == "audio":
        return _pcm_stream_response(body, engine, codec, speaker_emb, speaker_proj_weight)

    # ── Non-streaming mode ──────────────────────────────────────────────
    return await _full_response(body, engine, codec, speaker_emb, speaker_proj_weight)


# ── SSE streaming ───────────────────────────────────────────────────────────


def _sse_response(
    request: OpenAISpeechRequest,
    engine: TTSEngine,
    codec: NemoAudioCodec,
    speaker_emb: torch.Tensor | None,
    speaker_proj_weight: torch.Tensor | None,
) -> StreamingResponse:

    async def sse_generator():
        chunk_q: asyncio.Queue = asyncio.Queue(maxsize=100)
        gen_task = asyncio.create_task(
            _generate_to_queue(chunk_q, request, engine, codec, speaker_emb, speaker_proj_weight)
        )

        try:
            while True:
                try:
                    msg_type, data = await asyncio.wait_for(chunk_q.get(), timeout=60.0)
                except asyncio.TimeoutError:
                    log.warning("SSE generation stalled (60s timeout)")
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Generation stalled'})}\n\n"
                    break

                if msg_type == "chunk":
                    pcm = np.clip(data * 32767, -32768, 32767).astype(np.int16)
                    b64 = base64.b64encode(pcm.tobytes()).decode()
                    yield f"data: {json.dumps({'type': 'speech.audio.delta', 'audio': b64})}\n\n"

                elif msg_type == "done":
                    event = {"type": "speech.audio.done"}
                    if data:
                        event["usage"] = data
                    yield f"data: {json.dumps(event)}\n\n"
                    break

                elif msg_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'error': data})}\n\n"
                    break
        finally:
            await gen_task

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Raw PCM streaming ────────────────────────────────────────────────────────


def _pcm_stream_response(
    request: OpenAISpeechRequest,
    engine: TTSEngine,
    codec: NemoAudioCodec,
    speaker_emb: torch.Tensor | None,
    speaker_proj_weight: torch.Tensor | None,
) -> StreamingResponse:

    async def pcm_generator():
        chunk_q: asyncio.Queue = asyncio.Queue(maxsize=100)
        gen_task = asyncio.create_task(
            _generate_to_queue(chunk_q, request, engine, codec, speaker_emb, speaker_proj_weight)
        )

        try:
            while True:
                try:
                    msg_type, data = await asyncio.wait_for(chunk_q.get(), timeout=60.0)
                except asyncio.TimeoutError:
                    log.warning("PCM generation stalled (60s timeout)")
                    break
                if msg_type == "chunk":
                    pcm = np.clip(data * 32767, -32768, 32767).astype(np.int16)
                    yield pcm.tobytes()
                elif msg_type in ("done", "error"):
                    if msg_type == "error":
                        log.error("PCM stream error: %s", data)
                    break
        finally:
            await gen_task

    return StreamingResponse(
        pcm_generator(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        },
    )


# ── WebSocket streaming ──────────────────────────────────────────────────────


@app.websocket("/v1/ws/speech")
async def ws_speech(ws: WebSocket):
    await ws.accept()

    engine: TTSEngine = ws.app.state.engine
    codec: NemoAudioCodec = ws.app.state.codec
    speaker_manager: SpeakerManager = ws.app.state.speaker_manager
    speaker_proj_weight: torch.Tensor | None = ws.app.state.speaker_proj_weight

    if engine is None or codec is None:
        await ws.close(code=1011, reason="Models not initialised")
        return

    gen_task: asyncio.Task | None = None
    current_request_id: str | None = None
    writer_ref: list[StreamingAudioWriter | None] = [None]
    # Background cleanup tasks for previous generations (fire-and-forget)
    _cleanup_tasks: set[asyncio.Task] = set()

    def _cancel_current_sync() -> None:
        """Signal cancellation immediately (non-blocking).

        Stops the decoder thread and cancels the generation task, but does NOT
        await the task — cleanup is fire-and-forgotten to avoid blocking the
        new request on the previous generation's finalize/thread-join.
        """
        nonlocal gen_task, current_request_id
        if gen_task is not None and not gen_task.done():
            # Stop decoder thread immediately so finalize's thread.join is fast
            if writer_ref[0] is not None:
                writer_ref[0].running = False
            gen_task.cancel()

            # Fire-and-forget the await so we don't block the new request
            async def _cleanup(task: asyncio.Task) -> None:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            cleanup = asyncio.create_task(_cleanup(gen_task))
            _cleanup_tasks.add(cleanup)
            cleanup.add_done_callback(_cleanup_tasks.discard)
        gen_task = None
        writer_ref[0] = None
        current_request_id = None

    async def _cancel_current_await() -> None:
        """Full cancellation with await — used only on WS disconnect/error."""
        nonlocal gen_task, current_request_id
        if gen_task is not None and not gen_task.done():
            if writer_ref[0] is not None:
                writer_ref[0].running = False
            gen_task.cancel()
            try:
                await gen_task
            except (asyncio.CancelledError, Exception):
                pass
        gen_task = None
        writer_ref[0] = None
        current_request_id = None
        # Also await any pending cleanup tasks
        if _cleanup_tasks:
            await asyncio.gather(*_cleanup_tasks, return_exceptions=True)
            _cleanup_tasks.clear()

    async def _ws_consume(
        request_id: str,
        chunk_q: asyncio.Queue,
        produce_task: asyncio.Task,
    ) -> None:
        """Consume audio chunks from the generation queue and send over WS.

        This is the only task stored in gen_task.  The actual vLLM generation
        runs in produce_task (started BEFORE this task, so tokens are already
        flowing by the time we reach the first chunk_q.get()).
        """
        try:
            t_start = time.monotonic()
            t_first_byte = None

            while True:
                try:
                    msg_type, data = await asyncio.wait_for(chunk_q.get(), timeout=60.0)
                except asyncio.TimeoutError:
                    log.warning("WS chunk queue timeout (60s) for request %s", request_id)
                    break

                if msg_type == "chunk":
                    pcm = np.clip(data * 32767, -32768, 32767).astype(np.int16)
                    await ws.send_bytes(pcm.tobytes())
                    if t_first_byte is None:
                        t_first_byte = time.monotonic() - t_start
                elif msg_type == "done":
                    total_time = time.monotonic() - t_start
                    usage = dict(data) if data else {}
                    usage["total_time"] = round(total_time, 3)
                    if t_first_byte is not None:
                        usage["time_to_first_byte"] = round(t_first_byte, 3)
                    audio_dur = usage.get("audio_duration", 0)
                    if audio_dur > 0:
                        usage["realtime_factor"] = round(total_time / audio_dur, 3)
                    await ws.send_json({"type": "generation.done", "request_id": request_id, "usage": usage})
                    break
                elif msg_type == "cancelled":
                    break
                elif msg_type == "error":
                    await ws.send_json({"type": "error", "request_id": request_id, "error": data})
                    break
        finally:
            if not produce_task.done():
                produce_task.cancel()
                try:
                    await produce_task
                except (asyncio.CancelledError, Exception):
                    pass

    # ── Main receive loop ──
    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "generate":
                text = msg.get("input", "")
                if not isinstance(text, str):
                    text = ""
                voice = msg.get("voice", "speaker_1")
                request_id = msg.get("request_id") or str(uuid.uuid4())

                if len(text) > 10000:
                    await ws.send_json({
                        "type": "error", "request_id": request_id,
                        "error": "Input text exceeds maximum length (10000 characters)",
                    })
                    continue

                if not text:
                    await ws.send_json({"type": "error", "request_id": request_id, "error": "Empty input text"})
                    continue

                # ── Cancel previous generation (non-blocking) ──
                _cancel_current_sync()

                # ── Resolve speaker (sync — matches HTTP path) ──
                speaker_emb = None
                if (
                    voice != "random"
                    and speaker_manager is not None
                    and speaker_manager.list_names()
                ):
                    speaker_emb = speaker_manager.get(voice)
                    if speaker_emb is None:
                        await ws.send_json({
                            "type": "error", "request_id": request_id,
                            "error": f"Unknown voice '{voice}'. Available: {speaker_manager.list_names()}",
                        })
                        continue

                # ── Build request with all client-supplied fields ──
                ws_request = OpenAISpeechRequest(
                    input=text,
                    voice=voice,
                    chunk_size=msg.get("chunk_size"),
                    lookback_frames=msg.get("lookback_frames"),
                    enable_long_form=msg.get("enable_long_form", True),
                    max_chunk_duration=msg.get("max_chunk_duration"),
                    silence_duration=msg.get("silence_duration"),
                )

                # ── Start generation FIRST (single create_task, matching SSE path) ──
                chunk_q: asyncio.Queue = asyncio.Queue(maxsize=100)
                produce_task = asyncio.create_task(
                    _generate_to_queue(
                        chunk_q, ws_request, engine, codec,
                        speaker_emb, speaker_proj_weight, writer_ref=writer_ref,
                    )
                )

                # Send started notification (non-critical, can happen after create_task)
                await ws.send_json({"type": "generation.started", "request_id": request_id})

                # ── Start consumer task ──
                current_request_id = request_id
                gen_task = asyncio.create_task(
                    _ws_consume(request_id, chunk_q, produce_task)
                )

            elif msg_type == "cancel":
                cancel_id = msg.get("request_id")
                if cancel_id and cancel_id == current_request_id:
                    _cancel_current_sync()

    except WebSocketDisconnect:
        log.info("WS client disconnected")
    except Exception as e:
        log.warning("WS error: %s", e)
    finally:
        await _cancel_current_await()


# ── Non-streaming full response ─────────────────────────────────────────────


async def _full_response(
    request: OpenAISpeechRequest,
    engine: TTSEngine,
    codec: NemoAudioCodec,
    speaker_emb: torch.Tensor | None,
    speaker_proj_weight: torch.Tensor | None,
) -> Response:
    chunk_q: asyncio.Queue = asyncio.Queue()
    gen_task = asyncio.create_task(
        _generate_to_queue(chunk_q, request, engine, codec, speaker_emb, speaker_proj_weight)
    )

    # Drain by awaiting: audio chunks are enqueued from the decoder thread via
    # loop.call_soon_threadsafe, so we must yield to the loop (await get()) for
    # those callbacks to fire — get_nowait() would miss in-flight chunks.
    audio_parts: list[np.ndarray] = []
    try:
        while True:
            msg_type, data = await chunk_q.get()
            if msg_type == "chunk":
                audio_parts.append(data)
            elif msg_type == "done":
                break
            elif msg_type == "error":
                raise HTTPException(500, f"Generation error: {data}")
    finally:
        await gen_task

    if not audio_parts:
        raise HTTPException(500, "No audio generated")

    full_audio = np.concatenate(audio_parts)

    if request.response_format == "pcm":
        pcm_data = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)
        return Response(
            content=pcm_data.tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Channels": "1",
                "X-Bit-Depth": "16",
            },
        )

    # WAV — convert to 16-bit PCM for standard consumer compatibility
    pcm_int16 = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)
    wav_buf = io.BytesIO()
    wav_write(wav_buf, SAMPLE_RATE, pcm_int16)
    wav_buf.seek(0)
    return Response(content=wav_buf.read(), media_type="audio/wav")


@app.get("/")
async def root():
    return {
        "name": "KaniTTS-2 Server",
        "version": "1.0.0",
        "endpoints": {
            "/v1/audio/speech": "POST — TTS generation (streaming SSE or full WAV/PCM)",
            "/v1/ws/speech": "WS — WebSocket TTS (PCM streaming, cancel support)",
            "/health": "GET — Health check",
        },
    }


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
