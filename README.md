# KaniTTS-2 vLLM Server

A fast, OpenAI-compatible text-to-speech server for [KaniTTS-2](https://huggingface.co/nineninesix/kani-tts-2-pt),
built on [vLLM](https://docs.vllm.ai/) with [NVIDIA NanoCodec](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps)
for audio decoding.

- **OpenAI-compatible** `POST /v1/audio/speech` — drop-in for the `audio/speech` API
- **Low-latency streaming** — SSE, raw chunked PCM, and a WebSocket endpoint with cancellation
- **Sliding-window decoding** — frame-level lookback + Hann crossfade for artifact-free streaming
- **Custom voices** — 128-dim speaker embeddings injected via vLLM `prompt_embeds`
- **Long-form** — automatic sentence-boundary chunking for long inputs
- **Real vLLM** — continuous batching, paged KV cache, async streaming — not a `generate()` wrapper

KaniTTS-2 is a modified [LFM2](https://www.liquid.ai/) hybrid (attention + conv) model. A small
out-of-tree vLLM plugin ([`kanitts2-vllm/`](kanitts2-vllm/)) registers the custom model with
frame-level RoPE positions, learnable per-layer RoPE, and speaker-embedding projection.

---

## Quick start (Docker)

This is the recommended path. There is **no prebuilt image** — you build your own. Both models
(LM + codec) are public on the HF Hub and get baked into the image at build time, so the running
container needs no network and no tokens.

```bash
# 1. Build (context is this directory). Pick a base-image CUDA tag that matches your GPU —
#    see "Choosing a base image" below.
docker build -t kanitts2-server .

# 2. Run
docker run --gpus all -p 8000:8000 kanitts2-server

# 3. Generate (after the logs print "Ready")
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello from KaniTTS two."}' --output hello.wav
```

`GET /health` reports readiness:

```json
{ "status": "healthy", "engine_ready": true, "codec_ready": true, "speakers": ["speaker_1", "david_en", ...] }
```

### Choosing a base image

The Dockerfile builds on `vastai/vllm`, which ships vLLM + PyTorch + CUDA prebuilt. Any image with
**vLLM ≥ 0.16.0** works — match the CUDA version to your GPU:

| GPU family | Example base tag |
|---|---|
| Ada / Hopper (RTX 4090, L40, H100) | `vastai/vllm:v0.16.0-cuda-12.x` |
| Blackwell (RTX 5090) | `vastai/vllm:v0.16.0-cuda-13.0` (default) |

Edit the `FROM` line in the [`Dockerfile`](Dockerfile) to switch. To bake models from a gated/private
HF mirror, pass `--build-arg HUGGINGFACE_TOKEN=hf_...`.

---

## Local install (no Docker)

Requires a CUDA GPU and a working vLLM (≥ 0.16.0).

```bash
./install.sh
# or manually:
pip install ./kanitts2-vllm     # the model plugin
pip install -e .                # the server + deps (FastAPI, codec, vLLM)

python -m tts_server.server      # serves on :8000
```

---

## API

### `POST /v1/audio/speech`

| Field | Type | Default | Notes |
|---|---|---|---|
| `input` | string | — | Text to synthesize (≤ 10000 chars) |
| `voice` | string | `speaker_1` | A speaker name (see [Voices](#voices)) or `random` for the default voice |
| `response_format` | `wav` \| `pcm` | `wav` | Non-streaming output format |
| `stream_format` | `sse` \| `audio` \| unset | unset | `sse` = SSE events; `audio` = raw chunked PCM; unset = full WAV/PCM |
| `enable_long_form` | bool | `true` | Auto-chunk inputs longer than ~40 s |
| `max_chunk_duration` | float | `30.0` | Max seconds per long-form chunk |
| `silence_duration` | float | `0.2` | Silence inserted between chunks (s) |
| `chunk_size` | int | `25` | Frames per streaming decode iteration (per-request override) |
| `lookback_frames` | int | `15` | Sliding-window lookback frames (per-request override) |

`model`, `speed`, and `instructions` are accepted for OpenAI compatibility and ignored.

```bash
# Full WAV
curl -X POST localhost:8000/v1/audio/speech \
  -d '{"input":"The quick brown fox."}' -o out.wav

# SSE streaming (base64 PCM int16 deltas)
curl -N -X POST localhost:8000/v1/audio/speech \
  -d '{"input":"Streaming please.","stream_format":"sse"}'

# Raw chunked PCM (LiveKit / OpenAI SDK style); 22050 Hz mono s16le
curl -N -X POST localhost:8000/v1/audio/speech \
  -d '{"input":"Raw bytes.","stream_format":"audio"}' -o out.pcm
```

SSE event shape:

```
data: {"type": "speech.audio.delta", "audio": "<base64 PCM int16>"}
data: {"type": "speech.audio.done", "usage": {...}}
```

### `WS /v1/ws/speech`

Bidirectional streaming with mid-stream cancellation. Send JSON control messages, receive binary
PCM frames (22050 Hz, mono, s16le) plus JSON status events.

```jsonc
// client → server
{ "type": "generate", "input": "Hello", "voice": "speaker_1", "request_id": "abc" }
{ "type": "cancel", "request_id": "abc" }

// server → client
{ "type": "generation.started", "request_id": "abc" }
<binary PCM frames…>
{ "type": "generation.done", "request_id": "abc", "usage": { "time_to_first_byte": 0.21, ... } }
```

Sending a new `generate` automatically cancels the in-flight one on that socket.

### `GET /health` · `GET /`

Readiness probe and service info.

---

## Voices

Named voices are `*.pt` speaker embeddings in [`tts_server/speakers/`](tts_server/speakers/) — each
file's stem is the `voice` value. The repo ships the KaniTTS-2 voice set:

| Voice (`voice` value) | Name | Lang | | Voice (`voice` value) | Name | Lang |
|---|---|---|---|---|---|---|
| `speaker_1` | Kore | en | | `arjun_en` | Arjun | en |
| `speaker_2` | Puck | en | | `aisulu_ky` | Aisulu | ky |
| `speaker_3` | Andrew | en | | `adilet_ky` | Adilet | ky |
| `david_en` | David | en | | `bermet_ky` | Bermet | ky |
| `robert_en` | Robert | en | | `bakyt_ky` | Bakyt | ky |
| `linda_en` | Linda | en | | `speaker_6` | Ash | es |
| `john_en` | John | en | | `speaker_7` | Nova | es |
| `joseph_en` | Joseph | en | | `mila_en` | Mila | en |

(`speaker_map.json` in that folder holds these display names.) The default `voice` is `speaker_1`
(Kore); pass `"voice": "random"` for the model's built-in default voice with no embedding.

### Add your own voice

Clone any voice from a reference clip with the bundled helper — it uses the same 128-dim WavLM
embedder ([`Orange/Speaker-wavLM-tbr`](https://huggingface.co/Orange/Speaker-wavLM-tbr), 16 kHz) the
model expects:

```bash
pip install kani-tts-2                 # official package (only needed to make voices)
python make_voice.py reference.wav alice
# → tts_server/speakers/alice.pt  →  request with {"voice": "alice"}
```

Use ~10–20 s of clean, single-speaker reference audio. See
[`tts_server/speakers/README.md`](tts_server/speakers/README.md) for the raw embedding format.

---

## How it works

```
                    FastAPI (server.py)
POST /v1/audio/speech ─┐
WS   /v1/ws/speech ────┤
                       ▼
        ┌──────────────────────────────┐     ┌───────────────────────────┐
        │ TTSEngine (engine.py)        │     │ StreamingAudioWriter      │
        │  build prompt (+speaker emb) │     │  (streaming.py)           │
        │  vLLM AsyncLLMEngine ────────┼──▶  │  bg thread, sliding window│
        │  KaniTTS2ForCausalLM plugin  │ tok │  + Hann crossfade         │
        └──────────────────────────────┘ ids │            │              │
                                              │   NanoCodec (codec.py)    │
                                              │   tokens → 22.05kHz PCM   │
                                              └────────────┬──────────────┘
                                                     SSE / PCM / WS / WAV
```

- **Voices** — a named voice sends its 128-dim embedding through vLLM's `prompt_embeds`; `random`
  uses the model's built-in voice.
- **Low-latency streaming** — audio is decoded frame-by-frame on a background thread with a short
  lookback window and crossfaded chunk boundaries, so SSE/PCM/WS start playing within a few hundred ms.

### Tunable defaults ([`tts_server/config.py`](tts_server/config.py))

| Constant | Value | | Constant | Value |
|---|---|---|---|---|
| `SAMPLE_RATE` | 22050 | | `TEMPERATURE` | 0.6 |
| `CHUNK_SIZE` | 25 | | `TOP_P` | 0.95 |
| `LOOKBACK_FRAMES` | 15 | | `REPETITION_PENALTY` | 1.1 |
| `LONG_FORM_THRESHOLD_SECONDS` | 40 | | `MAX_TOKENS` | 3000 |

Engine settings (`engine.py`): `max_num_seqs=12`, `dtype=bfloat16`, `enforce_eager=True`. A single
RTX 5090 serves this comfortably.

---

## Benchmarks

RTX 5090 (32 GB), `max_num_seqs=12`, SSE streaming, short-sentence input, 5 trials/level — TTFB
(time to first audio byte):

| Concurrency | Avg TTFB | P50 | P95 |
|---|---|---|---|
| 1 | 250 ms | 251 ms | 253 ms |
| 2 | 295 ms | 294 ms | 301 ms |
| 4 | 369 ms | 370 ms | 379 ms |
| 6 | 454 ms | 458 ms | 468 ms |
| 8 | 544 ms | 545 ms | 555 ms |
| 10 | 621 ms | 627 ms | 638 ms |
| 12 | 728 ms | 734 ms | 744 ms |

Clean linear scaling to **12 concurrent requests at ~728 ms TTFB**; beyond that vLLM batching hits a
compute wall and latency climbs sharply, so `max_num_seqs=12` is the practical ceiling on a 5090.
Lower-VRAM GPUs ceiling earlier (RTX 4090 ≈ 9). Reproduce with:

```bash
python stress_test.py --host localhost --port 8000 --sse --levels 1,2,4,8,12 --requests 5
```

---

## Tests

```bash
pip install pytest pytest-asyncio httpx
pytest
```

Tests stub vLLM (it isn't CPU-installable) and exercise the FastAPI app, chunking, streaming, and the
SSE/WS contracts on CPU.

---

## Project layout

```
tts_server/            # FastAPI app server
├── server.py          # endpoints: SSE, PCM stream, WebSocket, full WAV
├── engine.py          # vLLM AsyncLLMEngine wrapper + prompt construction
├── codec.py           # NanoCodec token → waveform decoder
├── streaming.py       # background sliding-window decode + crossfade
├── chunking.py        # long-form sentence splitting
├── speakers.py        # loads .pt speaker embeddings
├── config.py          # constants
├── nanocodec/         # NanoCodec decoder
└── speakers/          # drop your *.pt voices here

kanitts2-vllm/         # out-of-tree vLLM plugin (registers KaniTTS2ForCausalLM)
Dockerfile             # build-your-own image, models baked in
make_voice.py          # clone a voice .pt from reference audio (official embedder)
stress_test.py         # latency benchmark across concurrency levels
```

---

## License & credits

- Model: [nineninesix/kani-tts-2-pt](https://huggingface.co/nineninesix/kani-tts-2-pt)
- Codec: [nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps)
- Inference: [vLLM](https://docs.vllm.ai/) ([plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/), [prompt embeds](https://docs.vllm.ai/en/latest/features/prompt_embeds/))
