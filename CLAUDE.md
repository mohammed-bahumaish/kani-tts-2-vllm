# tts-server/CLAUDE.md

KaniTTS-2 inference server: FastAPI + vLLM AsyncLLMEngine + NanoCodec decoder. Models live on HF and are baked into the Docker image; speaker embeddings live in `tts_server/speakers/`.

## Model + codec

- LM: `nineninesix/kani-tts-2-pt` (via vLLM AsyncLLMEngine).
- Codec: `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`, 4 codebooks, FSQ.

The vLLM plugin lives in `kanitts2-vllm/` and **registers `KaniTTS2ForCausalLM`** at import time.

## Build

```bash
# context: this directory (standalone repo). Models are public — HF token optional.
docker build -t kanitts2-server .
docker run --gpus all -p 8000:8000 kanitts2-server
```

Match the `FROM` base-image CUDA tag to the target GPU (12.x Ada/Hopper, 13.x Blackwell/5090).

## Layout

```
tts_server/
├── server.py          # FastAPI endpoints (SSE, PCM stream, WS, full WAV)
├── engine.py          # vLLM AsyncLLMEngine wrapper
├── codec.py           # NanoCodec token → waveform decoder
├── streaming.py       # StreamingAudioWriter — background thread, sliding-window decode, crossfade
├── chunking.py        # Long-form text splitting (sentence boundaries)
├── speakers.py        # SpeakerManager — loads .pt embeddings from speakers/
├── config.py          # constants (re-exports kanitts2_vllm.config)
├── utils.py           # safetensor weight loading
└── nanocodec/         # NanoCodec decoder

kanitts2_vllm/         # vLLM plugin (REGISTERS KaniTTS2ForCausalLM)
├── config.py          # special tokens, speaker embedding config
├── model.py           # KaniTTS2ForCausalLM with frame-level positions
├── positions.py       # Position remapping for audio frames
└── rope.py            # Learnable RoPE with per-layer frequencies
```

## API

| Endpoint | Method | Modes |
|---|---|---|
| `/v1/audio/speech` | POST | `stream_format`: `sse` / `audio` (raw PCM) / unset (full WAV) |
| `/v1/ws/speech` | WS | Binary PCM chunks + JSON control messages |
| `/health` | GET | `engine_ready`, `codec_ready`, `speakers` |

Per-request override params: `chunk_size`, `lookback_frames`.

## Generation constants

| Variable | Value | Notes |
|---|---|---|
| `SAMPLE_RATE` | 22050 | |
| `SAMPLES_PER_FRAME` | 1764 | codec frame size |
| `CHUNK_SIZE` | 25 | new frames per streaming decoder iteration |
| `LOOKBACK_FRAMES` | 15 | lookback for decode continuity |
| `LONG_FORM_THRESHOLD_SECONDS` | 40 | auto-chunk above this |
| `LONG_FORM_CHUNK_DURATION` | 30 | max sec per chunk |
| `LONG_FORM_SILENCE_DURATION` | 0.2 | silence between chunks |
| `TEMPERATURE` | 0.6 | |
| `REPETITION_PENALTY` | 1.1 | both paths (embeds path enabled by the penalty patch below) |
| `TOP_P` | 0.95 | |
| `MAX_TOKENS` | 3000 | |

Engine config (`engine.py`): `dtype=bfloat16`, `enforce_eager=True`, `enable_prefix_caching=False`, `max_num_seqs=12`. No fp8, no CUDA graphs — the decode-time frame-position remap isn't graph-safe yet (correctness first).

## Key patterns (READ before changing)

- **Two prompt paths**: token IDs (no speaker, `default_sampling_params`) vs `prompt_embeds` (speaker, `embeds_sampling_params`). 128-dim WavLM embedding → `speaker_emb_projection` → inserted at position 1.
- **Frame-level positions** (THE thing that makes audio coherent): all 4 audio tokens of a frame share one RoPE position. The plugin reads the prefill/decode split + per-request cache slots from `get_forward_context().attn_metadata` (a dict of short-conv metadata; mamba reorders **decode tokens first**, then prefill). Decode position = `_first_audio_pos[slot] + ((abs_pos − _first_audio_pos[slot]) // 4) * audio_step`. `_first_audio_pos[slot]` is recorded at prefill. See `model.py: forward()` + `_get_conv_metadata()`.
- **Sliding-window decode**: 15-frame lookback prevents chunk-boundary artifacts. **Crossfade**: Hann window (256 samples) at chunk boundaries.
- **Shared generation pipeline**: `_generate_to_queue()` is the single coroutine for all 4 modes.

## Critical bug workarounds (vLLM version sensitivity)

These broke when moving to a newer vLLM (the original worked on an older build). Both are handled in the plugin/server:

- **Frame positions via attn metadata, NOT a kwarg.** Older vLLM passed `mamba_cache_params` (with `state_indices_tensor`) into `forward()`; newer vLLM does not — read it from `get_forward_context().attn_metadata` instead. If `state_indices` is missing, the frame remap silently no-ops → audio tokens get sequential RoPE positions → **total gibberish**. Detect prefill vs decode from the metadata's `num_decodes`/`num_prefills` (the old `num_tokens > max_num_seqs` heuristic fails for short prompts).
- **Penalties + `prompt_embeds` (vLLM #28307).** vLLM's `apply_penalties` always scatters the request's *prompt* token IDs, which for `prompt_embeds` requests are `-1` placeholders → out-of-bounds CUDA assert that kills the EngineCore. The plugin's `register()` monkeypatches `get_token_bin_counts_and_mask` to clamp out-of-range IDs into the discarded pad bin, so `repetition_penalty` works on the speaker path (without it, some voices over-generate/ramble for tens of seconds).
