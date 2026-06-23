"""Configuration constants for the TTS server."""

# ── Tokenizer / special tokens (shared with kanitts2-vllm plugin) ───────────
from kanitts2_vllm.config import (  # noqa: E402
    AUDIO_TOKENS_START,
    END_OF_AI,
    END_OF_HUMAN,
    END_OF_SPEECH,
    END_OF_TEXT,
    PAD_TOKEN,
    SPEAKER_EMB_DIM,
    START_OF_AI,
    START_OF_HUMAN,
    START_OF_SPEECH,
    START_OF_TEXT,
    TOKENIZER_LENGTH,
)

# ── Audio codec ─────────────────────────────────────────────────────────────
CODEBOOK_SIZE = 4032
SAMPLE_RATE = 22050
SAMPLES_PER_FRAME = 1764          # 7×7×6×3×2 — exact codec hop length
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"

# ── Streaming ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 25            # new frames per decoder iteration
LOOKBACK_FRAMES = 15      # context frames for sliding window continuity

# ── Generation ──────────────────────────────────────────────────────────────
TEMPERATURE = 0.6
TOP_P = 0.95
REPETITION_PENALTY = 1.1
MAX_TOKENS = 3000

# ── Long-form ──────────────────────────────────────────────────────────────
LONG_FORM_THRESHOLD_SECONDS = 40.0
LONG_FORM_CHUNK_DURATION = 30.0
LONG_FORM_SILENCE_DURATION = 0.2

# ── Timeouts ──────────────────────────────────────────────────────────────
GENERATION_TIMEOUT = 300  # 5 minutes — max time for a single generation request

# ── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "nineninesix/kani-tts-2-pt"
