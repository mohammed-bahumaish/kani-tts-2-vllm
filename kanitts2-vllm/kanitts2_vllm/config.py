"""Constants for KaniTTS-2 — read from HF config at runtime with these defaults."""

# Token layout
TOKENIZER_LENGTH = 64400
AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10  # 64410

# Frame-level position encoding
TOKENS_PER_FRAME = 4
AUDIO_STEP = 0.5  # Position step per audio frame (0.5 compresses audio position space for v2)

# Speaker embedding
SPEAKER_EMB_DIM = 128

# Learnable RoPE
USE_LEARNABLE_ROPE = True
ALPHA_MIN = 0.1
ALPHA_MAX = 2.0

# Special tokens (for reference — used by the app server, not the model)
START_OF_TEXT = 1
END_OF_TEXT = 2
START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7
