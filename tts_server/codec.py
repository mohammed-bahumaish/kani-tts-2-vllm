"""NanoCodec audio decoder (standalone — no NeMo dependency).

Decodes audio token frames into waveforms using extracted NanoCodec weights.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from .config import (
    AUDIO_TOKENS_START,
    CODEBOOK_SIZE,
    CODEC_MODEL_NAME,
    END_OF_SPEECH,
    SAMPLE_RATE,
    START_OF_SPEECH,
)
from .nanocodec import (
    CausalHiFiGANDecoder,
    GroupFiniteScalarQuantizer,
    load_decoder_weights,
)

log = logging.getLogger(__name__)


class NemoAudioCodec:
    """Decode audio token frames into waveforms using NanoCodec.

    Public interface (unchanged from NeMo-based version):
        - decode_frames(audio_codes) -> np.ndarray | None
        - codebook_size, audio_tokens_start, start_of_speech, end_of_speech
        - sample_rate, device
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.codebook_size = CODEBOOK_SIZE
        self.audio_tokens_start = AUDIO_TOKENS_START
        self.start_of_speech = START_OF_SPEECH
        self.end_of_speech = END_OF_SPEECH
        self.sample_rate = SAMPLE_RATE

        # Pre-compute combined offset on device: codebook_offsets + audio_tokens_start
        # This replaces two separate subtractions with a single one.
        codebook_offsets = torch.tensor(
            [CODEBOOK_SIZE * i for i in range(4)], device=self.device
        )
        self._combined_offset = codebook_offsets + self.audio_tokens_start

        # Load standalone decoder
        self._vq, self._decoder = self._load(device)
        log.info("NanoCodec decoder loaded (standalone, no NeMo)")

    @staticmethod
    def _load(
        device: str,
    ) -> tuple[GroupFiniteScalarQuantizer, CausalHiFiGANDecoder]:
        """Load VQ dequantizer and HiFiGAN decoder with pretrained weights."""
        # NanoCodec config: 4 groups, each with FSQ levels [9, 8, 8, 7]
        vq = GroupFiniteScalarQuantizer(
            num_groups=4,
            num_levels_per_group=[[9, 8, 8, 7]] * 4,
        )
        decoder = CausalHiFiGANDecoder(
            input_dim=16,
            up_sample_rates=(7, 7, 6, 3, 2),
            base_channels=864,
        )

        decoder_sd, vq_sd = load_decoder_weights(CODEC_MODEL_NAME, device)
        vq.load_state_dict(vq_sd)
        decoder.load_state_dict(decoder_sd)

        # Remove weight_norm parametrizations — bakes computed weights
        # into plain tensors. Numerically identical but eliminates ~97
        # per-access recomputations in HiFiGAN's Conv1d/ConvTranspose1d.
        for m in decoder.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                if parametrize.is_parametrized(m, "weight"):
                    parametrize.remove_parametrizations(m, "weight")

        vq = vq.eval().to(device)
        decoder = decoder.eval().to(device)

        return vq, decoder

    def decode_frames(self, audio_codes: np.ndarray) -> np.ndarray | None:
        """Decode a chunk of audio frames.

        Args:
            audio_codes: shape [num_frames, 4] — raw token IDs (NOT de-offset).

        Returns:
            Float32 waveform or None if tokens are invalid.
        """
        if len(audio_codes) == 0:
            return None

        try:
            # Single subtraction with pre-computed combined offset
            codes = torch.tensor(audio_codes, device=self.device) - self._combined_offset

            if (codes < 0).any().item():
                return None

            # Shape: (4, 1, num_frames) — [num_groups, batch, time]
            codes = codes.T.unsqueeze(1)
            lengths = torch.tensor([codes.shape[-1]], device=self.device)

            with torch.inference_mode():
                # Dequantize: (4, 1, T) → (1, 16, T)
                dequantized = self._vq.decode(codes, lengths)
                # Decode: (1, 16, T) → (1, T_audio)
                audio, _ = self._decoder(dequantized, lengths)
                return audio.cpu().numpy().squeeze()
        except Exception:
            log.exception("decode_frames failed (%d frames)", len(audio_codes))
            return None
