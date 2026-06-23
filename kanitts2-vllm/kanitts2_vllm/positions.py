"""Frame-level position remapping for KaniTTS-2.

Audio tokens are grouped into frames of `tokens_per_frame` (default 4).
All tokens within the same frame share the same position ID so that RoPE
distance between frames is compressed.

Entry point:
  - remap_positions_prefill()  — used during prefill when input_ids is available
"""

import torch


def remap_positions_prefill(
    input_ids: torch.Tensor,
    audio_tokens_start: int,
    tokens_per_frame: int = 4,
    audio_step: float = 0.5,
) -> torch.Tensor:
    """Remap positions for prefill (flattened 1D tensor).

    Args:
        input_ids: [total_tokens] flattened across all sequences in the batch.
        audio_tokens_start: token ID threshold — ids >= this are audio tokens.
        tokens_per_frame: number of audio tokens per frame (default 4).
        audio_step: position increment per audio frame (default 0.5).

    Returns:
        positions: [total_tokens] long tensor with frame-level positions.
    """
    is_audio = input_ids >= audio_tokens_start
    text_mask = ~is_audio

    # Cumulative text tokens *before* each position (shift right by 1).
    text_before = torch.zeros_like(input_ids, dtype=torch.long)
    text_before[1:] = text_mask.long().cumsum(0)[:-1]

    # Cumulative audio tokens *before* each position.
    audio_before = torch.zeros_like(input_ids, dtype=torch.long)
    audio_before[1:] = is_audio.long().cumsum(0)[:-1]

    # Frame count = audio_before // tokens_per_frame
    frame_count = audio_before // tokens_per_frame

    # audio_step=0.5 compresses audio position space: 2 frames per position unit.
    # Floor to integer for RoPE (matches old server's .long() behavior).
    return (text_before + frame_count * audio_step).long()
