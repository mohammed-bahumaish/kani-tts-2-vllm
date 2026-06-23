"""Pure PyTorch building blocks for NanoCodec decoder.

Ported from NVIDIA NeMo v2.4.0:
- nemo/collections/tts/modules/audio_codec_modules.py
- nemo/collections/tts/parts/utils/helpers.py (mask_sequence_tensor)
"""

from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Masking ────────────────────────────────────────────────────────────────────


def mask_sequence_tensor(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Zero out elements beyond each sequence length.

    Args:
        tensor: (B, C, T) or (B, T)
        lengths: (B,) — valid length per sample
    """
    batch_size, *_, max_length = tensor.shape
    if len(tensor.shape) == 3:
        mask = torch.ones(
            batch_size, 1, max_length, dtype=lengths.dtype, device=lengths.device
        ).cumsum(dim=-1)
        mask = mask <= einops.rearrange(lengths, "B -> B 1 1")
    else:
        mask = torch.ones(
            batch_size, max_length, dtype=lengths.dtype, device=lengths.device
        ).cumsum(dim=-1)
        mask = mask <= einops.rearrange(lengths, "B -> B 1")
    return tensor * mask


# ── Activations ────────────────────────────────────────────────────────────────


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Snake activation: x + sin²(αx) / α."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + eps).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake(nn.Module):
    """Snake activation with learnable per-channel α."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


class HalfSnake(nn.Module):
    """Snake on first half of channels, LeakyReLU on second half."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.snake_channels = channels // 2
        self.snake_act = Snake(self.snake_channels)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.snake_act(x[:, : self.snake_channels, :]),
                self.lrelu(x[:, self.snake_channels :, :]),
            ],
            dim=1,
        )


class ClampActivation(nn.Module):
    """Clamp output to [min_value, max_value]."""

    def __init__(self, min_value: float = -1.0, max_value: float = 1.0) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min_value, self.max_value)


# ── Causal Convolutions ───────────────────────────────────────────────────────


def _get_extra_padding_for_conv1d(
    x: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding_total: int = 0,
) -> int:
    """Compute extra padding needed to make output length = ceil(input_length / stride)."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total).float() / stride + 1
    ideal_length = (torch.ceil(n_frames).long() - 1) * stride + kernel_size - padding_total
    return (ideal_length - length).item()


def _pad1d(x: torch.Tensor, paddings: tuple[int, int], mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    """Pad 1D tensor, handling negative padding (trim) on either side."""
    pad_l, pad_r = paddings
    # Handle negative padding (trimming)
    if pad_l < 0 or pad_r < 0:
        trim_l = max(-pad_l, 0)
        trim_r = max(-pad_r, 0)
        x = x[..., trim_l : x.shape[-1] - trim_r if trim_r > 0 else x.shape[-1]]
        pad_l = max(pad_l, 0)
        pad_r = max(pad_r, 0)
    if pad_l > 0 or pad_r > 0:
        return F.pad(x, (pad_l, pad_r), mode=mode, value=value)
    return x


class CausalConv1dNorm(nn.Module):
    """Causal Conv1d with weight normalization.

    Left-pads the input so the convolution is strictly causal (no future leakage).
    Uses the Encodec-style extra-padding logic for exact length divisibility.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "zeros",
        extra_pad_mode: str = "constant",
    ) -> None:
        super().__init__()
        self.extra_pad_mode = extra_pad_mode

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_mode,
        )

        kernel_size_eff = torch.tensor((kernel_size - 1) * dilation + 1, dtype=torch.int64)
        # Non-persistent: computed from constructor args, not saved in checkpoint
        self.register_buffer("stride", torch.tensor(stride, dtype=torch.int64), persistent=False)
        self.register_buffer("kernel_size", kernel_size_eff, persistent=False)
        self.register_buffer("padding_total", kernel_size_eff - torch.tensor(stride, dtype=torch.int64), persistent=False)

        self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        extra_padding = _get_extra_padding_for_conv1d(
            inputs, self.kernel_size, self.stride, self.padding_total.item()
        )
        hidden = _pad1d(inputs, (self.padding_total.item(), extra_padding), mode=self.extra_pad_mode)
        hidden = self.conv(hidden)
        hidden = mask_sequence_tensor(hidden, input_len)
        return hidden


class CausalConvTranspose1dNorm(nn.Module):
    """Causal transposed Conv1d with weight normalization.

    Trims all right-side padding for causal behavior (trim_right_ratio=1.0).
    Default groups=out_channels (depthwise) matching NanoCodec config.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if groups is None:
            groups = out_channels

        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias
        )

        padding_total = kernel_size - stride
        # trim_right_ratio = 1.0 → all padding trimmed from right
        self.padding_right = math.ceil(padding_total * 1.0)
        self.padding_left = padding_total - self.padding_right

        self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(inputs)
        end = hidden.shape[-1] - self.padding_right if self.padding_right > 0 else hidden.shape[-1]
        hidden = hidden[..., self.padding_left : end]
        hidden = mask_sequence_tensor(hidden, input_len)
        return hidden
