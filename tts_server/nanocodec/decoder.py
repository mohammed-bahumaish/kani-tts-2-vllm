"""NanoCodec HiFiGAN decoder and FSQ dequantizer.

Ported from NVIDIA NeMo v2.4.0:
- nemo/collections/tts/modules/audio_codec_modules.py
  (CausalHiFiGANDecoder, ResidualBlock, HiFiGANResBlock, HiFiGANResLayer)
- nemo/collections/tts/modules/quantizers.py
  (FiniteScalarQuantizer, GroupFiniteScalarQuantizer)
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn

from .modules import (
    CausalConv1dNorm,
    CausalConvTranspose1dNorm,
    ClampActivation,
    HalfSnake,
    mask_sequence_tensor,
)


# ── Finite Scalar Quantizer (decode-only) ─────────────────────────────────────


class FiniteScalarQuantizer(nn.Module):
    """Single-group FSQ dequantizer. Decode-only — no encode path.

    Converts flat integer indices back to continuous codes by decomposing
    the index into per-dimension non-negative values, then centering them.

    Buffers (loaded from checkpoint):
        dim_base_index: (1, D, 1) — divisors for each dimension
        num_levels:     (1, D, 1) — modulo for each dimension
    """

    def __init__(self, num_levels_per_dim: list[int]) -> None:
        super().__init__()
        num_levels = torch.tensor(num_levels_per_dim, dtype=torch.int64)

        # Compute base indices for decomposing flat index: [1, L0, L0*L1, ...]
        dim_base_index = torch.cumprod(
            torch.cat([torch.ones(1, dtype=torch.int64), num_levels[:-1]]), dim=0
        )

        # Register as buffers with shapes (1, D, 1) for broadcasting
        self.register_buffer("dim_base_index", dim_base_index.unsqueeze(0).unsqueeze(-1))
        self.register_buffer("num_levels", num_levels.unsqueeze(0).unsqueeze(-1))

    def decode(self, indices: torch.Tensor, input_len: torch.Tensor | None = None) -> torch.Tensor:
        """Decode flat indices to continuous codes.

        Args:
            indices: (D, B, T) — one codebook dimension per group
            input_len: (B,) — valid lengths for masking
        Returns:
            (B, D, T) — dequantized continuous values in [-1, 1]
        """
        indices = einops.rearrange(indices, "D B T -> B D T")
        codes_nonneg = (indices // self.dim_base_index) % self.num_levels

        # Center: nonneg -> [-1, 1] range
        scale = self.num_levels // 2  # also used as offset
        dequantized = (codes_nonneg - scale).float() / scale.float()

        if input_len is not None:
            dequantized = mask_sequence_tensor(dequantized, input_len)
        return dequantized


class GroupFiniteScalarQuantizer(nn.Module):
    """Multi-group FSQ dequantizer. NanoCodec uses 4 groups with levels [9,8,8,7].

    Each group independently dequantizes its codebook, then results are
    concatenated along the feature dimension.
    """

    def __init__(
        self,
        num_groups: int = 4,
        num_levels_per_group: list[list[int]] | None = None,
    ) -> None:
        super().__init__()
        if num_levels_per_group is None:
            num_levels_per_group = [[9, 8, 8, 7]] * num_groups

        self.num_groups = num_groups
        self.fsqs = nn.ModuleList(
            [FiniteScalarQuantizer(levels) for levels in num_levels_per_group]
        )

    def decode(self, indices: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Decode grouped indices.

        Args:
            indices: (num_groups, B, T) — one row per codebook group
            input_len: (B,)
        Returns:
            (B, total_features, T) — concatenated dequantized features
        """
        groups = indices.chunk(self.num_groups, dim=0)
        dequantized = [
            fsq.decode(group, input_len) for group, fsq in zip(groups, self.fsqs)
        ]
        return torch.cat(dequantized, dim=1)


# ── HiFiGAN Decoder ───────────────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """Single residual block: act → dilated_conv → act → conv → add input."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.input_activation = HalfSnake(channels)
        self.input_conv = CausalConv1dNorm(channels, channels, kernel_size, dilation=dilation)
        self.skip_activation = HalfSnake(channels)
        self.skip_conv = CausalConv1dNorm(channels, channels, kernel_size, dilation=1)

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        hidden = self.input_activation(inputs)
        hidden = self.input_conv(hidden, input_len)
        hidden = self.skip_activation(hidden)
        hidden = self.skip_conv(hidden, input_len)
        return inputs + hidden


class HiFiGANResBlock(nn.Module):
    """Sequential chain of ResidualBlocks with different dilations."""

    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels, kernel_size, d) for d in dilations]
        )

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        out = inputs
        for block in self.res_blocks:
            out = block(out, input_len)
        return out


class HiFiGANResLayer(nn.Module):
    """Parallel multi-kernel ResBlocks, outputs averaged.

    Creates one HiFiGANResBlock per kernel_size, each with the full dilation list.
    All branches run in parallel and their outputs are averaged.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...] = (3, 7, 11),
        dilations: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [HiFiGANResBlock(channels, ks, dilations) for ks in kernel_sizes]
        )

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        residuals = [block(inputs, input_len) for block in self.res_blocks]
        return sum(residuals) / len(residuals)


class CausalHiFiGANDecoder(nn.Module):
    """Causal HiFiGAN decoder for NanoCodec.

    Architecture (from model_config.yaml):
        pre_conv(16 → 864) → 5× [HalfSnake → CausalConvTranspose1d → ResLayer]
        → HalfSnake → post_conv(27 → 1) → Clamp(-1, 1)

    Channel progression: 864 → 432 → 216 → 108 → 54 → 27
    Upsample rates: [7, 7, 6, 3, 2] → total 1764× (= samples_per_frame)
    """

    def __init__(
        self,
        input_dim: int = 16,
        up_sample_rates: tuple[int, ...] = (7, 7, 6, 3, 2),
        base_channels: int = 864,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()

        self.up_sample_rates = up_sample_rates
        channels = base_channels

        # Input projection
        self.pre_conv = CausalConv1dNorm(input_dim, channels, kernel_size=7)

        # Upsample blocks
        self.activations = nn.ModuleList()
        self.up_sample_conv_layers = nn.ModuleList()
        self.res_layers = nn.ModuleList()

        for rate in up_sample_rates:
            out_channels = channels // 2
            self.activations.append(HalfSnake(channels))
            self.up_sample_conv_layers.append(
                CausalConvTranspose1dNorm(
                    channels, out_channels, kernel_size=rate * 2, stride=rate
                )
            )
            self.res_layers.append(
                HiFiGANResLayer(out_channels, resblock_kernel_sizes, resblock_dilation_sizes)
            )
            channels = out_channels

        # Output projection
        self.post_activation = HalfSnake(channels)
        self.post_conv = CausalConv1dNorm(channels, 1, kernel_size=3)
        self.out_activation = ClampActivation(-1.0, 1.0)

    def forward(
        self, inputs: torch.Tensor, input_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent features to waveform.

        Args:
            inputs: (B, input_dim, T) — dequantized FSQ codes
            input_len: (B,) — valid frame lengths
        Returns:
            audio: (B, T_audio) — waveform samples
            audio_len: (B,) — valid audio lengths
        """
        audio_len = input_len
        out = self.pre_conv(inputs, audio_len)

        for act, up_conv, res_layer, rate in zip(
            self.activations, self.up_sample_conv_layers, self.res_layers, self.up_sample_rates
        ):
            audio_len = audio_len * rate
            out = act(out)
            out = up_conv(out, audio_len)
            out = res_layer(out, audio_len)

        out = self.post_activation(out)
        out = self.post_conv(out, audio_len)
        audio = self.out_activation(out)
        audio = einops.rearrange(audio, "B 1 T -> B T")
        return audio, audio_len
