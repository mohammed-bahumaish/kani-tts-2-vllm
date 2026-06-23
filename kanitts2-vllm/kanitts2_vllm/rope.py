"""Learnable RoPE for KaniTTS-2.

Each attention layer has a learnable alpha that scales the base RoPE frequencies:
    theta_i^(l) = alpha^(l) * base^(-2i/d)
where alpha is constrained to [alpha_min, alpha_max] via sigmoid.

The forward signature matches vLLM's RotaryEmbedding:
    forward(positions, query, key) -> (query_rotated, key_rotated)

All operations use tensor ops only (no Python if/else on tensor values)
for CUDA graph compatibility.
"""

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NeoX-style rotation: split last dim in half, swap and negate."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to a tensor (q or k).

    Args:
        t:   [total_tokens, num_heads, head_dim]
        cos: [total_tokens, head_dim]
        sin: [total_tokens, head_dim]
    Returns:
        rotated tensor of same shape as t.
    """
    # Broadcast cos/sin over the heads dimension.
    return t * cos.unsqueeze(1) + _rotate_half(t) * sin.unsqueeze(1)


class LearnableRotaryEmbedding(nn.Module):
    """Learnable RoPE with per-layer frequency scaling.

    Parameters loaded from checkpoint:
        alpha_weight  — unconstrained scalar, mapped to [alpha_min, alpha_max] via sigmoid.
        inv_freq_base — buffer, base inverse frequencies.
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 10000.0,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        inv_freq_base = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        self.register_buffer("inv_freq_base", inv_freq_base, persistent=False)

        # Learnable scalar — will be loaded from checkpoint.
        self.alpha_weight = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))

        # Lazily cached inv_freq — alpha_weight is frozen during inference,
        # so inv_freq_base * alpha is constant after the first forward call.
        self._cached_inv_freq: torch.Tensor | None = None

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply learnable rotary embedding.

        Args:
            positions: [total_tokens] — position index per token.
            query:     [total_tokens, num_heads, head_dim]
            key:       [total_tokens, num_kv_heads, head_dim]

        Returns:
            (query_rotated, key_rotated) with same shapes.
        """
        # Cache inv_freq — alpha_weight is frozen during inference.
        if self._cached_inv_freq is None:
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(
                self.alpha_weight
            )
            self._cached_inv_freq = (self.inv_freq_base * alpha).detach()
        inv_freq = self._cached_inv_freq  # [head_dim/2]

        # positions: [T] -> freqs: [T, head_dim/2]
        freqs = torch.outer(positions.float(), inv_freq)
        # Duplicate to full head_dim: [T, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(query.dtype)
        sin = emb.sin().to(query.dtype)

        return _apply_rotary_pos_emb(query, cos, sin), _apply_rotary_pos_emb(
            key, cos, sin
        )
