"""Shared utilities for the TTS server."""

from __future__ import annotations

import json
import logging

import torch

log = logging.getLogger(__name__)


def load_safetensor_weight(model_name: str, key: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Load a single weight tensor from a HuggingFace safetensors checkpoint.

    Tries single-file ``model.safetensors`` first, then falls back to the
    sharded index to find which shard contains *key*.

    Args:
        model_name: HuggingFace model repo (e.g. ``nineninesix/kani-tts-2-pt``).
        key: Weight key (e.g. ``model.embed_tokens.weight``).
        dtype: Target dtype for the returned tensor.

    Returns:
        The weight tensor cast to *dtype*.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    try:
        path = hf_hub_download(model_name, "model.safetensors")
    except Exception:
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shard = index["weight_map"][key]
        path = hf_hub_download(model_name, shard)

    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(dtype=dtype)
