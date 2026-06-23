"""Load NanoCodec decoder weights from .nemo archive.

The .nemo file is a tar archive containing:
  - model_config.yaml
  - model_weights.ckpt (PyTorch state_dict)

We extract only `audio_decoder.*` (387 params) and `vector_quantizer.*`
(8 buffers) — the encoder and discriminator are not needed for decode.
"""

from __future__ import annotations

import io
import tarfile

import torch
from huggingface_hub import hf_hub_download

NEMO_FILENAME = "nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo"
WEIGHTS_ENTRY = "model_weights.ckpt"

DECODER_PREFIX = "audio_decoder."
VQ_PREFIX = "vector_quantizer."


def load_decoder_weights(
    model_name: str, device: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Extract decoder and VQ weights from a .nemo archive.

    Args:
        model_name: HuggingFace repo ID (e.g. "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")
        device: Target device for tensors

    Returns:
        (decoder_state_dict, vq_state_dict) with prefixes stripped.
    """
    nemo_path = hf_hub_download(
        repo_id=model_name,
        filename=NEMO_FILENAME,
        local_files_only=True,
    )

    # Extract model_weights.ckpt from the tar archive
    with tarfile.open(nemo_path, "r") as tar:
        # The weights file may be at root or nested (e.g. "./model_weights.ckpt")
        weights_member = None
        for member in tar.getmembers():
            if member.name.endswith(WEIGHTS_ENTRY):
                weights_member = member
                break
        if weights_member is None:
            raise FileNotFoundError(f"{WEIGHTS_ENTRY} not found in {nemo_path}")

        f = tar.extractfile(weights_member)
        if f is None:
            raise FileNotFoundError(f"Could not extract {weights_member.name}")

        buf = io.BytesIO(f.read())

    checkpoint = torch.load(buf, map_location=device, weights_only=True)

    # Filter and strip prefixes
    decoder_sd: dict[str, torch.Tensor] = {}
    vq_sd: dict[str, torch.Tensor] = {}

    for key, value in checkpoint.items():
        if key.startswith(DECODER_PREFIX):
            stripped = key[len(DECODER_PREFIX) :]
            # NeMo wraps activations in a CodecActivation container that stores
            # the actual activation as `.activation`. Our modules use the
            # activation directly, so strip the wrapper level.
            stripped = stripped.replace(".activation.snake_act.", ".snake_act.")
            decoder_sd[stripped] = value
        elif key.startswith(VQ_PREFIX):
            vq_sd[key[len(VQ_PREFIX) :]] = value

    if not decoder_sd:
        raise RuntimeError(f"No decoder weights found (prefix '{DECODER_PREFIX}')")
    if not vq_sd:
        raise RuntimeError(f"No VQ weights found (prefix '{VQ_PREFIX}')")

    return decoder_sd, vq_sd
