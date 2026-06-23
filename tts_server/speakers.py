"""Speaker embedding manager — loads .pt files from a directory."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


class SpeakerManager:
    """Load and serve speaker embeddings from .pt files."""

    def __init__(self, speakers_dir: str | Path) -> None:
        self.speakers_dir = Path(speakers_dir)
        self._embeddings: dict[str, torch.Tensor] = {}
        self._load()

    def _load(self) -> None:
        if not self.speakers_dir.exists():
            log.warning("Directory %s does not exist — no speakers loaded", self.speakers_dir)
            return

        for pt_file in sorted(self.speakers_dir.glob("*.pt")):
            name = pt_file.stem
            try:
                emb = torch.load(pt_file, weights_only=True)
            except Exception:
                # Speaker files are local, operator-provided assets (baked into
                # the image / placed in this dir) — not untrusted input. Some
                # embedding save formats trip the weights_only unpickler on
                # certain torch versions; fall back to a full load for these.
                log.warning("weights_only load failed for %s; retrying full load", pt_file.name)
                emb = torch.load(pt_file, weights_only=False)
            if not torch.is_tensor(emb):
                emb = torch.as_tensor(emb)
            emb = emb.float()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)  # [128] → [1, 128]
            self._embeddings[name] = emb

        log.info("Loaded %d speakers: %s", len(self._embeddings), list(self._embeddings.keys()))

    def get(self, name: str) -> torch.Tensor | None:
        """Return speaker embedding [1, 128] or None if not found."""
        return self._embeddings.get(name)

    def list_names(self) -> list[str]:
        return list(self._embeddings.keys())
