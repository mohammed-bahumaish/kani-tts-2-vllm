"""Create a speaker voice (.pt) from a reference audio clip.

Clones a voice from reference audio using the same embedder the KaniTTS-2 model
uses (`Orange/Speaker-wavLM-tbr`, 128-dim, 16 kHz) via the official `kani-tts-2`
package, so the resulting embedding is byte-for-byte what the model expects.

Requires the official package (only for this script, not the server):

    pip install kani-tts-2

Usage:

    python make_voice.py reference.wav alice
    # → writes tts_server/speakers/alice.pt  →  request with {"voice": "alice"}

Tip: 10–20 seconds of clean, single-speaker reference audio works best.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

DEFAULT_OUT = Path(__file__).parent / "tts_server" / "speakers"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="Path to a reference .wav (single speaker, ~10-20s)")
    parser.add_argument("name", help="Voice name — becomes the .pt filename and the API 'voice' value")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT,
        help=f"Directory to write <name>.pt into (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    try:
        from kani_tts import SpeakerEmbedder
    except ImportError as exc:
        raise SystemExit(
            "The official embedder is required: pip install kani-tts-2"
        ) from exc

    ref = Path(args.reference)
    if not ref.is_file():
        raise SystemExit(f"Reference audio not found: {ref}")

    embedder = SpeakerEmbedder()
    embedding = embedder.embed_audio_file(str(ref))  # [128] normalized

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.name}.pt"
    torch.save(embedding, out_path)
    print(f"Saved voice '{args.name}' → {out_path}  (shape {tuple(embedding.shape)})")
    print(f"Use it with:  curl ... -d '{{\"input\":\"Hello\",\"voice\":\"{args.name}\"}}'")


if __name__ == "__main__":
    main()
