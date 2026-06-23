"""Long-form text splitting for TTS generation."""

from __future__ import annotations

import re


def split_into_sentences(
    text: str, max_duration_seconds: float = 12.0
) -> list[str]:
    """Split text into chunks suitable for TTS generation.

    Uses ~15 chars/second heuristic. Splits on sentence boundaries first,
    falls back to word-level splitting for very long sentences.
    """
    max_chars = int(max_duration_seconds * 15)

    # Split on sentence-ending punctuation, keeping punctuation with the sentence.
    parts = re.split(r"([.!?]+[\s\n]+|[.!?]+$)", text)

    sentences: list[str] = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        if i + 1 < len(parts):
            s += parts[i + 1]
        s = s.strip()
        if s:
            sentences.append(s)
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    # Group sentences into chunks
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            # Word-level splitting
            words = sentence.split()
            word_chunk = ""
            for word in words:
                if len(word_chunk) + len(word) + 1 <= max_chars:
                    word_chunk += word + " "
                else:
                    chunks.append(word_chunk.strip())
                    word_chunk = word + " "
            if word_chunk.strip():
                current = word_chunk.strip()
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


def estimate_duration(text: str, chars_per_second: float = 15.0) -> float:
    """Estimate speech duration in seconds."""
    return len(text) / chars_per_second
