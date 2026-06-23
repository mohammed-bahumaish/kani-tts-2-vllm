"""Streaming audio writer with sliding-window decoder.

Runs a background thread that accumulates audio tokens, decodes them in
chunks with a lookback window for frame-boundary continuity, and pushes
decoded PCM chunks to ``audio_chunks``.  An optional ``on_chunk`` callback
is invoked for each decoded chunk (used by SSE streaming).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Callable

import numpy as np

log = logging.getLogger(__name__)

from .config import CHUNK_SIZE, LOOKBACK_FRAMES, SAMPLE_RATE, SAMPLES_PER_FRAME

if TYPE_CHECKING:
    from .codec import NemoAudioCodec


CROSSFADE_SAMPLES = 256  # ~11.6ms at 22050Hz


class StreamingAudioWriter:
    """Sliding-window token-to-audio decoder running in a background thread."""

    def __init__(
        self,
        codec: NemoAudioCodec,
        chunk_size: int = CHUNK_SIZE,
        lookback_frames: int = LOOKBACK_FRAMES,
        sample_rate: int = SAMPLE_RATE,
        on_chunk: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.codec = codec
        self.chunk_size = chunk_size
        self.lookback_frames = lookback_frames
        self.sample_rate = sample_rate
        self._on_chunk = on_chunk

        self.token_queue: queue.Queue[int | None] = queue.Queue()
        self.audio_chunks: list[np.ndarray] = []
        self.running = True

        # Internal state — prompt includes START_OF_SPEECH, so generation
        # begins inside the speech context already.
        self._inside_speech = True
        self._all_tokens: list[int] = []
        self._frames_decoded = 0

        # Uniform chunks — no progressive ramp. Ramp transitions caused buffer
        # underruns (health < 100%) at high concurrency. With CUDA graphs the
        # decode RTF is low enough that uniform chunk_size sustains 100% health.
        self._chunk_schedule: list[int] = []
        self._chunk_schedule_idx = 0

        # Crossfade state for smooth chunk boundaries
        self._prev_tail: np.ndarray | None = None
        t = np.linspace(0, 1, CROSSFADE_SAMPLES, dtype=np.float32)
        self._fade_in = 0.5 * (1 - np.cos(np.pi * t))
        self._fade_out = 1.0 - self._fade_in

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        self._decoder_thread = threading.Thread(
            target=self._decoder_worker, daemon=True
        )
        self._decoder_thread.start()

    def add_token(self, token_id: int) -> None:
        self.token_queue.put(token_id)

    def finalize(self) -> np.ndarray | None:
        """Stop decoder thread and return concatenated audio (or None)."""
        self.running = False
        self.token_queue.put(None)  # sentinel — wake up blocked get()
        self._decoder_thread.join(timeout=30)
        if self._decoder_thread.is_alive():
            log.warning("decoder thread did not exit within 30s")
        # Flush any remaining crossfade tail
        if self._prev_tail is not None:
            self._emit(self._prev_tail)
            self._prev_tail = None
        if self.audio_chunks:
            return np.concatenate(self.audio_chunks)
        return None

    # ── Internal ─────────────────────────────────────────────────────────

    def _emit(self, chunk: np.ndarray) -> None:
        """Append chunk to audio_chunks and invoke callback if set."""
        self.audio_chunks.append(chunk)
        if self._on_chunk is not None:
            self._on_chunk(chunk)

    def _emit_with_crossfade(
        self, chunk: np.ndarray, is_final: bool = False
    ) -> None:
        """Apply Hann crossfade at chunk boundaries, then emit."""
        cf = CROSSFADE_SAMPLES

        if len(chunk) < cf:
            # Too short to crossfade — flush prev_tail + chunk directly
            if self._prev_tail is not None:
                self._emit(np.concatenate([self._prev_tail, chunk]))
                self._prev_tail = None
            else:
                self._emit(chunk)
            return

        if self._prev_tail is not None:
            blended = (
                self._prev_tail * self._fade_out
                + chunk[:cf] * self._fade_in
            )
            if is_final:
                self._emit(np.concatenate([blended, chunk[cf:]]))
                self._prev_tail = None
            else:
                self._emit(np.concatenate([blended, chunk[cf:-cf]]))
                self._prev_tail = chunk[-cf:].copy()
        else:
            if is_final:
                self._emit(chunk)
                self._prev_tail = None
            else:
                self._emit(chunk[:-cf])
                self._prev_tail = chunk[-cf:].copy()

    # ── Background decoder ──────────────────────────────────────────────

    def _decoder_worker(self) -> None:
        speech_ended = False

        while True:
            token_id = self.token_queue.get()  # blocking, no timeout
            if token_id is None:
                break

            if token_id == self.codec.start_of_speech:
                self._inside_speech = True
                speech_ended = False
                continue

            if token_id == self.codec.end_of_speech:
                self._decode(is_final=True)
                self._inside_speech = False
                speech_ended = True
                continue

            if self._inside_speech and not speech_ended:
                self._all_tokens.append(token_id)
                total_frames = len(self._all_tokens) // 4
                new_frames = total_frames - self._frames_decoded

                # Progressive ramp: 1 → 5 → 10 → chunk_size
                if self._chunk_schedule_idx < len(self._chunk_schedule):
                    threshold = self._chunk_schedule[self._chunk_schedule_idx]
                else:
                    threshold = self.chunk_size
                if new_frames >= threshold:
                    self._decode(new_frames=threshold)
                    if self._chunk_schedule_idx < len(self._chunk_schedule):
                        self._chunk_schedule_idx += 1

    # ── Decode ───────────────────────────────────────────────────────────

    def _decode(self, new_frames: int | None = None, is_final: bool = False) -> None:
        total_frames = len(self._all_tokens) // 4
        if new_frames is None:
            new_frames = total_frames - self._frames_decoded
        if new_frames < 1:
            return

        start_frame = max(0, self._frames_decoded - self.lookback_frames)
        start_token = start_frame * 4
        tokens = self._all_tokens[start_token:]
        num_frames = len(tokens) // 4
        if num_frames == 0:
            return

        codes = np.array(tokens[: num_frames * 4]).reshape(-1, 4)
        audio = self.codec.decode_frames(codes)
        if audio is None:
            log.warning("decode_frames returned None (frames=%d, decoded_so_far=%d)", num_frames, self._frames_decoded)
            return

        lookback_skip = min(self._frames_decoded, self.lookback_frames)
        skip = lookback_skip * SAMPLES_PER_FRAME
        if is_final:
            self._emit_with_crossfade(audio[skip:], is_final=True)
        else:
            keep = new_frames * SAMPLES_PER_FRAME
            self._emit_with_crossfade(audio[skip : skip + keep])
        self._frames_decoded += new_frames
