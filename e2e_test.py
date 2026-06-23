"""End-to-end correctness test for the KaniTTS-2 server.

Probes a *running* server over HTTP/WS and checks every output mode plus the
properties unit tests can't: non-silent audio, voice distinctness, no engine
death on a 2nd speaker request, long-form, error paths — and (with --asr) the
gold-standard intelligibility check: round-trip ASR word-error-rate.

    python e2e_test.py --url http://localhost:8000
    python e2e_test.py --url http://localhost:8000 --asr        # + Whisper WER
    python e2e_test.py --url http://localhost:8000 --concurrency 12

Deps: httpx, websockets, numpy, scipy   (+ faster-whisper for --asr)
Exit code 0 = all passed, 1 = something failed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
import time

import numpy as np
import httpx
from scipy.io.wavfile import read as wav_read

SR = 22050
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((x.astype(np.float64) / 32768.0) ** 2)))


def pcm16(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.int16)


# ── HTTP modes ───────────────────────────────────────────────────────────────

def test_health(base: str) -> list[str]:
    r = httpx.get(f"{base}/health", timeout=30)
    j = r.json()
    check(r.status_code == 200 and j.get("status") == "healthy", "health 200/healthy", str(j.get("status")))
    check(bool(j.get("engine_ready")), "engine_ready")
    check(bool(j.get("codec_ready")), "codec_ready")
    speakers = j.get("speakers", [])
    check(len(speakers) > 0, "speakers loaded", f"{len(speakers)} voices")
    return speakers


def test_full_wav(base: str, voice: str) -> np.ndarray:
    r = httpx.post(f"{base}/v1/audio/speech",
                   json={"input": "The quick brown fox jumps over the lazy dog.", "voice": voice},
                   timeout=120)
    ok = check(r.status_code == 200, "full WAV: 200", f"HTTP {r.status_code}")
    if not ok:
        return np.array([], dtype=np.int16)
    check(r.headers.get("content-type") == "audio/wav", "full WAV: content-type", r.headers.get("content-type", ""))
    sr, audio = wav_read(io.BytesIO(r.content))
    check(sr == SR, "full WAV: sample rate", f"{sr} Hz")
    dur = len(audio) / sr
    check(dur > 0.5, "full WAV: duration > 0.5s", f"{dur:.2f}s")
    check(rms(audio) > 0.005, "full WAV: non-silent", f"rms={rms(audio):.4f}")
    return audio


def test_full_pcm(base: str, voice: str) -> None:
    r = httpx.post(f"{base}/v1/audio/speech",
                   json={"input": "Testing raw PCM output.", "voice": voice, "response_format": "pcm"},
                   timeout=120)
    check(r.status_code == 200, "full PCM: 200", f"HTTP {r.status_code}")
    a = pcm16(r.content)
    check(len(a) > SR // 2 and rms(a) > 0.005, "full PCM: non-silent audio", f"{len(a)/SR:.2f}s rms={rms(a):.4f}")


def test_sse(base: str, voice: str) -> None:
    parts, done = [], False
    with httpx.stream("POST", f"{base}/v1/audio/speech",
                      json={"input": "Streaming over server sent events.", "voice": voice, "stream_format": "sse"},
                      timeout=120) as r:
        check(r.status_code == 200, "SSE: 200", f"HTTP {r.status_code}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            evt = json.loads(line[5:].strip())
            if evt.get("type") == "speech.audio.delta":
                parts.append(pcm16(base64.b64decode(evt["audio"])))
            elif evt.get("type") == "speech.audio.done":
                done = True
            elif evt.get("type") == "error":
                check(False, "SSE: no error event", evt.get("error", ""))
    check(done, "SSE: got done event")
    check(len(parts) >= 1, "SSE: received audio deltas", f"{len(parts)} chunks")
    if parts:
        a = np.concatenate(parts)
        check(rms(a) > 0.005, "SSE: non-silent audio", f"{len(a)/SR:.2f}s rms={rms(a):.4f}")


def test_pcm_stream(base: str, voice: str) -> None:
    buf = bytearray()
    with httpx.stream("POST", f"{base}/v1/audio/speech",
                      json={"input": "Raw chunked PCM stream.", "voice": voice, "stream_format": "audio"},
                      timeout=120) as r:
        check(r.status_code == 200, "PCM stream: 200", f"HTTP {r.status_code}")
        check(r.headers.get("x-sample-rate") == str(SR), "PCM stream: x-sample-rate header", r.headers.get("x-sample-rate", ""))
        for chunk in r.iter_bytes():
            buf.extend(chunk)
    a = pcm16(bytes(buf))
    check(len(a) > SR // 2 and rms(a) > 0.005, "PCM stream: non-silent audio", f"{len(a)/SR:.2f}s rms={rms(a):.4f}")


async def test_websocket(base: str, voice: str) -> None:
    import websockets
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/v1/ws/speech"
    frames, started, done = bytearray(), False, False
    try:
        async with websockets.connect(ws_url, max_size=None, open_timeout=30) as ws:
            await ws.send(json.dumps({"type": "generate", "input": "WebSocket streaming test.",
                                      "voice": voice, "request_id": "e2e-1"}))
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
                if isinstance(msg, bytes):
                    frames.extend(msg)
                else:
                    evt = json.loads(msg)
                    if evt.get("type") == "generation.started":
                        started = True
                    elif evt.get("type") == "generation.done":
                        done = True
                        break
                    elif evt.get("type") == "error":
                        check(False, "WS: no error", evt.get("error", ""))
                        break
    except Exception as e:
        check(False, "WS: connected + streamed", repr(e))
        return
    check(started, "WS: started event")
    check(done, "WS: done event")
    a = pcm16(bytes(frames))
    check(len(a) > SR // 2 and rms(a) > 0.005, "WS: non-silent audio", f"{len(a)/SR:.2f}s rms={rms(a):.4f}")


# ── Properties unit tests can't prove ────────────────────────────────────────

def test_voice_distinctness(base: str, voices: list[str]) -> None:
    """Same text, two different voices → audibly different output (embedding works)."""
    if len(voices) < 2:
        check(False, "voice distinctness: need ≥2 voices", f"have {voices}")
        return
    text = "Hello, this is a voice comparison test."
    outs = []
    for v in voices[:2]:
        r = httpx.post(f"{base}/v1/audio/speech", json={"input": text, "voice": v}, timeout=120)
        if r.status_code != 200:
            check(False, f"voice distinctness: {v} generated", f"HTTP {r.status_code}")
            return
        _, a = wav_read(io.BytesIO(r.content))
        outs.append(a.astype(np.float64))
    n = min(len(outs[0]), len(outs[1]))
    if n == 0:
        check(False, "voice distinctness: non-empty", "")
        return
    a, b = outs[0][:n], outs[1][:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    corr = float(np.dot(a, b) / denom)
    check(corr < 0.95, f"voice distinctness ({voices[0]} vs {voices[1]})", f"waveform corr={corr:.3f} (want <0.95)")


def test_second_speaker_no_crash(base: str, voice: str) -> None:
    """The vLLM #28307 regression: 2nd consecutive prompt_embeds request kills the engine."""
    for i in (1, 2, 3):
        r = httpx.post(f"{base}/v1/audio/speech", json={"input": f"Speaker request number {i}.", "voice": voice}, timeout=120)
        if not check(r.status_code == 200, f"speaker request #{i} ok (no engine death)", f"HTTP {r.status_code}"):
            return


def test_long_form(base: str, voice: str) -> None:
    text = ("This is a deliberately long passage designed to exceed the long form threshold. " * 12).strip()
    r = httpx.post(f"{base}/v1/audio/speech", json={"input": text, "voice": voice, "enable_long_form": True}, timeout=300)
    ok = check(r.status_code == 200, "long-form: 200", f"HTTP {r.status_code}")
    if ok:
        sr, a = wav_read(io.BytesIO(r.content))
        check(len(a) / sr > 10, "long-form: produced long audio", f"{len(a)/sr:.1f}s")


def test_error_paths(base: str, voices: list[str]) -> None:
    r = httpx.post(f"{base}/v1/audio/speech", json={"input": "x", "voice": "definitely_not_a_voice_zzz"}, timeout=30)
    check(r.status_code == 400, "unknown voice → 400", f"HTTP {r.status_code}")
    r = httpx.post(f"{base}/v1/audio/speech", json={"input": "", "voice": voices[0]}, timeout=30)
    check(r.status_code in (400, 422, 500), "empty input rejected", f"HTTP {r.status_code}")


def test_concurrency(base: str, voice: str, n: int) -> None:
    async def one(client, i):
        r = await client.post(f"{base}/v1/audio/speech", json={"input": f"Concurrent request {i}.", "voice": voice}, timeout=180)
        return r.status_code, len(r.content)

    async def run():
        async with httpx.AsyncClient() as c:
            t0 = time.monotonic()
            res = await asyncio.gather(*[one(c, i) for i in range(n)], return_exceptions=True)
            return res, time.monotonic() - t0

    res, dt = asyncio.run(run())
    ok = sum(1 for r in res if isinstance(r, tuple) and r[0] == 200)
    check(ok == n, f"concurrency={n}: all succeeded", f"{ok}/{n} ok in {dt:.1f}s")


# ── Intelligibility: round-trip ASR (the real correctness signal) ────────────

def _wer(ref: str, hyp: str) -> float:
    def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
    r, h = norm(ref), norm(hyp)
    if not r:
        return 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def test_asr_intelligibility(base: str, voice: str) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        check(False, "ASR: faster-whisper installed", "pip install faster-whisper")
        return
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming technology.",
        "She sells seashells by the seashore.",
    ]
    try:
        model = WhisperModel("small", device="cuda", compute_type="float16")
        model.transcribe(np.zeros(16000, dtype=np.float32))  # probe CUDA libs
    except Exception as e:
        print(f"        (whisper CUDA unavailable: {type(e).__name__}; using CPU)")
        model = WhisperModel("small", device="cpu", compute_type="int8")
    wers = []
    for s in sentences:
        r = httpx.post(f"{base}/v1/audio/speech", json={"input": s, "voice": voice}, timeout=120)
        if r.status_code != 200:
            check(False, "ASR: generation ok", f"HTTP {r.status_code}")
            return
        sr, audio = wav_read(io.BytesIO(r.content))
        af = audio.astype(np.float32) / 32768.0
        segs, _ = model.transcribe(af, language="en", beam_size=5)
        hyp = " ".join(seg.text for seg in segs)
        w = _wer(s, hyp)
        wers.append(w)
        print(f"        ref: {s!r}\n        asr: {hyp.strip()!r}  → WER={w:.2f}")
    avg = sum(wers) / len(wers)
    check(avg < 0.25, "ASR intelligibility: avg WER < 0.25", f"avg WER={avg:.3f}")


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--asr", action="store_true", help="run round-trip ASR WER check (needs faster-whisper)")
    ap.add_argument("--concurrency", type=int, default=0, help="run N concurrent requests")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print(f"\n=== E2E correctness test against {base} ===\n")
    print("[health]")
    voices = test_health(base)
    v0 = "speaker_1" if "speaker_1" in voices else (voices[0] if voices else "random")

    print("\n[output modes]")
    test_full_wav(base, v0)
    test_full_pcm(base, v0)
    test_sse(base, v0)
    test_pcm_stream(base, v0)
    asyncio.run(test_websocket(base, v0))

    print("\n[correctness properties]")
    test_voice_distinctness(base, voices)
    test_second_speaker_no_crash(base, v0)
    test_long_form(base, v0)
    test_error_paths(base, voices)

    if args.concurrency:
        print("\n[concurrency]")
        test_concurrency(base, v0, args.concurrency)

    if args.asr:
        print("\n[intelligibility / round-trip ASR]")
        test_asr_intelligibility(base, v0)

    n_fail = sum(1 for ok, _, _ in results if not ok)
    n_pass = len(results) - n_fail
    print(f"\n=== {n_pass} passed, {n_fail} failed ===")
    if n_fail:
        print("FAILURES:")
        for ok, name, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
