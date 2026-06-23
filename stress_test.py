"""Stress test for the KaniTTS-2 vLLM server.

Measures latency (TTFB / total / throughput) across concurrency levels.
Concurrent requests are batched by vLLM up to `max_num_seqs` (engine.py),
so this measures how the server behaves as load approaches that ceiling.

Usage:
    python stress_test.py [--host HOST] [--port PORT] [--max-concurrency N]
"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp

# Test prompts of varying lengths
PROMPTS = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we interact with technology in our daily lives.",
    "Welcome to the text to speech demonstration. This system converts written text into natural sounding speech.",
]

# Concurrency levels to test
DEFAULT_CONCURRENCY_LEVELS = [1, 2, 3, 4, 5, 8, 10]
REQUESTS_PER_LEVEL = 5  # requests per concurrency level


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    voice: str = "speaker_1",
    request_id: int = 0,
) -> dict:
    """Send a single TTS request and measure timing."""
    payload = {
        "input": prompt,
        "voice": voice,
        "response_format": "pcm",  # PCM is faster than WAV (no encoding overhead)
    }

    start = time.monotonic()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            first_byte_time = time.monotonic()
            data = await resp.read()
            end = time.monotonic()

            return {
                "request_id": request_id,
                "status": resp.status,
                "ttfb": first_byte_time - start,  # time to first byte
                "total": end - start,
                "audio_bytes": len(data),
                "prompt_len": len(prompt),
                "error": None,
            }
    except Exception as e:
        end = time.monotonic()
        return {
            "request_id": request_id,
            "status": 0,
            "ttfb": end - start,
            "total": end - start,
            "audio_bytes": 0,
            "prompt_len": len(prompt),
            "error": str(e),
        }


async def send_sse_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    voice: str = "speaker_1",
    request_id: int = 0,
) -> dict:
    """Send a single SSE streaming TTS request and measure timing."""
    payload = {
        "input": prompt,
        "voice": voice,
        "response_format": "pcm",
        "stream_format": "sse",
    }

    start = time.monotonic()
    first_chunk_time = None
    chunk_count = 0
    total_audio_bytes = 0

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            async for line in resp.content:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])

                if data["type"] == "speech.audio.delta":
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                    chunk_count += 1
                    # base64 audio — approximate raw size
                    total_audio_bytes += len(data.get("audio", "")) * 3 // 4

                elif data["type"] == "speech.audio.done":
                    break

            end = time.monotonic()
            return {
                "request_id": request_id,
                "status": resp.status,
                "ttfb": (first_chunk_time or end) - start,
                "total": end - start,
                "audio_bytes": total_audio_bytes,
                "prompt_len": len(prompt),
                "chunks": chunk_count,
                "error": None,
            }
    except Exception as e:
        end = time.monotonic()
        return {
            "request_id": request_id,
            "status": 0,
            "ttfb": (first_chunk_time or end) - start,
            "total": end - start,
            "audio_bytes": 0,
            "prompt_len": len(prompt),
            "chunks": chunk_count,
            "error": str(e),
        }


async def send_pcm_stream_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    voice: str = "speaker_1",
    request_id: int = 0,
) -> dict:
    """Send a PCM streaming TTS request and measure timing."""
    payload = {
        "input": prompt,
        "voice": voice,
        "response_format": "pcm",
        "stream_format": "pcm_stream",
    }

    start = time.monotonic()
    first_chunk_time = None
    chunk_count = 0
    total_audio_bytes = 0

    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            async for chunk in resp.content.iter_any():
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                chunk_count += 1
                total_audio_bytes += len(chunk)

            end = time.monotonic()
            return {
                "request_id": request_id,
                "status": resp.status,
                "ttfb": (first_chunk_time or end) - start,
                "total": end - start,
                "audio_bytes": total_audio_bytes,
                "prompt_len": len(prompt),
                "chunks": chunk_count,
                "error": None,
            }
    except Exception as e:
        end = time.monotonic()
        return {
            "request_id": request_id,
            "status": 0,
            "ttfb": (first_chunk_time or end) - start,
            "total": end - start,
            "audio_bytes": 0,
            "prompt_len": len(prompt),
            "chunks": chunk_count,
            "error": str(e),
        }


async def run_concurrency_level(
    url: str,
    concurrency: int,
    num_requests: int,
    mode: str = "full",
    voice: str = "speaker_1",
) -> list[dict]:
    """Run num_requests at given concurrency level.

    NOTE: With max_num_seqs=8, vLLM batches concurrent requests.
    concurrency=1 submits strictly one-at-a-time for baseline latency;
    higher levels measure batched throughput.

    mode: "full" (default), "sse", or "pcm_stream"
    """
    def _pick_sender(session, url, prompt, voice, request_id):
        if mode == "sse":
            return send_sse_request(session, url, prompt, voice=voice, request_id=request_id)
        elif mode == "pcm_stream":
            return send_pcm_stream_request(session, url, prompt, voice=voice, request_id=request_id)
        else:
            return send_request(session, url, prompt, voice=voice, request_id=request_id)

    # Use a fresh session per request to avoid connection reuse issues
    # with vLLM's prompt_embeds path
    if concurrency == 1:
        results = []
        for i in range(num_requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            async with aiohttp.ClientSession() as session:
                r = await _pick_sender(session, url, prompt, voice, i)
            results.append(r)
        return results

    # Concurrent batches
    results = []
    for batch_start in range(0, num_requests, concurrency):
        batch_size = min(concurrency, num_requests - batch_start)
        async with aiohttp.ClientSession() as session:
            coros = []
            for i in range(batch_size):
                idx = batch_start + i
                prompt = PROMPTS[idx % len(PROMPTS)]
                coros.append(_pick_sender(session, url, prompt, voice, idx))
            batch_results = await asyncio.gather(*coros)
        results.extend(batch_results)

    return results


def print_results(concurrency: int, results: list[dict], mode: str = "full"):
    """Print formatted results for a concurrency level."""
    successful = [r for r in results if r["error"] is None and r["status"] == 200]
    failed = [r for r in results if r["error"] is not None or r["status"] != 200]

    if not successful:
        print(f"\n  Concurrency {concurrency}: ALL {len(results)} REQUESTS FAILED")
        for r in failed[:3]:
            err = r["error"] or f"HTTP {r['status']}"
            print(f"    Error: {err}")
        return

    ttfbs = [r["ttfb"] for r in successful]
    totals = [r["total"] for r in successful]
    audio_sizes = [r["audio_bytes"] for r in successful]

    print(f"\n  Concurrency {concurrency}:")
    print(f"    Requests:    {len(successful)} OK, {len(failed)} failed")
    print(f"    TTFB:        min={min(ttfbs):.2f}s  avg={statistics.mean(ttfbs):.2f}s  "
          f"max={max(ttfbs):.2f}s  p50={statistics.median(ttfbs):.2f}s")
    print(f"    Total:       min={min(totals):.2f}s  avg={statistics.mean(totals):.2f}s  "
          f"max={max(totals):.2f}s  p50={statistics.median(totals):.2f}s")
    print(f"    Audio size:  avg={statistics.mean(audio_sizes)/1024:.1f}KB")
    print(f"    Throughput:  {len(successful) / max(totals):.2f} req/s (wall clock)")

    if mode in ("sse", "pcm_stream"):
        chunks = [r.get("chunks", 0) for r in successful]
        label = "SSE chunks" if mode == "sse" else "PCM chunks"
        print(f"    {label}:  avg={statistics.mean(chunks):.1f}")

    if len(totals) > 1:
        print(f"    Stdev:       ttfb={statistics.stdev(ttfbs):.2f}s  total={statistics.stdev(totals):.2f}s")

    if failed:
        print(f"    Failures:")
        for r in failed[:3]:
            err = r["error"] or f"HTTP {r['status']}"
            print(f"      req#{r['request_id']}: {err}")


async def warmup(url: str, voice: str = "speaker_1"):
    """Send a warmup request to prime the engine.

    NOTE: vLLM V1 has a known issue where switching from token_ids to
    prompt_embeds crashes (scatter/gather assertion). So we warm with
    the SAME voice type that will be used for testing.
    """
    async with aiohttp.ClientSession() as session:
        print(f"  Warming up with voice={voice}...")
        r = await send_request(session, url, "Warmup request.", voice=voice, request_id=-1)
        if r["error"] or r["status"] != 200:
            err = r["error"] or f"HTTP {r['status']}"
            print(f"  Warmup FAILED: {err}")
            return False
        print(f"  Warmup OK: {r['total']:.2f}s")
        return True


async def main():
    parser = argparse.ArgumentParser(description="Stress test KaniTTS-2 server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--max-concurrency", type=int, default=10,
                        help="Maximum concurrency level to test")
    parser.add_argument("--requests", type=int, default=REQUESTS_PER_LEVEL,
                        help="Requests per concurrency level")
    parser.add_argument("--sse", action="store_true", help="Test SSE streaming endpoint")
    parser.add_argument("--pcm-stream", action="store_true", help="Test raw PCM streaming endpoint")
    parser.add_argument("--levels", type=str, default=None,
                        help="Comma-separated concurrency levels (e.g. '1,2,4,8')")
    parser.add_argument("--voice", type=str, default="speaker_1",
                        help="Voice to use for test requests (default: speaker_1)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/v1/audio/speech"

    if args.levels:
        levels = [int(x) for x in args.levels.split(",")]
    else:
        levels = [l for l in DEFAULT_CONCURRENCY_LEVELS if l <= args.max_concurrency]

    if args.pcm_stream:
        mode = "pcm_stream"
        mode_label = "PCM streaming"
    elif args.sse:
        mode = "sse"
        mode_label = "SSE streaming"
    else:
        mode = "full"
        mode_label = "non-streaming PCM"

    print(f"=" * 70)
    print(f"KaniTTS-2 Stress Test")
    print(f"=" * 70)
    print(f"  Server:       {base_url}")
    print(f"  Mode:         {mode_label}")
    print(f"  Voice:        {args.voice}")
    print(f"  Concurrency:  {levels}")
    print(f"  Requests/lvl: {args.requests}")
    print(f"  Total:        {len(levels) * args.requests} requests")
    print()

    # Health check
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                health = await resp.json()
                print(f"  Health: {health['status']} | engine={health['engine_ready']} | codec={health['codec_ready']}")
                print(f"  Speakers: {health.get('speakers', [])}")
        except Exception as e:
            print(f"  Health check FAILED: {e}")
            print(f"  Aborting.")
            return

    print()

    # Warmup
    ok = await warmup(url, voice=args.voice)
    if not ok:
        print("Aborting — warmup failed.")
        return

    # Run tests
    print(f"\n{'=' * 70}")
    print(f"Results ({mode_label}):")
    print(f"{'=' * 70}")

    all_results = {}
    for level in levels:
        num_requests = max(args.requests, level)  # at least 1 batch
        print(f"\n  --- Testing concurrency={level} ({num_requests} requests) ---")

        start = time.monotonic()
        results = await run_concurrency_level(url, level, num_requests, mode=mode, voice=args.voice)
        wall_time = time.monotonic() - start

        all_results[level] = results
        print_results(level, results, mode=mode)
        print(f"    Wall time:   {wall_time:.2f}s")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"Summary Table:")
    print(f"{'=' * 70}")
    print(f"{'Concurrency':>12} {'OK':>4} {'Fail':>5} {'Avg TTFB':>10} {'Avg Total':>11} {'Max Total':>11} {'Throughput':>12}")
    print(f"{'-'*12:>12} {'-'*4:>4} {'-'*5:>5} {'-'*10:>10} {'-'*11:>11} {'-'*11:>11} {'-'*12:>12}")

    for level in levels:
        results = all_results[level]
        successful = [r for r in results if r["error"] is None and r["status"] == 200]
        failed = [r for r in results if r["error"] is not None or r["status"] != 200]

        if successful:
            totals = [r["total"] for r in successful]
            ttfbs = [r["ttfb"] for r in successful]
            throughput = len(successful) / max(totals)
            print(f"{level:>12} {len(successful):>4} {len(failed):>5} "
                  f"{statistics.mean(ttfbs):>9.2f}s {statistics.mean(totals):>10.2f}s "
                  f"{max(totals):>10.2f}s {throughput:>10.2f}/s")
        else:
            print(f"{level:>12} {0:>4} {len(failed):>5} {'N/A':>10} {'N/A':>11} {'N/A':>11} {'N/A':>12}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
