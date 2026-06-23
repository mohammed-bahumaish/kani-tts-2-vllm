#!/bin/bash
# Local (non-Docker) install for the KaniTTS-2 vLLM server.
# Requires a CUDA GPU and a working vLLM install (>= 0.16.0).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KANITTS2_DIR="$SCRIPT_DIR/kanitts2-vllm"

echo "=== Step 1: Installing kanitts2-vllm plugin ==="
pip install "$KANITTS2_DIR"

echo "=== Step 2: Installing tts-server (pulls vLLM 0.16+, FastAPI, codec deps) ==="
pip install -e "$SCRIPT_DIR"

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Verify:"
echo "  python -c 'import vllm; print(\"vLLM:\", vllm.__version__)'"
echo "  python -c 'from tts_server.nanocodec import CausalHiFiGANDecoder; print(\"NanoCodec decoder: OK\")'"
echo "  python -c 'import kanitts2_vllm; print(\"kanitts2-vllm plugin: OK\")'"
echo ""
echo "Run the server:"
echo "  python -m tts_server.server"
echo ""
