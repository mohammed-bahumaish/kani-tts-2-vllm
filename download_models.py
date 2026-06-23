"""Pre-download models for Docker image baking."""
from huggingface_hub import snapshot_download

print("[download] KaniTTS-2 model...")
snapshot_download("nineninesix/kani-tts-2-pt")

print("[download] NeMo NanoCodec...")
snapshot_download("nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")

print("[download] All models ready")
