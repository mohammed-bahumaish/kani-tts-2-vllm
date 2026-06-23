# KaniTTS-2 vLLM inference server.
#
# Build context is THIS directory:
#   docker build -t kanitts2-server .
#
# Base: vastai/vllm ships vLLM + PyTorch + CUDA prebuilt. Pick a tag whose
# CUDA version matches your GPU/driver (12.x for Ada/Hopper, 13.x for Blackwell
# / RTX 5090). Any image with vLLM >= 0.16.0 works.
FROM vastai/vllm:v0.16.0-cuda-13.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# =============================================================================
# STEP 1: Install the kanitts2-vllm plugin (registers KaniTTS2ForCausalLM via
# the vllm.general_plugins entry point so it's available in every vLLM process).
# =============================================================================
COPY kanitts2-vllm /app/kanitts2-vllm
RUN pip install --no-cache-dir --no-deps \
    --target=/usr/local/lib/python3.12/dist-packages \
    /app/kanitts2-vllm

# =============================================================================
# STEP 2: Install the tts-server package + runtime deps.
# The base image's pip targets its own venv (/opt/sys-venv); --target forces
# install into the global site-packages that `python3` actually uses.
# =============================================================================
COPY pyproject.toml /app/pyproject.toml
COPY tts_server /app/tts_server
RUN pip install --no-cache-dir --no-deps \
    --target=/usr/local/lib/python3.12/dist-packages \
    /app \
 && pip install --no-cache-dir --ignore-installed \
    --target=/usr/local/lib/python3.12/dist-packages \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    numpy scipy safetensors "huggingface_hub>=0.34.0,<1.0" einops

ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages:${PYTHONPATH}

# =============================================================================
# STEP 3: Bake the models into the image (separate layer — stays cached unless
# download_models.py changes). Both models are public on the HF Hub; HF_TOKEN
# is only needed if you mirror them behind a gated/private repo.
# =============================================================================
COPY download_models.py /app/
ARG HUGGINGFACE_TOKEN=""
ENV HF_TOKEN=${HUGGINGFACE_TOKEN}
RUN python3 /app/download_models.py

# =============================================================================
# STEP 4: Copy the rest (speaker embeddings, etc.) — doesn't bust the model cache.
# =============================================================================
COPY . /app/

# =============================================================================
# Runtime config
# =============================================================================
ENV VLLM_PLUGINS=kanitts2
ENV HF_HUB_OFFLINE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_MODULE_LOADING=LAZY
# Workaround for CUDA error 803 seen on some recent NVIDIA drivers.
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/cuda/lib64

EXPOSE 8000

CMD ["python", "-m", "tts_server.server"]
