"""TTSEngine — wraps vLLM AsyncLLMEngine for KaniTTS-2 inference.

Handles:
  - Engine initialisation with the kanitts2-vllm plugin
  - Prompt construction (special tokens + text tokenisation)
  - Speaker embedding injection via prompt_embeds
  - Async token streaming
"""

from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator

import torch
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .config import (
    END_OF_AI,
    END_OF_SPEECH,
    END_OF_TEXT,
    END_OF_HUMAN,
    MAX_TOKENS,
    MODEL_NAME,
    REPETITION_PENALTY,
    START_OF_AI,
    START_OF_HUMAN,
    START_OF_SPEECH,
    TEMPERATURE,
    TOP_P,
)
from .utils import load_safetensor_weight

log = logging.getLogger(__name__)


class TTSEngine:
    """Async TTS engine backed by vLLM + KaniTTS-2 plugin."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_model_len: int = 4608,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 12,
        enforce_eager: bool = True,
        enable_prompt_embeds: bool = True,
    ) -> None:
        self.model_name = model_name

        self.engine_args = AsyncEngineArgs(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            dtype="bfloat16",
            disable_log_stats=True,
            # KaniTTS-2 uses LFM2 hybrid architecture (attention + conv layers).
            # The kanitts2_vllm plugin registers KaniTTS2ForCausalLM with vLLM's
            # ModelRegistry, enabling native vLLM inference with custom features:
            # frame-level positions, learnable RoPE, speaker embedding injection.
            trust_remote_code=True,
            # Required since vLLM v0.11.1 for EmbedsPrompt (speaker embedding injection).
            enable_prompt_embeds=enable_prompt_embeds,
            skip_tokenizer_init=True,
            swap_space=0,
            enable_prefix_caching=False,
        )

        self.engine: AsyncLLMEngine | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Cache the model's embedding layer weights for prompt_embeds construction.
        # Loaded lazily on first use.
        self._embed_weight: torch.Tensor | None = None

        self.default_sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            stop_token_ids=[END_OF_SPEECH, END_OF_AI],
            detokenize=False,
        )

        # prompt_embeds path uses the same repetition_penalty as the reference
        # model (curbs over-generation). vLLM's penalty kernel would normally
        # crash on prompt_embeds requests (their prompt token IDs are -1
        # placeholders → out-of-bounds scatter, #28307); the kanitts2 plugin
        # installs a small patch at import time that makes it safe.
        self.embeds_sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            stop_token_ids=[END_OF_SPEECH, END_OF_AI],
            detokenize=False,
        )

    async def initialize(self) -> None:
        """Call during FastAPI startup."""
        if self.engine is None:
            log.info("Initializing vLLM with model=%s", self.model_name)
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            log.info("vLLM engine ready")
            self._get_embed_weight()  # Pre-load instead of lazy first-request

    # ── Prompt construction ─────────────────────────────────────────────

    def build_prompt_token_ids(self, text: str) -> list[int]:
        """Build the token-ID sequence for a TTS request.

        Layout:
            [START_OF_HUMAN] + tokenizer(text) + [END_OF_TEXT, END_OF_HUMAN,
             START_OF_AI, START_OF_SPEECH]

        START_OF_AI and START_OF_SPEECH are included in the prompt so the
        model starts generating audio tokens immediately.
        """
        text_ids = self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        return (
            [START_OF_HUMAN]
            + text_ids
            + [END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
        )

    def _get_embed_weight(self) -> torch.Tensor:
        """Lazy-load only the embedding weight matrix from safetensors.

        Avoids loading the entire HF model (~400M params) just for the
        embedding table.  Uses huggingface_hub to download only the
        relevant shard and safetensors to load the single tensor.
        """
        if self._embed_weight is None:
            log.info("Loading embedding weights from safetensors...")
            self._embed_weight = load_safetensor_weight(
                self.model_name, "model.embed_tokens.weight"
            )
            log.info("Embedding weights loaded")
        return self._embed_weight

    def build_prompt_embeds(
        self,
        text: str,
        speaker_emb: torch.Tensor | None = None,
        speaker_proj_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the full prompt_embeds tensor for a TTS request.

        If ``speaker_emb`` is provided it is projected and inserted at
        position 1 (after START_OF_HUMAN), matching the HF model's
        ``prepare_inputs_for_generation`` behaviour.

        Args:
            text: Input text.
            speaker_emb: [1, 128] speaker embedding (optional).
            speaker_proj_weight: [hidden_size, 128] projection weights.
                If None, speaker embedding is skipped even if provided.

        Returns:
            prompt_embeds: [seq_len, hidden_size] tensor.
        """
        token_ids = self.build_prompt_token_ids(text)
        embed_weight = self._get_embed_weight()

        ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        embeds = embed_weight[ids_tensor]  # [seq_len, hidden_size]

        if speaker_emb is not None and speaker_proj_weight is not None:
            # Project: [1, 128] @ [128, hidden_size] → [1, hidden_size]
            speaker_emb_bf16 = speaker_emb.to(dtype=embed_weight.dtype)
            projected = torch.nn.functional.linear(
                speaker_emb_bf16, speaker_proj_weight
            )  # [1, hidden_size]

            # Insert at position 1: [tok0, speaker, tok1, tok2, ...]
            embeds = torch.cat(
                [embeds[:1], projected, embeds[1:]], dim=0
            )

        return embeds

    # ── Token streaming ─────────────────────────────────────────────────

    async def generate_stream(
        self,
        text: str,
        speaker_emb: torch.Tensor | None = None,
        speaker_proj_weight: torch.Tensor | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncIterator[int]:
        """Async iterator yielding generated token IDs one by one.

        Args:
            text: Input text.
            speaker_emb: Optional [1, 128] speaker embedding.
            speaker_proj_weight: Optional projection weight matrix.
            sampling_params: Override default sampling params.
        """
        assert self.engine is not None, "Call initialize() first"

        request_id = f"tts-{uuid.uuid4().hex}"

        # Use prompt_embeds only if we have both speaker_emb and projection weight
        # Otherwise fall back to token IDs (model will use default speaker)
        if speaker_emb is not None and speaker_proj_weight is not None:
            prompt_embeds = self.build_prompt_embeds(
                text, speaker_emb, speaker_proj_weight
            )
            prompt = {"prompt_embeds": prompt_embeds}
            params = sampling_params or self.embeds_sampling_params
        else:
            # No speaker injection - use token IDs
            # The model should still generate audio with default speaker
            token_ids = self.build_prompt_token_ids(text)
            prompt = {"prompt_token_ids": token_ids}
            params = sampling_params or self.default_sampling_params

        prev_len = 0
        async for request_output in self.engine.generate(
            prompt, params, request_id=request_id
        ):
            new_ids = request_output.outputs[0].token_ids
            for token_id in new_ids[prev_len:]:
                yield token_id
            prev_len = len(new_ids)
