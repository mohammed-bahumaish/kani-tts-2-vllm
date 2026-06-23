"""KaniTTS2ForCausalLM — vLLM model class for KaniTTS-2 (modified LFM2).

Four customisations on top of the stock vLLM Lfm2ForCausalLM:
  1. Frame-level position remapping (audio frames share a position)
  2. Learnable per-layer RoPE (alpha-scaled inverse frequencies)
  3. Speaker embedding projection (128-d → hidden_size, inserted at pos 1)
  4. Custom weight mapping (learnable_rope_layers → rotary_emb)

Uses composition: wraps vLLM's Lfm2Model and manually iterates layers so we
can inject per-layer learnable RoPE and remapped positions.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.lfm2 import (
    Lfm2ForCausalLM,
    Lfm2Model,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .config import (
    ALPHA_MAX,
    ALPHA_MIN,
    AUDIO_STEP,
    AUDIO_TOKENS_START,
    END_OF_AI,
    END_OF_SPEECH,
    SPEAKER_EMB_DIM,
    TOKENS_PER_FRAME,
    USE_LEARNABLE_ROPE,
)
from .positions import remap_positions_prefill
from .rope import LearnableRotaryEmbedding


class KaniTTS2ForCausalLM(nn.Module, HasInnerState, SupportsPP, IsHybrid):
    """KaniTTS-2 model for vLLM with frame-level positions and learnable RoPE."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.config = hf_config

        # --- KaniTTS-2 config (from HF config.json, with defaults) ---
        self.audio_tokens_start = getattr(hf_config, "audio_tokens_start", AUDIO_TOKENS_START)
        self.tokens_per_frame = getattr(hf_config, "tokens_per_frame", TOKENS_PER_FRAME)
        self.audio_step = getattr(hf_config, "audio_step", AUDIO_STEP)
        self.speaker_emb_dim = getattr(hf_config, "speaker_emb_dim", SPEAKER_EMB_DIM)
        self.use_learnable_rope = getattr(hf_config, "use_learnable_rope", USE_LEARNABLE_ROPE)
        self.alpha_min = getattr(hf_config, "alpha_min", ALPHA_MIN)
        self.alpha_max = getattr(hf_config, "alpha_max", ALPHA_MAX)

        # --- Backbone: reuse vLLM Lfm2Model (handles attention, conv, KV cache) ---
        self.model = Lfm2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        # --- Speaker embedding projection (128 → hidden_size) ---
        self.speaker_emb_projection = nn.Linear(
            self.speaker_emb_dim, hf_config.hidden_size, bias=False
        )

        # --- Learnable RoPE: one per attention layer ---
        self.learnable_rope_layers: nn.ModuleList | None = None
        if self.use_learnable_rope:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            layer_types = getattr(hf_config, "layer_types", None)

            # nn.ModuleList with None entries — PyTorch state_dict uses
            # sparse indices matching layer_idx (e.g. 0, 5, 10 for attention
            # layers), which must match the checkpoint keys exactly.
            self.learnable_rope_layers = nn.ModuleList()
            for idx in range(hf_config.num_hidden_layers):
                is_attn = (
                    layer_types[idx] == "full_attention" if layer_types else True
                )
                if is_attn:
                    self.learnable_rope_layers.append(
                        LearnableRotaryEmbedding(
                            head_dim=head_dim,
                            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
                            alpha_min=self.alpha_min,
                            alpha_max=self.alpha_max,
                        )
                    )
                else:
                    self.learnable_rope_layers.append(None)

        # --- Output head (tied to embed_tokens) ---
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                hf_config.vocab_size,
                hf_config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(hf_config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # --- Reduced LM head: only compute logits for valid output tokens ---
        # Audio tokens + END_OF_SPEECH + END_OF_AI. Saves ~80% of the lm_head GEMV.
        audio_start = getattr(hf_config, "audio_tokens_start", AUDIO_TOKENS_START)
        valid_ids = [END_OF_SPEECH, END_OF_AI] + list(range(audio_start, hf_config.vocab_size))
        self.register_buffer(
            "_valid_token_ids", torch.tensor(valid_ids, dtype=torch.long)
        )
        self._full_vocab_size = hf_config.vocab_size

        # --- Per-request decode position tracking ---
        # Records each request's first-audio position, indexed by its mamba/
        # short-conv cache SLOT (state_indices_tensor). register_buffer keeps it
        # on-device. Slot values aren't bounded by max_num_seqs, so size it
        # generously.
        self._slot_capacity = 16384
        self.register_buffer(
            "_first_audio_pos",
            torch.zeros(self._slot_capacity, dtype=torch.long),
            persistent=False,
        )


    # ------------------------------------------------------------------ #
    # HasInnerState delegates (conv cache for hybrid LFM2 layers)
    # ------------------------------------------------------------------ #

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return Lfm2ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return Lfm2ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Lfm2ForCausalLM.get_mamba_state_copy_func()

    # ------------------------------------------------------------------ #
    # Embedding helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_conv_metadata():
        """Short-conv (mamba) attention metadata for this forward pass, or None.

        vLLM V1 keeps it on the forward context, keyed by layer prefix. Any
        short-conv layer's metadata carries the batch-wide ``state_indices_tensor``
        and the prefill/decode split (``num_decodes`` / ``num_decode_tokens`` /
        ``num_prefills`` / ``query_start_loc_p``). Returns None on profiling runs.
        """
        try:
            from vllm.forward_context import get_forward_context

            md = get_forward_context().attn_metadata
        except Exception:
            return None
        if md is None:
            return None
        if isinstance(md, dict):
            for v in md.values():
                if getattr(v, "state_indices_tensor", None) is not None:
                    return v
            return None
        return md if getattr(md, "state_indices_tensor", None) is not None else None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_input_ids(input_ids)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with frame-level position remapping.

        When learnable RoPE is *disabled* we simply remap ``positions`` and
        delegate to the stock ``Lfm2Model.forward()``.

        When learnable RoPE is *enabled* we must iterate layers ourselves so
        that each attention layer receives its own (cos, sin) pair produced by
        its ``LearnableRotaryEmbedding`` using the remapped positions.

        Position handling (driven by the V1 short-conv attention metadata, which
        splits the batch into decode-first then prefill tokens):
          - Prefill with token IDs: remap via ``remap_positions_prefill``
          - Prefill with embeds (speaker injection): keep sequential positions
            (all text/special tokens, no audio yet)
          - Decode: frame-level audio positions derived from the absolute
            position and the request's recorded first-audio position
        """

        # --- 1. Remap positions to frame-level (all 4 tokens of an audio frame
        #        share one RoPE position) ---
        #
        # vLLM V1 reorders every batch as [decode tokens | prefill tokens] and
        # exposes the per-request cache slots + the prefill/decode split on the
        # short-conv (mamba) attention metadata via get_forward_context() — NOT
        # via a kwarg (that older API was removed). We read it from there.
        #
        #   - Decode tokens (1 per request, all audio): position =
        #     first_audio_pos[slot] + ((abs_pos - first_audio_pos[slot]) // 4) * step
        #   - Prefill tokens: remap via remap_positions_prefill (token IDs) or keep
        #     vLLM's sequential positions (speaker-embedding prompt has no audio).
        #     We record first_audio_pos[slot] = last prefill position + 1.
        md = self._get_conv_metadata()
        if md is not None:
            si = md.state_indices_tensor
            n_dec = md.num_decodes
            n_dec_tok = md.num_decode_tokens
            n_pre = md.num_prefills
            new_pos = positions.clone()

            if n_dec > 0:
                slots_d = si[:n_dec]
                F = self._first_audio_pos[slots_d]
                frame = torch.div(
                    positions[:n_dec_tok] - F, self.tokens_per_frame,
                    rounding_mode="floor",
                )
                new_pos[:n_dec_tok] = F + (frame.to(torch.float32) * self.audio_step).long()

            if n_pre > 0:
                slots_p = si[n_dec:n_dec + n_pre]
                qsl = md.query_start_loc_p  # [n_pre+1], offsets within prefill region
                for j in range(n_pre):
                    s = n_dec_tok + int(qsl[j].item())
                    e = n_dec_tok + int(qsl[j + 1].item())
                    if input_ids is not None:
                        new_pos[s:e] = remap_positions_prefill(
                            input_ids[s:e],
                            audio_tokens_start=self.audio_tokens_start,
                            tokens_per_frame=self.tokens_per_frame,
                            audio_step=self.audio_step,
                        )
                    # else: speaker-embedding prefill — vLLM sequential positions
                    # are already correct (no audio tokens in the prompt).
                    self._first_audio_pos[int(slots_p[j].item())] = (
                        int(new_pos[e - 1].item()) + 1
                    )
            positions = new_pos
        # else: profiling / dummy run (no attn metadata) — leave positions as-is.

        # --- 2a. Standard path (no learnable RoPE) ---
        if not self.use_learnable_rope:
            hidden_states = self.model(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
            return hidden_states

        # --- 2b. Learnable RoPE path: iterate layers manually ---
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.model.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer_idx, layer in enumerate(
            islice(self.model.layers, self.model.start_layer, self.model.end_layer)
        ):
            actual_idx = self.model.start_layer + layer_idx
            rope_module = self.learnable_rope_layers[actual_idx] if self.learnable_rope_layers else None

            if rope_module is not None:
                # Attention layer — replace rotary_emb.forward temporarily
                # so that the layer uses our learnable (cos, sin).
                attn = layer.self_attn  # Lfm2Attention
                original_rotary_emb = attn.rotary_emb

                # Swap in our learnable RoPE
                attn.rotary_emb = rope_module
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
                # Restore original (so weight loading / state_dict stays clean)
                attn.rotary_emb = original_rotary_emb
            else:
                # Conv layer — pass through unchanged
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.model.embedding_norm(hidden_states, residual)
        return hidden_states

    # ------------------------------------------------------------------ #
    # Logits
    # ------------------------------------------------------------------ #

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Restrict generation to valid output tokens: audio codes + END_OF_SPEECH
        # + END_OF_AI. Besides saving ~80% of the lm_head GEMV, this guarantees
        # the model can only ever emit in-vocab audio/stop tokens — which keeps
        # vLLM's frequency-penalty scatter in-bounds (the full head can sample
        # padded-vocab ids that overflow the penalty bin and crash the engine).
        weight = self.lm_head.weight
        reduced_weight = weight[self._valid_token_ids]
        reduced_logits = torch.nn.functional.linear(hidden_states, reduced_weight)

        full_logits = torch.full(
            (hidden_states.shape[0], self._full_vocab_size),
            float("-inf"),
            device=hidden_states.device,
            dtype=reduced_logits.dtype,
        )
        full_logits[:, self._valid_token_ids] = reduced_logits
        return full_logits

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with custom name mapping.

        Checkpoint prefix ``model.`` wraps both backbone and custom layers.
        Our structure hoists custom layers to the top level:

            Checkpoint                                  → Ours
            model.learnable_rope_layers.{i}.alpha_weight → learnable_rope_layers.{i}.alpha_weight
            model.speaker_emb_projection.weight          → speaker_emb_projection.weight
            lm_head.weight                               → (skip, tied to embed_tokens)
            model.*                                      → model.* (backbone, unchanged)
        """

        def _remap(name: str) -> str | None:
            if name.startswith("model.learnable_rope_layers."):
                return name.replace("model.learnable_rope_layers.", "learnable_rope_layers.", 1)
            if name.startswith("model.speaker_emb_projection."):
                return name.replace("model.speaker_emb_projection.", "speaker_emb_projection.", 1)
            if name == "lm_head.weight":
                return None
            return name

        remapped: list[tuple[str, torch.Tensor]] = []
        loaded_names: set[str] = set()

        for name, tensor in weights:
            new_name = _remap(name)
            if new_name is None:
                loaded_names.add(name)
                continue
            remapped.append((new_name, tensor))

        loader = AutoWeightsLoader(self)
        loaded_names.update(loader.load_weights(iter(remapped)))
        return loaded_names
