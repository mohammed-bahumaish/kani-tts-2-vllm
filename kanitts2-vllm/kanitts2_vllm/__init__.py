"""KaniTTS-2 vLLM plugin — registers the custom model with vLLM's ModelRegistry."""

_registered = False


def _patch_penalty_prompt_embeds_oob() -> None:
    """Make vLLM's sampling penalties safe for ``prompt_embeds`` requests.

    ``apply_penalties`` always scatters the request's *prompt* token IDs to build
    the repetition mask. For a ``prompt_embeds`` request those IDs are ``-1``
    placeholders, so the scatter indexes out of bounds and triggers a device-side
    CUDA assert that kills the EngineCore (vLLM #28307). Speaker voices need
    ``prompt_embeds`` AND ``repetition_penalty`` (to stop over-generation), so we
    clamp any out-of-range token ID into the discarded pad bin — the placeholder
    prompt tokens then contribute nothing to the penalty, exactly as intended.
    """
    try:
        from vllm.model_executor.layers import utils as u

        if getattr(u, "_kani_penalty_patched", False):
            return
        _orig = u.get_token_bin_counts_and_mask

        def _safe(tokens, vocab_size, num_seqs):
            tokens = tokens.masked_fill(tokens < 0, vocab_size).clamp(max=vocab_size)
            return _orig(tokens, vocab_size, num_seqs)

        u.get_token_bin_counts_and_mask = _safe
        u._kani_penalty_patched = True
    except Exception:
        # Best-effort: if the vLLM internals move, fall back to no-penalty
        # sampling on the embeds path (configured in the server) rather than fail.
        pass


def register():
    global _registered
    if _registered:
        return
    _registered = True
    from vllm import ModelRegistry

    _patch_penalty_prompt_embeds_oob()

    ModelRegistry.register_model(
        "KaniTTS2ForCausalLM",
        "kanitts2_vllm.model:KaniTTS2ForCausalLM",
    )
