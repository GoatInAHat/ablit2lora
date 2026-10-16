"""EXPERIMENTAL, UNVERIFIED: GLM-5.3-Flash native vLLM LoRA registration.

This is a head-start for vLLM maintainers, not a validated feature. It has
never been run against a live vLLM here, and its coverage of GLM-5.3-Flash's
attention stack is known-incomplete (see the TODO below). Tracking:
vllm-project/vllm#56327.

Background (verified against vLLM main, 2026-10-16): vLLM's
``Glm5NextForCausalLM`` (``vllm/models/glm5next/common/model.py``, registered
as ``vllm.models.glm5next`` in ``vllm/model_executor/models/registry.py``)
does not declare ``SupportsLoRA`` or a ``packed_modules_mapping`` --
unlike the architecturally similar ``Glm4MoeForCausalLM``
(``vllm/model_executor/models/glm4_moe.py``) and ``DeepseekV2ForCausalLM``
(``vllm/model_executor/models/deepseek_v2.py``), both of which do. vLLM's
generic MoE+LoRA layers (``FusedMoEWithLoRA``) and its "every linear layer
supports LoRA" auto-discovery already exist, so enabling the architecture is
in principle a small per-model declaration -- this module attempts exactly
that declaration via vLLM's out-of-tree ``ModelRegistry.register_model``
plugin mechanism, loaded through the ``vllm.general_plugins`` entry point
(see pyproject.toml). It does not patch, fork, or vendor vLLM.

CRITICAL HONESTY: this has NOT been exercised against a live vLLM engine or
a real GLM-5.3-Flash checkpoint anywhere in this repository's history. It is
unverified against GLM-5.3-Flash's KDA (gated linear attention) and MLA
(multi-head latent attention) projections specifically -- see the TODO
inside ``register()`` for exactly what remains unchecked and why. Do not
represent this as a working feature; it is a registration scaffold.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ARCH = "Glm5NextForCausalLM"


def register() -> None:
    """``vllm.general_plugins`` entry point.

    Called with no arguments by ``vllm.plugins.load_general_plugins()`` once
    per process (engine core and each worker) at vLLM startup; return value
    is ignored. Must never raise -- a broken plugin must not take down
    everyone else's vLLM startup just because ablit2lora is installed.
    """
    try:
        from vllm import ModelRegistry
        from vllm.model_executor.models.interfaces import SupportsLoRA
        from vllm.models.glm5next import Glm5NextForCausalLM as _StockGlm5Next
    except ImportError:
        # vLLM is not installed, or this build predates vllm.models.glm5next
        # (confirmed present on vLLM main as of 2026-10-16). Either way there
        # is nothing to patch; do not raise.
        logger.debug(
            "ablit2lora glm5next_lora: vllm.models.glm5next not importable "
            "(vLLM not installed, or this build predates the architecture); "
            "skipping"
        )
        return

    class Glm5NextForCausalLMWithLoRA(_StockGlm5Next, SupportsLoRA):
        """``Glm5NextForCausalLM`` plus the ``SupportsLoRA`` marker.

        UNVERIFIED -- see module docstring. Structurally sufficient to pass
        vLLM's own ``supports_lora()`` check (it only requires the
        ``packed_modules_mapping``/``embedding_modules`` attributes to
        exist, which ``SupportsLoRA``'s Protocol defaults already provide);
        whether vLLM's LoRA runtime then actually loads and applies an
        adapter correctly on this specific hybrid KDA+MLA+MoE model has not
        been checked.
        """

        # Verified by precedent, not by running this: the dense/shared-expert
        # MLP fuses gate_proj and up_proj into one gate_up_proj parameter,
        # exactly like Glm4MoeForCausalLM and DeepseekV2ForCausalLM (both
        # already SupportsLoRA upstream), and like the sister
        # Glm5NextForConditionalGeneration class in this same vLLM source
        # file, which carries this identical entry today (for its own,
        # unrelated per-layer quant-routing reason -- see the comment above
        # its packed_modules_mapping in vllm/models/glm5next/common/model.py).
        # LoRA on a fused column-parallel projection needs this mapping to
        # slice per-original-module adapter weights correctly.
        # ablit2lora's usual abliteration targets, o_proj and down_proj, are
        # plain RowParallelLinear and need no entry here at all.
        packed_modules_mapping = {
            "gate_up_proj": ["gate_proj", "up_proj"],
        }

        # TODO(unverified -- exact remaining check): GLM-5.3-Flash's
        # attention is a hybrid of gated linear attention (KDA) and dense
        # multi-head latent attention (MLA), not plain multi-head attention,
        # and NEITHER attention path's fused projection is declared above:
        #
        # 1. Dense MLA layers (vllm/models/glm5next/common/attention.py)
        #    reuse DeepSeekV2FusedQkvAProjLinear from
        #    vllm/model_executor/models/deepseek_v2.py. DeepseekV2ForCausalLM
        #    only adds packed_modules_mapping["fused_qkv_a_proj"] =
        #    ["q_a_proj", "kv_a_proj_with_mqa"] conditionally -- guarded in
        #    its __init__ by `hasattr(config, "q_lora_rank") and
        #    config.q_lora_rank is not None`. Whether Glm5NextTextConfig
        #    exposes that same gate, or always takes the fused path, is not
        #    verified here.
        # 2. KDA (linear-attention) layers
        #    (vllm/models/glm5next/common/kda.py) fuse q/k/v plus extra
        #    gate components into one in_proj_qkvbfg_a projection (per the
        #    weight-loading stacked-params map in
        #    vllm/models/glm5next/common/model.py: in_proj_qkvbfg_a <-
        #    q_proj/k_proj/v_proj at indices 0/1/2, plus further
        #    beta/forget/gate slices this scaffold does not enumerate). No
        #    vLLM architecture with LoRA support fuses a linear-attention
        #    gate projection this way today, so there is no precedent to
        #    copy here, and this scaffold deliberately leaves it unmapped
        #    rather than guess at a shape.
        #
        # What "verified" would require: load a real GLM-5.3-Flash
        # checkpoint on a live vLLM with this plugin active, try a LoRA
        # adapter whose target_modules is exactly ["o_proj", "down_proj"]
        # (ablit2lora's usual convert output -- needs no
        # packed_modules_mapping entry at all), and confirm it (a) loads,
        # and (b) measurably changes generation output versus the base.
        # Separately: decide whether the KDA/MLA fused projections above
        # need their own packed_modules_mapping entries before a LoRA
        # adapter could ever target them too. Track vllm-project/vllm#56327
        # for the upstream enablement this scaffold is a head start on.

    ModelRegistry.register_model(_ARCH, Glm5NextForCausalLMWithLoRA)
    logger.info(
        "ablit2lora: registered an EXPERIMENTAL, UNVERIFIED SupportsLoRA "
        "override for %s -- see ablit2lora.vllm_plugins.glm5next_lora's "
        "module docstring before relying on this",
        _ARCH,
    )
