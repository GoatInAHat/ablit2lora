"""EXPERIMENTAL, UNVERIFIED: GLM-5.3-Flash native vLLM LoRA registration.

This is a head-start for vLLM maintainers, not a validated feature. It has
never been run against a live vLLM here, and its coverage of GLM-5.3-Flash's
attention stack is known-incomplete (see the TODO below). Tracking:
vllm-project/vllm#56327.

Which architecture. The official ``zai-org/GLM-5.3-Flash`` config.json
declares ``"architectures": ["Glm5NextForConditionalGeneration"]`` (with a
``text_config`` wrapper); the text-only ``Glm5NextForCausalLM`` is a separate
class. Both are registered here, each through its own lazy ``module:ClassName``
string, so the override takes effect for whichever name a checkpoint declares.

Background (read from vLLM main, 2026-10-17; nothing here was run):

* ``Glm5NextForCausalLM`` (``vllm/models/glm5next/common/model.py``, exported
  from ``vllm.models.glm5next``) does not declare ``SupportsLoRA`` or a
  ``packed_modules_mapping``, unlike the architecturally similar
  ``Glm4MoeForCausalLM`` and ``DeepseekV2ForCausalLM``. The override below
  adds that declaration.
* ``Glm5NextForConditionalGeneration`` (same file and export) derives from
  ``Glm4vForConditionalGeneration``, so upstream it already carries
  ``SupportsLoRA``, ``SupportsMultiModal``, ``get_mm_mapping`` and a
  ``packed_modules_mapping`` for ``gate_up_proj``. Its override here is a thin
  subclass that restates ``SupportsLoRA``; it adds nothing the stock class
  lacks, and exists so both architecture names resolve to this plugin.
* What the stock classes are still missing is the MLA/indexer runtime hooks
  and ``embedding_modules`` that vllm-project/vllm#56327 (open) adds. A
  plugin class body cannot supply those hooks. Per that PR's discussion,
  adapters on ``kv_b_proj``, ``wk`` and ``weights_proj`` load but are not
  applied without them, and ``lm_head`` adapters are dropped without
  ``embedding_modules``.
* Adapter module names come from the converted checkpoints' tensor names.
  vLLM's ``hf_to_vllm_mapper`` maps ``model.language_model.`` to
  ``language_model.model.``; a ``model.layers.`` naming (the text-only HF
  class) would not be renamed and would match no module on the
  ConditionalGeneration class.

The registration goes through vLLM's out-of-tree ``ModelRegistry.register_model``
plugin mechanism, loaded via the ``vllm.general_plugins`` entry point (see
pyproject.toml). It does not patch, fork, or vendor vLLM.

CRITICAL HONESTY: this has NOT been exercised against a live vLLM engine or
a real GLM-5.3-Flash checkpoint anywhere in this repository's history. It is
unverified against GLM-5.3-Flash's KDA (gated linear attention) and MLA
(multi-head latent attention) projections specifically -- see the TODO
inside ``Glm5NextForCausalLMWithLoRA`` below. Do not represent this as a
working feature; it is a registration scaffold.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_CAUSAL_ARCH = "Glm5NextForCausalLM"
_CONDITIONAL_ARCH = "Glm5NextForConditionalGeneration"

# The lazy "module:ClassName" string form vLLM recommends for out-of-tree
# model registration (see register() below), rather than handing
# ModelRegistry an already-built class object.
_CAUSAL_CLASS_PATH = "ablit2lora.vllm_plugins.glm5next_lora:Glm5NextForCausalLMWithLoRA"
_CONDITIONAL_CLASS_PATH = (
    "ablit2lora.vllm_plugins.glm5next_lora:Glm5NextForConditionalGenerationWithLoRA"
)

try:
    from vllm.model_executor.models.interfaces import SupportsLoRA
    from vllm.models.glm5next import Glm5NextForCausalLM as _StockCausalLM
except ImportError:
    # vLLM is not installed, or this build predates vllm.models.glm5next.
    # Guarded at module level -- not just inside register() -- so that merely
    # importing this module is always safe; the classes below (needed for the
    # lazy registration strings to resolve to anything) are only defined when
    # their own base classes actually exist.
    SupportsLoRA = None
    _StockCausalLM = None

try:
    from vllm.models.glm5next import (
        Glm5NextForConditionalGeneration as _StockConditional,
    )
except ImportError:
    _StockConditional = None


if _StockCausalLM is not None:

    class Glm5NextForCausalLMWithLoRA(_StockCausalLM, SupportsLoRA):
        """``Glm5NextForCausalLM`` plus the ``SupportsLoRA`` marker.

        UNVERIFIED -- see module docstring. Structurally sufficient to pass
        vLLM's own ``supports_lora()`` check (it only requires the
        ``packed_modules_mapping``/``embedding_modules`` attributes to
        exist, which ``SupportsLoRA``'s Protocol defaults already provide);
        whether vLLM's LoRA runtime then actually loads and applies an
        adapter correctly on this specific hybrid KDA+MLA+MoE model has not
        been checked.

        Module-level (not nested inside register()) so it has a plain,
        re-importable dotted path for register()'s lazy "module:ClassName"
        registration string to resolve, and so the class itself is picklable
        by reference rather than only by value.
        """

        # By precedent, not by running this: the dense/shared-expert MLP
        # fuses gate_proj and up_proj into one gate_up_proj parameter,
        # exactly like Glm4MoeForCausalLM and DeepseekV2ForCausalLM (both
        # already SupportsLoRA upstream), and like the sister
        # Glm5NextForConditionalGeneration class in the same vLLM source
        # file, which carries this identical entry today. LoRA on a fused
        # column-parallel projection needs this mapping to slice
        # per-original-module adapter weights correctly. ablit2lora's usual
        # abliteration targets, o_proj and down_proj, are plain
        # RowParallelLinear and need no entry here at all.
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


if _StockConditional is not None:

    class Glm5NextForConditionalGenerationWithLoRA(_StockConditional, SupportsLoRA):
        """``Glm5NextForConditionalGeneration`` with ``SupportsLoRA`` stated.

        UNVERIFIED, and deliberately thin: upstream this class already
        inherits ``SupportsLoRA``, ``get_mm_mapping`` and the
        ``gate_up_proj`` packed mapping from ``Glm4vForConditionalGeneration``
        (see module docstring), so this subclass changes no behaviour of the
        stock class. The remaining gaps are the MLA/indexer hooks and
        ``embedding_modules`` that vllm-project/vllm#56327 adds.
        """


def _registrations():
    return (
        (_CAUSAL_ARCH, _CAUSAL_CLASS_PATH, _StockCausalLM),
        (_CONDITIONAL_ARCH, _CONDITIONAL_CLASS_PATH, _StockConditional),
    )


def register() -> None:
    """``vllm.general_plugins`` entry point.

    Called with no arguments by ``vllm.plugins.load_general_plugins()`` once
    per process (engine core and each worker) at vLLM startup; return value
    is ignored. Must never raise -- a broken plugin must not take down
    everyone else's vLLM startup just because ablit2lora is installed.
    """
    try:
        from vllm import ModelRegistry
    except ImportError:
        logger.debug(
            "ablit2lora glm5next_lora: vllm is not installed; skipping"
        )
        return

    # Registered via the lazy "module:ClassName" string form -- vLLM's own
    # recommendation for out-of-tree models -- rather than passing the class
    # object directly: ModelRegistry resolves (imports) the class itself,
    # lazily, which avoids import-order/CUDA-init issues in forked workers.
    # Each architecture registers independently, so a build that has only
    # one of the two classes still gets that one.
    for arch, class_path, stock in _registrations():
        if stock is None:
            logger.debug(
                "ablit2lora glm5next_lora: %s not importable from "
                "vllm.models.glm5next (this build predates it); skipping",
                arch,
            )
            continue
        try:
            ModelRegistry.register_model(arch, class_path)
        except Exception:
            logger.warning(
                "ablit2lora glm5next_lora: registering %s failed; skipping",
                arch,
                exc_info=True,
            )
            continue
        logger.info(
            "ablit2lora: registered an EXPERIMENTAL, UNVERIFIED SupportsLoRA "
            "override for %s -- see ablit2lora.vllm_plugins.glm5next_lora's "
            "module docstring before relying on this",
            arch,
        )
