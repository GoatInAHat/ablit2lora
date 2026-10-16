"""ablit2lora: compact artifacts and serving recipes for abliterated models.

Pure converter: the abliteration labs (audnai/penclaw, orcarouter, dealignai,
huihui-ai, ...) do the science and publish modified checkpoints; ablit2lora
diffs such a checkpoint against its base and re-expresses the edit as a
MB-scale PEFT LoRA adapter, then emits native multi-adapter vLLM/SGLang
serving recipes for it. ablit2lora never finds refusal directions itself.
"""

__version__ = "0.4.0"
