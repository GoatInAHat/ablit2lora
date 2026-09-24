"""ablit2lora: convert published abliterated checkpoints into LoRA adapters.

Pure converter: the abliteration labs (audnai/penclaw, orcarouter, dealignai,
huihui-ai, ...) do the science and publish modified checkpoints; ablit2lora
diffs such a checkpoint against its base and re-expresses the edit as a
MB-scale PEFT LoRA adapter -- one base model copy on disk, hot-swappable in
vLLM via --enable-lora. ablit2lora never finds refusal directions itself.
"""

__version__ = "0.2.0"
