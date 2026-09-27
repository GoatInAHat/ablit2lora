"""ablit2lora: compact artifacts and serving recipes for abliterated models.

Pure converter: the abliteration labs (audnai/penclaw, orcarouter, dealignai,
huihui-ai, ...) do the science and publish modified checkpoints; ablit2lora
diffs such a checkpoint against its base and re-expresses the edit as a
MB-scale PEFT LoRA adapter. Serving recipes support PEFT where available and
separately installed request-local writer-projection runtimes where needed.
ablit2lora never finds refusal directions itself.
"""

__version__ = "0.3.1"
