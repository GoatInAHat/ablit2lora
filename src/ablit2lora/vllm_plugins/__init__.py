"""Optional, EXPERIMENTAL out-of-tree vLLM model-registry plugins.

Nothing here is imported by ablit2lora itself, by the ``ablit2lora`` CLI, or
by its test suite -- `import ablit2lora` never touches vLLM. Modules in this
subpackage are only ever loaded by vLLM's own ``vllm.general_plugins``
entry-point discovery (see the ``[project.entry-points]`` table in
pyproject.toml), inside a process that already has vLLM installed. See
``glm5next_lora`` for the current (unverified) plugin and its honesty notes.
"""
