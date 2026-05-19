---
type: concept
title: Logix
created: 2026-05-05
updated: 2026-05-05
sources: [logix]
tags: [software, influence-function, llm-scale, pytorch-hooks]
---

# Logix

A PyTorch-based software package for LLM-scale data valuation and influence analysis. Built on hooks, so it interoperates with FSDP, autocast, compile, HF Transformers, DeepSpeed without API friction. Converts existing training code into valuation code via a context manager around the training loop.

Implements [[concepts/logra|LoGra]] as the default scalable influence method, but is extensible — users can plug in custom statistics (e.g., other gradient projections) via hooks.

Code: `https://github.com/logix-project/logix`.

See [[sources/logix]] for full details.

## See also

- [[concepts/logra]]
- [[concepts/influence-function]]
