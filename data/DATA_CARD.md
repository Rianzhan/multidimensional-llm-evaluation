# Evaluation Data Card

## Purpose

This frozen suite compares GLM, Kimi, and DeepSeek on difficult knowledge,
instruction following, and product-behavior tasks. Selection is fixed before
model execution and never uses historical model errors.

## Public benchmark components

- **MMLU-Pro (50):** 10 each from math, computer science, physics, health, and
  law. Candidates are ranked by a model-independent multi-step/length proxy;
  one variable-choice item is retained per subject when available.
- **IFEval (20):** URL-free prompts selected greedily for instruction-family
  coverage with seed `20260928`.
- **Stability diagnostic (15):** three frozen MMLU-Pro items per subject. These
  are repeated three times and are not added again to base accuracy.

## Provenance and licenses

Exact revisions, retrieval times, and SHA256 values are recorded in
`source_manifest.json`. MMLU-Pro is attributed under Apache-2.0; the pinned
Google Research IFEval evaluator code is Apache-2.0 and its data files are
covered by the repository-wide CC BY 4.0 data notice. See
`third_party/NOTICE.md` for the exact boundary and links.

## Limitations

The 50-question knowledge slice is intentionally small and optimized for
difficulty, so confidence intervals must accompany rankings. Public benchmarks
may have appeared in model training data. IFEval measures verifiable surface
constraints rather than factual quality. Product cases are reported separately.
