# Third-party notices

This repository includes frozen benchmark samples and a pinned evaluator so
that experiment inputs and grading behavior can be audited. These materials
remain subject to their upstream licenses; the repository's Apache-2.0 license
does not replace those terms.

## MMLU-Pro

- Project: `TIGER-Lab/MMLU-Pro`
- Source: https://github.com/TIGER-AI-Lab/MMLU-Pro
- Dataset mirror: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- Pinned revision: `b189ec765aa7ed75c8acfea42df31fdae71f97be`
- Upstream license: Apache License 2.0

The public repository contains only the frozen evaluation slice used by this
experiment. File hashes and provenance are recorded in
`data/source_manifest.json`.

## Google Research IFEval

- Project: `google-research/google-research/instruction_following_eval`
- Source: https://github.com/google-research/google-research/tree/d36068b845da4c2b24927fee2cea1e6ef98dadda/instruction_following_eval
- Pinned revision: `d36068b845da4c2b24927fee2cea1e6ef98dadda`
- Evaluator source code: Apache License 2.0
- Dataset files: Creative Commons Attribution 4.0 International (CC BY 4.0), under the Google Research repository-wide data notice
- Google Research license: https://github.com/google-research/google-research/blob/master/LICENSE
- CC BY 4.0 terms: https://creativecommons.org/licenses/by/4.0/legalcode

The copied evaluator source retains its upstream copyright headers. The
selected input rows retain their upstream provenance.
