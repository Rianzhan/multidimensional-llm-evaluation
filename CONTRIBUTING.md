# Contributing

1. Create a virtual environment and install `.[dev]`.
2. Keep API credentials in environment variables only.
3. Add or update tests for behavior changes.
4. Run `python -m pytest -q` and `python evaluate.py --dry-run` before opening a pull request.
5. Do not commit raw provider responses, checkpoints, downloaded caches, or personal data.

Dataset changes must update `data/DATA_CARD.md` and the pinned hashes in
`data/source_manifest.json`.
