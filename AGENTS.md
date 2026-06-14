# Repository Guidelines

## Project Overview

This repository implements a Python and shell pipeline for designing and
ranking SOX2/KLF4 protein variants. The top-level orchestrator runs
RFdiffusion, ProteinMPNN, Protenix, FoldX, and the final ranking stage.

## Repository Layout

- `pipeline_orchestrator.sh`: end-to-end entry point.
- `scripts/`: Python stage CLIs and shared utilities.
- `configs/`: target-specific YAML configuration.
- `templates/`: checked-in PDB inputs.
- `setup/`: environment and accelerator setup helpers.
- `docs/pathways/`: platform-specific operating guides.
- `workspace/`: generated run output; do not commit it.

## Development

- Use Python 3.10 or newer.
- Install development dependencies with `pip install -e ".[dev]"`.
- Keep changes scoped and preserve the existing CLI and configuration
  conventions.
- Do not require GPU tools or model weights for routine validation.
- Treat generated pipeline outputs, caches, logs, model weights, and local
  virtual environments as untracked artifacts.

## Validation

Run the checks relevant to the change:

```bash
ruff check .
pytest
./pipeline_orchestrator.sh --config configs/sox2.yaml --dry-run --top-n 10
```

There may be no collected tests until a test suite is added; do not treat that
alone as evidence that behavior was validated. Use the dry-run smoke test for
changes that affect stage wiring, CLI arguments, configuration, or output
schemas.

## Commit Safety

Before every `git commit`, use the `pre-commit-secrets` skill in
`.agents/skills/pre-commit-secrets/SKILL.md`. Run `.scripts/check-secrets .`
immediately before the commit, after all intended edits and staging changes.
Do not commit if the scan exits nonzero. Remove the detected secret and rerun
the scan instead of bypassing it.

Never add credentials, private keys, `.env` files, access tokens, or licensed
third-party binaries and model weights to the repository.
