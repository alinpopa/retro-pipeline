# Pathway: Dry-Run (No GPU Required)

Dry-run mode generates schema-valid stub outputs for every stage.  
It is intended for:

- CI wiring checks
- local development on non-GPU machines
- smoke-testing CLI and file contracts

## Command

```bash
cd retro_pipeline
source .venv/bin/activate

./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --dry-run \
  --backend cpu \
  --top-n 5
```

## Typical runtime

- Small override (`--num_designs` in stage command): seconds
- Default config (`num_backbones=10000`): can still be minutes because many stubs are written

## Suggested fast smoke sequence

```bash
python -m scripts.run_rfdiffusion --config configs/sox2.yaml --out_dir workspace/01_backbones --num_designs 4 --dry-run
python -m scripts.run_proteinmpnn --config configs/sox2.yaml --in_dir workspace/01_backbones --out_dir workspace/02_sequences --num_seq_per_target 5 --dry-run
python -m scripts.run_protenix --config configs/sox2.yaml --in_dir workspace/02_sequences --out_dir workspace/03_predictions --max_jobs_per_shard 10 --dry-run
python -m scripts.run_foldx_filter --config configs/sox2.yaml --in_dir workspace/03_predictions --out_dir workspace/04_thermodynamics --dry-run
python -m scripts.score_and_rank --predictions workspace/03_predictions --thermo workspace/04_thermodynamics --top_n 5
```

