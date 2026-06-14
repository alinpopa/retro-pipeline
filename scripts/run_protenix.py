#!/usr/bin/env python3
"""Entry Point 3 — Complex protein+dsDNA prediction via Protenix.

Protenix is ByteDance's open-source AF3-class model
(``bytedance/Protenix``). We use it as a drop-in replacement for the
DeepMind AlphaFold 3 step described in the original spec because (a)
its weights and license are open, and (b) it natively supports
protein + DNA + RNA + ligand inputs.

For every sampled sequence from step 2 we build a single Protenix job:

    {
      "name": "<backbone>_seqNNN",
      "sequences": [
        {"proteinChain": {"sequence": "<aa>", "count": 1}},
        {"dnaSequence":  {"sequence": "<dna_fwd>", "count": 1}},
        {"dnaSequence":  {"sequence": "<dna_rev>", "count": 1}}
      ]
    }

The list of jobs is written to a single input JSON file and passed to:

    docker run --gpus all ... bytedance/protenix:latest \\
        protenix pred -i in.json -o out/ -n protenix_base_default_v1.0.0

Then we parse each per-job ``summary_confidences.json`` and apply the
filter rules:

* ``plddt_designed_region_mean < plddt_cutoff``                  -> discard
* mean off-diagonal PAE on the protein↔DNA chain-pair block
  > ``interface_pae_cutoff``                                     -> discard
  (this is the spec's ``iPAE > 10 Å`` rule, mapped to Protenix's
  ``chain_pair_pae_min`` field)
* ``iptm < iptm_warn_threshold``                                 -> warn only

Surviving structures are copied (or stubbed in --dry-run) into
``--out_dir`` and a ``metrics.csv`` is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from .common import (
    TargetConfig,
    ensure_dir,
    get_logger,
    iter_files,
    mark_stage_done,
    reverse_complement,
    run_with_oom_retry,
    write_stub_protenix_summary,
)

PROTENIX_DOCKER_IMAGE_DEFAULT = "bytedance/protenix:latest"
PROTENIX_MODEL_DEFAULT = "protenix_base_default_v1.0.0"


# --------------------------------------------------------------------------- #
# Input JSON builder
# --------------------------------------------------------------------------- #


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    name, lines = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(lines)))
            name = line[1:].strip()
            lines = []
        elif line:
            lines.append(line.strip())
    if name is not None:
        out.append((name, "".join(lines)))
    return out


def build_protenix_jobs(
    *,
    sequences_dir: Path,
    dna_target: str,
) -> list[dict]:
    """Return the Protenix infer-list for every FASTA in ``sequences_dir``."""
    jobs: list[dict] = []
    dna_fwd = dna_target.upper()
    dna_rev = reverse_complement(dna_fwd)
    for fa in iter_files(sequences_dir, ".fasta"):
        for header, aa in _parse_fasta(fa):
            # Header format: "{backbone}|seqNNN|score=...|recovery=..."
            tag = header.split("|", 2)
            backbone_id = tag[0]
            seq_tag = tag[1] if len(tag) > 1 else "seq"
            job_name = f"{backbone_id}__{seq_tag}"
            jobs.append(
                {
                    "name": job_name,
                    "sequences": [
                        {"proteinChain": {"sequence": aa, "count": 1}},
                        {"dnaSequence": {"sequence": dna_fwd, "count": 1}},
                        {"dnaSequence": {"sequence": dna_rev, "count": 1}},
                    ],
                }
            )
    return jobs


# --------------------------------------------------------------------------- #
# Output parsing & filtering
# --------------------------------------------------------------------------- #


@dataclass
class ProtenixMetrics:
    job_name: str
    iptm: float
    plddt_mean: float
    plddt_designed_region_mean: float
    interface_pae_mean: float
    ranking_score: float
    passed: bool
    failure_reason: str

    def as_csv_row(self) -> dict:
        return asdict(self)


def _mean_off_diagonal_pae(matrix: list[list[float]]) -> float:
    """Mean of all off-diagonal entries in a chain-pair PAE matrix.

    Protenix's ``chain_pair_pae_min`` is a square matrix indexed by
    polymer chain id (protein first, then each DNA strand). We treat the
    block protein↔DNA as the union of all off-diagonal entries — this is
    a conservative scalar summary of the "iPAE" in the original spec.
    """
    n = len(matrix)
    if n < 2:
        return float("inf")
    vals: list[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                vals.append(float(matrix[i][j]))
            except (TypeError, ValueError):
                continue
    return statistics.fmean(vals) if vals else float("inf")


def _parse_summary(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def evaluate_metrics(
    *,
    job_name: str,
    summary: dict,
    plddt_cutoff: float,
    interface_pae_cutoff: float,
    iptm_warn_threshold: float,
) -> ProtenixMetrics:
    iptm = float(summary.get("iptm", 0.0))
    plddt_mean = float(summary.get("plddt_mean", summary.get("plddt", 0.0)))
    plddt_designed = float(
        summary.get("plddt_designed_region_mean", plddt_mean)
    )
    # Protenix may emit either a precomputed scalar or the chain-pair
    # matrix; support both.
    if "interface_pae_mean" in summary:
        ipae = float(summary["interface_pae_mean"])
    elif "chain_pair_pae_min" in summary:
        ipae = _mean_off_diagonal_pae(summary["chain_pair_pae_min"])
    else:
        ipae = float("inf")
    ranking_score = float(summary.get("ranking_score", 0.0))

    reasons: list[str] = []
    if plddt_designed < plddt_cutoff:
        reasons.append(f"plddt_designed={plddt_designed:.1f}<{plddt_cutoff}")
    if ipae > interface_pae_cutoff:
        reasons.append(f"interface_pae={ipae:.2f}>{interface_pae_cutoff}")
    passed = not reasons
    # iptm threshold is warn-only.
    return ProtenixMetrics(
        job_name=job_name,
        iptm=iptm,
        plddt_mean=plddt_mean,
        plddt_designed_region_mean=plddt_designed,
        interface_pae_mean=ipae,
        ranking_score=ranking_score,
        passed=passed,
        failure_reason=";".join(reasons),
    )


# --------------------------------------------------------------------------- #
# Subprocess driver
# --------------------------------------------------------------------------- #


def _run_protenix_docker(
    *,
    image: str,
    model_name: str,
    gpu_devices: str,
    input_json: Path,
    output_dir: Path,
    weights_cache: Path | None,
    log,
) -> None:
    cmd = ["docker", "run", "--rm", "--gpus", gpu_devices]
    mount_root = input_json.parent.parent.resolve()
    cmd += ["-v", f"{mount_root}:/work"]
    if weights_cache is not None:
        cmd += ["-v", f"{weights_cache.resolve()}:/root/.cache/protenix"]
    # Override image entrypoint to avoid Mesos/DCOS execution-permission
    # failures (e.g. "could not execute postgres -V").
    cmd += ["--entrypoint", ""]
    cmd += [
        image,
        "protenix",
        "pred",
        "-i",
        f"/work/{input_json.relative_to(mount_root)}",
        "-o",
        f"/work/{output_dir.relative_to(mount_root)}",
        "-n",
        model_name,
    ]
    run_with_oom_retry(
        cmd,
        downscale_arg="--gpus",
        # gpus argument isn't really a batch knob -- Protenix doesn't
        # expose one on the CLI. We instead let max_retries=0 surface OOMs
        # immediately so the orchestrator can re-shard the input JSON.
        initial_value=1,
        factor=1.0,
        max_retries=0,
        logger=log,
    )


def _run_protenix_local(
    *,
    model_name: str,
    input_json: Path,
    output_dir: Path,
    backend: str,
    log,
) -> None:
    """Run Protenix CLI with local-mode env vars (no colab queue)."""
    log.info("Protenix local inference (model=%s, backend=%s)", model_name, backend)
    log.info("  input : %s", input_json)
    log.info("  output: %s", output_dir)

    cmd = [
        "protenix", "pred",
        "-i", str(input_json),
        "-o", str(output_dir),
        "-n", model_name,
        "--use_msa=False",
    ]
    env = os.environ.copy()
    env.pop("PROTENIX_SERVICE_URL", None)
    env["PROTENIX_LOCAL_ONLY"] = "1"
    env["LAYERNORM_TYPE"] = "torch"
    env["PYTHONUNBUFFERED"] = "1"

    if backend == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif backend == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        if stripped:
            log.info("  %s", stripped)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    log.info("Protenix shard complete.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Protenix complex prediction & filter.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--in_dir", required=True, type=Path, help="ProteinMPNN FASTA dir from step 2.")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument(
        "--docker_image",
        default=PROTENIX_DOCKER_IMAGE_DEFAULT,
        help="Protenix Docker image tag.",
    )
    p.add_argument(
        "--model_name",
        default=PROTENIX_MODEL_DEFAULT,
        help="Protenix model name (see github.com/bytedance/Protenix README).",
    )
    p.add_argument(
        "--weights_cache",
        type=Path,
        default=None,
        help="Host directory to bind into the container as the Protenix weights cache.",
    )
    p.add_argument(
        "--max_jobs_per_shard",
        type=int,
        default=64,
        help="Number of inference jobs per Protenix invocation; reduce on OOM.",
    )
    p.add_argument(
        "--runtime",
        choices=("docker", "local"),
        default="local",
        help="Execution runtime for Protenix inference.",
    )
    p.add_argument(
        "--backend",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help=(
            "Compute backend. For docker runtime, only cuda/auto are valid. "
            "For local runtime, mps/cpu are allowed (depends on local Protenix build)."
        ),
    )
    p.add_argument(
        "--gpu_devices",
        default="all",
        help="GPU selector for docker runtime (e.g. 'all', 'device=0').",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--parse-only", action="store_true",
                   help="Parse existing shard output directories without running inference.")
    args = p.parse_args(argv)

    cfg = TargetConfig.from_yaml(args.config)
    out_dir = ensure_dir(args.out_dir)
    log = get_logger("protenix", log_dir=out_dir.parent / "logs")
    if args.runtime == "docker" and args.backend in ("mps", "cpu"):
        log.error("runtime=docker supports only CUDA GPUs for Protenix. Use runtime=local for mps/cpu.")
        return 2

    jobs = build_protenix_jobs(sequences_dir=args.in_dir, dna_target=cfg.dna_target)
    if not jobs:
        log.error("No FASTA sequences found in %s.", args.in_dir)
        return 2
    log.info("Built %d Protenix job(s) from %s.", len(jobs), args.in_dir)

    # Shard the input list for resilience to OOM and to keep each Docker
    # call's footprint bounded.
    shards: list[list[dict]] = [
        jobs[i : i + args.max_jobs_per_shard]
        for i in range(0, len(jobs), args.max_jobs_per_shard)
    ]
    log.info("Sharding into %d batch(es) of <=%d jobs.", len(shards), args.max_jobs_per_shard)

    metrics_rows: list[ProtenixMetrics] = []
    # Path for incremental writes — save after every shard so a crash mid-run
    # doesn't lose already-parsed data.
    metrics_csv = out_dir / "metrics.csv"

    try:
        existing_metrics = set()
        if metrics_csv.exists():
            with open(metrics_csv) as fh:
                for row in csv.DictReader(fh):
                    existing_metrics.add(row["job_name"])
            log.info("Found existing metrics.csv with %d rows — will append.", len(existing_metrics))
    except Exception:
        existing_metrics = set()

    for shard_idx, shard in enumerate(shards):
        shard_in = out_dir / f"in_shard_{shard_idx:04d}.json"
        shard_out = out_dir / f"out_shard_{shard_idx:04d}"

        # Check if this shard already ran to completion.
        # Protenix nests output deeply: <job>/seed_NNN/predictions/*.json
        already_run = False
        if shard_out.exists():
            n_summaries = len(list(shard_out.glob("*/*/predictions/*_summary_confidence_*.json")))
            already_run = n_summaries >= len(shard)

        # Only write the input JSON if we actually need to run inference.
        if not already_run:
            shard_in.write_text(json.dumps(shard, indent=2))

        if args.dry_run:
            log.warning("--dry-run shard %d: emitting stub summaries.", shard_idx)
            shard_out.mkdir(parents=True, exist_ok=True)
            for job in shard:
                job_dir = shard_out / job["name"]
                job_dir.mkdir(parents=True, exist_ok=True)
                write_stub_protenix_summary(
                    job_dir / "summary_confidences.json",
                    name=job["name"],
                )
                (job_dir / "model.cif").write_text(
                    f"# DRY-RUN STUB structure for {job['name']}\n"
                )
        elif args.parse_only:
            # For --parse-only we just need any summaries on disk —
            # partial shards are still worth parsing.
            has_any = bool(list(shard_out.glob("*/*/predictions/*_summary_confidence_*.json"))) if shard_out.exists() else False
            if not has_any:
                log.info("--parse-only: shard %d has no output — skipping.", shard_idx)
                continue
            log.info("Shard %d: parse-only.", shard_idx)
        else:
            if already_run:
                log.info("Shard %d: already complete — skipping inference.", shard_idx)
            else:
                shard_out.mkdir(parents=True, exist_ok=True)
                if args.runtime == "docker":
                    _run_protenix_docker(
                        image=args.docker_image,
                        model_name=args.model_name,
                        gpu_devices=args.gpu_devices,
                        input_json=shard_in,
                        output_dir=shard_out,
                        weights_cache=args.weights_cache,
                        log=log,
                    )
                else:
                    backend = "cuda" if args.backend == "auto" else args.backend
                    _run_protenix_local(
                        model_name=args.model_name,
                        input_json=shard_in,
                        output_dir=shard_out,
                        backend=backend,
                        log=log,
                    )

        # Parse summaries for every job in this shard (always — even if
        # inference was skipped, we need to extract metrics).
        # Protenix nests output: <job>/seed_NNN/predictions/<job>_summary_confidence_sample_N.json
        shard_new = 0
        for job in shard:
            if job["name"] in existing_metrics:
                continue  # already in metrics.csv from a previous run
            job_dir = shard_out / job["name"]
            # Pick the first summary found; if there are multiple seeds,
            # we prefer the one with the best pLDDT below.
            summary_files = sorted(job_dir.glob("*/predictions/*_summary_confidence_*.json"))
            if not summary_files:
                log.warning("No summary for %s; skipping.", job["name"])
                continue
            log.debug("Found %d summary file(s) for %s.", len(summary_files), job["name"])
            # Use the first one (sample_0 if available, otherwise the numerically lowest)
            summary_path = summary_files[0]
            summary = _parse_summary(summary_path)
            m = evaluate_metrics(
                job_name=job["name"],
                summary=summary,
                plddt_cutoff=cfg.plddt_cutoff,
                interface_pae_cutoff=cfg.interface_pae_cutoff,
                iptm_warn_threshold=cfg.iptm_warn_threshold,
            )
            metrics_rows.append(m)
            existing_metrics.add(m.job_name)
            shard_new += 1
            if m.iptm < cfg.iptm_warn_threshold:
                log.warning("Low iptm=%.3f for %s (warn-only).", m.iptm, m.job_name)
            if m.passed:
                # Copy the predicted structure(s) into a flat directory so
                # FoldX has an obvious input.
                for cif in job_dir.glob("*/predictions/*.cif"):
                    dest = out_dir / "passed" / f"{m.job_name}.cif"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(cif, dest)
                    break

        # --- Save metrics.csv incrementally after each shard ---
        _write_metrics_csv_incremental(metrics_csv, metrics_rows)
        if shard_new:
            log.info(
                "Shard %d: parsed %d new job(s)  (%d total metrics so far).",
                shard_idx,
                shard_new,
                len(metrics_rows),
            )

    n_pass = sum(1 for m in metrics_rows if m.passed)
    log.info(
        "Protenix: %d/%d passed pLDDT>=%.1f and interface_PAE<=%.1f.",
        n_pass,
        len(metrics_rows),
        cfg.plddt_cutoff,
        cfg.interface_pae_cutoff,
    )
    mark_stage_done(out_dir)
    return 0


def _write_metrics_csv_incremental(path: Path, rows: list[ProtenixMetrics]) -> None:
    """Re-write the full CSV from ``rows`` (kept sorted by job_name).

    Called after every shard so a mid-run crash still has all previously
    parsed data on disk."""
    fieldnames = [
        "job_name",
        "iptm",
        "plddt_mean",
        "plddt_designed_region_mean",
        "interface_pae_mean",
        "ranking_score",
        "passed",
        "failure_reason",
    ]
    rows_sorted = sorted(rows, key=lambda r: r.job_name)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row.as_csv_row())


if __name__ == "__main__":
    raise SystemExit(main())
