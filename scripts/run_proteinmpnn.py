#!/usr/bin/env python3
"""Entry Point 2 — Inverse folding via ProteinMPNN.

For every backbone PDB in ``--in_dir`` this script:

1. Calls ``helper_scripts/parse_multiple_chains.py`` to build
   ``parsed_pdbs.jsonl``.
2. Calls ``helper_scripts/make_fixed_positions_dict.py`` with the
   motif residue list from the YAML config (HMG-box for SOX2, ZF1-3 for
   KLF4) — this is the spec's "fixed-position masking" implemented via
   the upstream helper, NOT by patching the model.
3. Runs ``protein_mpnn_run.py`` at ``sampling_temp=0.1`` and
   ``num_seq_per_target=50`` (defaults from spec; overridable from YAML).
4. Rewrites the resulting ProteinMPNN FASTA into a per-backbone file
   under ``--out_dir`` with headers
   ``>{backbone_id}|seq{idx}|score=...|recovery=...``.

Under ``--dry-run`` the upstream binary is not called; we instead emit
schema-valid stub FASTAs so the rest of the pipeline can be exercised on
a Mac.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .common import (
    TargetConfig,
    ensure_dir,
    expand_motif_residues,
    get_logger,
    iter_files,
    mark_stage_done,
    write_stub_fasta,
)


def _run(
    cmd: list[str],
    log,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    progress_interval: int = 1,
) -> None:
    """Run ``cmd``, streaming stdout line-by-line at ``log.info``.
    
    ``progress_interval`` controls how often to emit a log line (1 = every
    line, useful for the parse/fix helpers; higher values = less noise,
    useful for the main MPNN run)."""
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    assert proc.stdout is not None
    line_count = 0
    suppressed = 0
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        if not stripped:
            continue
        line_count += 1
        if line_count % progress_interval == 0:
            if suppressed:
                log.info("  ... (%d lines omitted)", suppressed)
                suppressed = 0
            log.info("  [%d] %s", line_count, stripped)
        else:
            suppressed += 1
    if suppressed:
        log.info("  ... (%d lines omitted)", suppressed)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _parse_chains(
    *,
    proteinmpnn_dir: Path,
    backbones_dir: Path,
    work_dir: Path,
    log,
    env: dict[str, str] | None = None,
) -> Path:
    parsed = work_dir / "parsed_pdbs.jsonl"
    cmd = [
        sys.executable,
        str(proteinmpnn_dir / "helper_scripts" / "parse_multiple_chains.py"),
        f"--input_path={backbones_dir}",
        f"--output_path={parsed}",
    ]
    _run(cmd, log, env=env, progress_interval=1)
    return parsed


def _make_fixed_positions(
    *,
    proteinmpnn_dir: Path,
    parsed_jsonl: Path,
    work_dir: Path,
    chain: str,
    motif_residues: list[int],
    log,
    env: dict[str, str] | None = None,
) -> Path:
    fixed = work_dir / "fixed_positions.jsonl"
    position_list = " ".join(str(r) for r in motif_residues)
    cmd = [
        sys.executable,
        str(proteinmpnn_dir / "helper_scripts" / "make_fixed_positions_dict.py"),
        f"--input_path={parsed_jsonl}",
        f"--output_path={fixed}",
        f"--chain_list={chain}",
        f"--position_list={position_list}",
    ]
    _run(cmd, log, env=env, progress_interval=1)
    return fixed


def _run_proteinmpnn(
    *,
    proteinmpnn_dir: Path,
    parsed_jsonl: Path,
    fixed_jsonl: Path,
    out_dir: Path,
    num_seq_per_target: int,
    sampling_temp: float,
    batch_size: int,
    log,
    env: dict[str, str] | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(proteinmpnn_dir / "protein_mpnn_run.py"),
        f"--jsonl_path={parsed_jsonl}",
        f"--fixed_positions_jsonl={fixed_jsonl}",
        f"--out_folder={out_dir}",
        f"--num_seq_per_target={num_seq_per_target}",
        f"--sampling_temp={sampling_temp}",
        f"--batch_size={batch_size}",
    ]
    _run(cmd, log, env=env, progress_interval=50)


def _rewrite_fasta(
    *,
    raw_fasta: Path,
    backbone_id: str,
    out_path: Path,
) -> None:
    """ProteinMPNN emits one FASTA per input containing all sampled
    sequences. We rewrite the headers so they include the backbone id and
    a stable seq index downstream stages can key on."""
    blocks: list[str] = []
    header: str | None = None
    seq_lines: list[str] = []
    seq_idx = -1  # the very first record is the WT, which MPNN labels T=0

    def _flush() -> None:
        nonlocal seq_idx, header, seq_lines
        if header is None:
            return
        seq = "".join(seq_lines)
        if seq_idx < 0:
            tag = "wt"
        else:
            tag = f"seq{seq_idx:03d}"
        # Try to lift score / recovery from the upstream header.
        score = ""
        recovery = ""
        for part in header.lstrip(">").split(","):
            part = part.strip()
            if part.startswith("score="):
                score = part.split("=", 1)[1]
            elif part.startswith("seq_recovery="):
                recovery = part.split("=", 1)[1]
        blocks.append(
            f">{backbone_id}|{tag}|score={score}|recovery={recovery}\n{seq}"
        )
        header, seq_lines = None, []

    with open(raw_fasta) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                _flush()
                header = line
                seq_idx += 1
                seq_lines = []
            else:
                seq_lines.append(line)
        _flush()
    out_path.write_text("\n".join(blocks) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ProteinMPNN inverse-folding wrapper.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--in_dir", required=True, type=Path, help="Backbone PDB dir from step 1.")
    p.add_argument("--out_dir", required=True, type=Path, help="Per-backbone FASTA output dir.")
    p.add_argument(
        "--proteinmpnn_dir",
        type=Path,
        default=Path("/opt/ProteinMPNN"),
        help="Path to the upstream ProteinMPNN checkout.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="ProteinMPNN batch size (reduce if OOM).",
    )
    p.add_argument(
        "--num_seq_per_target",
        type=int,
        default=None,
        help="Override config.mpnn_seqs_per_backbone.",
    )
    p.add_argument(
        "--sampling_temp",
        type=float,
        default=None,
        help="Override config.mpnn_temp.",
    )
    p.add_argument(
        "--backend",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help=(
            "Backend hint for ProteinMPNN subprocess environment. "
            "cuda=default GPU path; mps/cpu disable CUDA and enable MPS fallback."
        ),
    )
    p.add_argument(
        "--cuda_devices",
        type=str,
        default="all",
        help="CUDA device selector when backend is cuda/auto (e.g. '0', '0,1', 'all').",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--work_dir",
        type=Path,
        default=None,
        help="Persistent intermediate work directory. Defaults to <out_dir>/_mpnn_work.",
    )
    args = p.parse_args(argv)

    cfg = TargetConfig.from_yaml(args.config)
    out_dir = ensure_dir(args.out_dir)
    log = get_logger("proteinmpnn", log_dir=out_dir.parent / "logs")
    work_base = args.work_dir or (out_dir / "_mpnn_work")

    num_seq = args.num_seq_per_target if args.num_seq_per_target is not None else cfg.mpnn_seqs_per_backbone
    temp = args.sampling_temp if args.sampling_temp is not None else cfg.mpnn_temp

    backbones = list(iter_files(args.in_dir, ".pdb"))
    if not backbones:
        log.error("No backbone .pdb files found in %s", args.in_dir)
        return 2

    # --- Auto-resume: move any already-done backbone PDBs into handled/ ---
    handled_dir = ensure_dir(args.in_dir / "handled")
    already_done: list[Path] = []
    for bb in backbones:
        if (out_dir / f"{bb.stem}.fasta").exists():
            already_done.append(bb)
    if already_done:
        for bb in already_done:
            shutil.move(str(bb), str(handled_dir / bb.name))
        log.info(
            "Auto-resume: moved %d already-done backbone(s) to %s (%d remaining).",
            len(already_done),
            handled_dir,
            len(backbones) - len(already_done),
        )
        backbones = [bb for bb in backbones if bb not in already_done]
        if not backbones:
            log.info("All backbones already processed — nothing to do.")
            mark_stage_done(out_dir)
            return 0

    log.info("Found %d backbone(s) to process. Designing %d sequences each at T=%g.", len(backbones), num_seq, temp)

    motif_residues = expand_motif_residues(cfg.motif)
    mpnn_env = os.environ.copy()
    if args.backend in ("mps", "cpu"):
        mpnn_env["CUDA_VISIBLE_DEVICES"] = ""
        mpnn_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    elif args.backend in ("auto", "cuda") and args.cuda_devices != "all":
        mpnn_env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Report the device ProteinMPNN will use.
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                log.info(
                    "CUDA device %d: %s (memory %.1f GB free / %.1f GB total)",
                    i,
                    torch.cuda.get_device_name(i),
                    (torch.cuda.get_device_properties(i).total_mem - torch.cuda.memory_allocated(i)) / 1e9,
                    torch.cuda.get_device_properties(i).total_mem / 1e9,
                )
        else:
            log.warning("CUDA not detected — ProteinMPNN may run on CPU (slow).")
    except ImportError:
        log.warning("torch not available in this process; cannot verify CUDA device. "
                     "ProteinMPNN subprocess will use its own PyTorch.")

    if args.dry_run:
        log.warning("--dry-run: skipping ProteinMPNN; writing stub FASTAs.")
        for bb in backbones:
            backbone_id = bb.stem
            out_path = out_dir / f"{backbone_id}.fasta"
            write_stub_fasta(
                out_path,
                backbone_id=backbone_id,
                num_sequences=num_seq,
                sequence_length=cfg.protein_length,
            )
        mark_stage_done(out_dir)
        log.info("Dry-run: wrote %d stub FASTA(s) to %s", len(backbones), out_dir)
        return 0

    if not args.proteinmpnn_dir.exists():
        log.error(
            "ProteinMPNN not found at %s. Install ProteinMPNN or pass "
            "--proteinmpnn_dir. (Use --dry-run for a Mac sanity check.)",
            args.proteinmpnn_dir,
        )
        return 2

    # ProteinMPNN's helpers expect a directory of PDBs, so we run the
    # whole batch in one shot, then split the output FASTAs per backbone.
    # Using a persistent work directory so crashes don't lose progress.
    work = ensure_dir(work_base)
    raw_out = work / "mpnn_out"
    raw_out.mkdir(parents=True, exist_ok=True)

    # Step 1: parse all PDB chains (cached — skip if already done).
    parsed = work / "parsed_pdbs.jsonl"
    if parsed.exists():
        log.info("Reusing cached %s (delete to force re-parse).", parsed)
    else:
        _parse_chains(
            proteinmpnn_dir=args.proteinmpnn_dir,
            backbones_dir=args.in_dir,
            work_dir=work,
            log=log,
            env=mpnn_env,
        )

    # Step 2: fixed-positions dict (cached — skip if already done).
    fixed = work / "fixed_positions.jsonl"
    if fixed.exists():
        log.info("Reusing cached %s (delete to force regenerate).", fixed)
    else:
        _make_fixed_positions(
            proteinmpnn_dir=args.proteinmpnn_dir,
            parsed_jsonl=parsed,
            work_dir=work,
            chain=cfg.motif.chain,
            motif_residues=motif_residues,
            log=log,
            env=mpnn_env,
        )

    # Step 3: run ProteinMPNN (re-runnable; skips backbones that already
    #         have .fa files in the seqs/ directory when resuming).
    seqs_dir = raw_out / "seqs"
    need_mpnn = not seqs_dir.exists()
    if not need_mpnn:
        # Check if every backbone we care about has a .fa already.
        for bb in backbones:
            if not (seqs_dir / f"{bb.stem}.fa").exists():
                need_mpnn = True
                break
    if need_mpnn:
        _run_proteinmpnn(
            proteinmpnn_dir=args.proteinmpnn_dir,
            parsed_jsonl=parsed,
            fixed_jsonl=fixed,
            out_dir=raw_out,
            num_seq_per_target=num_seq,
            sampling_temp=temp,
            batch_size=args.batch_size,
            log=log,
            env=mpnn_env,
        )
    else:
        log.info("All %d backbones already have MPNN output — skipping inference.", len(backbones))

    # Step 4: rewrite FASTA headers per backbone, then move source PDB
    #         into handled/ so it is automatically skipped on re-runs.
    if not seqs_dir.exists():
        log.error("ProteinMPNN did not produce a seqs/ directory in %s", raw_out)
        return 3
    n_written = 0
    for bb in backbones:
        raw_fa = seqs_dir / f"{bb.stem}.fa"
        if not raw_fa.exists():
            log.warning("No ProteinMPNN output for backbone %s", bb.stem)
            continue
        dest = out_dir / f"{bb.stem}.fasta"
        if dest.exists():
            n_written += 1
            # Still move the PDB to handled/ even if FASTA already existed
            # (covers edge cases from interrupted previous runs).
            if bb.exists():
                shutil.move(str(bb), str(handled_dir / bb.name))
            continue
        _rewrite_fasta(
            raw_fasta=raw_fa,
            backbone_id=bb.stem,
            out_path=dest,
        )
        n_written += 1
        # Move the source PDB to handled/ now that the FASTA is safely on disk.
        shutil.move(str(bb), str(handled_dir / bb.name))
    log.info("Wrote %d per-backbone FASTA(s) to %s", n_written, out_dir)

    mark_stage_done(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
