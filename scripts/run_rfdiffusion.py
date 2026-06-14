#!/usr/bin/env python3
"""Entry Point 1 — Partial Diffusion via RFdiffusion.

Generates ``num_backbones`` structural backbones where the DNA-binding
motif (HMG-box for SOX2, ZF1-3 for KLF4) is physically locked in 3D and
the rest of the protein is randomized via partial diffusion.

This is a *wrapper* around the upstream
``RosettaCommons/RFdiffusion/scripts/run_inference.py`` Hydra entry point.
We do not reimplement RFdiffusion: we assemble its CLI from the
``configs/<target>.yaml`` so the user never hand-crafts a contig string,
and we wrap the subprocess in our CUDA-OOM retry loop.

Reference invocation (partial diffusion, motif-locked):

    python run_inference.py \\
        inference.input_pdb=templates/6T7B.pdb \\
        'contigmap.contigs=[40-40/A41-117/200-200/0 B1-147]' \\
        diffuser.partial_T=40 \\
        inference.num_designs=10000 \\
        inference.output_prefix=workspace/01_backbones/sox2

Under ``--dry-run`` we skip the subprocess entirely and emit placeholder
CA-only PDB stubs so downstream stages can be exercised on a Mac.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .common import (
    TargetConfig,
    contig_from_motif,
    count_residues_in_pdb,
    ensure_dir,
    get_logger,
    mark_stage_done,
    run_with_oom_retry,
    write_stub_backbone_pdb,
)


def _build_command(
    *,
    python_exe: str,
    rfdiffusion_script: Path,
    input_pdb: Path,
    contig: str,
    partial_T: int,
    num_designs: int,
    output_prefix: Path,
) -> list[str]:
    return [
        python_exe,
        str(rfdiffusion_script),
        f"inference.input_pdb={input_pdb}",
        f"contigmap.contigs={contig}",
        f"diffuser.partial_T={partial_T}",
        f"inference.num_designs={num_designs}",
        f"inference.output_prefix={output_prefix}",
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RFdiffusion partial-diffusion wrapper.")
    p.add_argument("--config", required=True, type=Path, help="configs/<target>.yaml")
    p.add_argument(
        "--input_pdb",
        type=Path,
        default=None,
        help="Override the template PDB from the config (defaults to config.template_pdb).",
    )
    p.add_argument(
        "--partial_T",
        type=int,
        default=None,
        help="Override config.partial_T (allowed range: 0-50; spec target 35-45).",
    )
    p.add_argument(
        "--num_designs",
        type=int,
        default=None,
        help="Override config.num_backbones.",
    )
    p.add_argument(
        "--contigs",
        type=str,
        default=None,
        help="Override the auto-built contig string (advanced).",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Directory to write backbone .pdb files into.",
    )
    p.add_argument(
        "--rfdiffusion_script",
        type=Path,
        default=None,
        help=(
            "Path to RFdiffusion run_inference.py. Defaults to the upstream CUDA "
            "checkout (~/Code/RFdiffusion/...) for cuda/auto backends, and to the "
            "Apple-Metal fork checkout (~/Code/RFdiffusion-mps/...) for mps/cpu."
        ),
    )
    p.add_argument(
        "--metal_rfdiffusion_script",
        type=Path,
        default=Path.home() / "Code/RFdiffusion-mps/scripts/run_inference.py",
        help=(
            "Path to the Apple-Metal RFdiffusion fork's run_inference.py "
            "(YaoYinYing/RFdiffusion@mps-test). Used when --backend is mps/cpu "
            "unless --rfdiffusion_script is given explicitly. "
            "Default: ~/Code/RFdiffusion-mps/scripts/run_inference.py."
        ),
    )
    p.add_argument(
        "--rfdiffusion_python",
        type=str,
        default=None,
        help=(
            "Python interpreter used to launch run_inference.py. RFdiffusion "
            "(esp. the Apple-Metal fork) lives in its OWN conda env, separate "
            "from this pipeline's venv, so point this at that env's python, e.g. "
            "~/miniconda3/envs/RFdiffusion/bin/python. Defaults to the current "
            "interpreter (only correct if the pipeline runs inside that env)."
        ),
    )
    p.add_argument(
        "--backend",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help=(
            "Execution backend. cuda/auto use the upstream CUDA RFdiffusion. "
            "mps runs real diffusion on Apple Silicon via the MPS-enabled fork "
            "(YaoYinYing/RFdiffusion@mps-test); cpu forces the same fork onto CPU."
        ),
    )
    p.add_argument(
        "--cuda_devices",
        type=str,
        default="all",
        help="CUDA device selector for real runs (e.g. '0', '0,1', or 'all').",
    )
    p.add_argument(
        "--dna_chain_length",
        type=int,
        default=None,
        help="If the template includes DNA, length of chain B to keep locked in the contig.",
    )
    p.add_argument("--dry-run", action="store_true", help="Emit stub PDBs; do not call RFdiffusion.")
    args = p.parse_args(argv)

    cfg = TargetConfig.from_yaml(args.config)
    out_dir = ensure_dir(args.out_dir)
    log = get_logger("rfdiffusion", log_dir=out_dir.parent / "logs")

    input_pdb = (args.input_pdb or cfg.template_pdb).resolve()
    partial_T = args.partial_T if args.partial_T is not None else cfg.partial_T
    num_designs = args.num_designs if args.num_designs is not None else cfg.num_backbones

    # Auto-build the contig from the motif spec unless user overrode it.
    if args.contigs:
        contig = args.contigs
    else:
        dna_len = args.dna_chain_length
        contig = contig_from_motif(
            protein_length=cfg.protein_length,
            motif=cfg.motif,
            dna_length=dna_len,
        )
    log.info("Target=%s  partial_T=%d  num_designs=%d  contig=%s", cfg.name, partial_T, num_designs, contig)

    output_prefix = out_dir / cfg.name

    if args.dry_run:
        log.warning("--dry-run: skipping RFdiffusion subprocess; writing stub backbones.")
        # Decide stub backbone length: prefer counting the real template if
        # available, else fall back to the config-declared protein length.
        try:
            stub_len = count_residues_in_pdb(input_pdb, cfg.protein_chain) or cfg.protein_length
        except FileNotFoundError:
            stub_len = cfg.protein_length
        for i in range(num_designs):
            stub_path = out_dir / f"{cfg.name}_{i:05d}.pdb"
            write_stub_backbone_pdb(
                stub_path,
                num_residues=stub_len,
                seed_key=f"{cfg.name}|backbone|{i}",
            )
        mark_stage_done(out_dir)
        log.info("Dry-run: wrote %d stub backbone(s) to %s", num_designs, out_dir)
        return 0

    # Pick the right RFdiffusion checkout for the backend. The Apple-Metal
    # fork (YaoYinYing/RFdiffusion@mps-test) exposes the *identical*
    # run_inference.py Hydra entry point and flags, so it is a drop-in for the
    # normal flow — only the device selection and the install differ.
    is_metal = args.backend in ("mps", "cpu")
    if args.rfdiffusion_script is not None:
        rfdiffusion_script = args.rfdiffusion_script
    elif is_metal:
        rfdiffusion_script = args.metal_rfdiffusion_script
    else:
        rfdiffusion_script = Path.home() / "Code/RFdiffusion/scripts/run_inference.py"

    if not rfdiffusion_script.exists():
        if is_metal:
            log.error(
                "Apple-Metal RFdiffusion fork not found at %s. Install the "
                "MPS-enabled fork (see docs/pathways/apple-metal/README.md) or "
                "pass --rfdiffusion_script. (Use --dry-run for a no-install check.)",
                rfdiffusion_script,
            )
        else:
            log.error(
                "RFdiffusion script not found at %s. Install RFdiffusion or pass "
                "--rfdiffusion_script. (Use --dry-run for a Mac sanity check.)",
                rfdiffusion_script,
            )
        return 2
    if not input_pdb.exists():
        log.error("Template PDB %s not found.", input_pdb)
        return 2

    python_exe = args.rfdiffusion_python or sys.executable
    if args.rfdiffusion_python is not None and not Path(python_exe).exists():
        log.error(
            "--rfdiffusion_python %s does not exist. Point it at the RFdiffusion "
            "env interpreter, e.g. ~/miniconda3/envs/RFdiffusion/bin/python.",
            python_exe,
        )
        return 2
    if args.rfdiffusion_python is None:
        log.warning(
            "Launching RFdiffusion with the current interpreter (%s). If the fork "
            "lives in a separate conda env, pass --rfdiffusion_python "
            "(e.g. ~/miniconda3/envs/RFdiffusion/bin/python) or you'll hit "
            "ModuleNotFoundError (omegaconf/torch/...).",
            python_exe,
        )

    cmd = _build_command(
        python_exe=python_exe,
        rfdiffusion_script=rfdiffusion_script,
        input_pdb=input_pdb,
        contig=contig,
        partial_T=partial_T,
        num_designs=num_designs,
        output_prefix=output_prefix,
    )

    env = os.environ.copy()
    if is_metal:
        # The fork auto-selects the torch device (cuda -> mps -> cpu). Hide any
        # CUDA device so it falls through to Metal, and enable the CPU fallback
        # for the handful of ops MPS does not yet implement.
        env["CUDA_VISIBLE_DEVICES"] = "" if args.backend == "cpu" else "-1"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if args.backend == "cpu":
            # Belt-and-suspenders: some torch builds still probe MPS even with
            # no CUDA visible; this nudges the fork onto CPU explicitly.
            env["PYTORCH_MPS_DISABLE"] = "1"
        log.info(
            "Apple-Metal backend=%s using fork %s (PYTORCH_ENABLE_MPS_FALLBACK=1).",
            args.backend,
            rfdiffusion_script,
        )
    elif args.backend in ("cuda", "auto"):
        if args.cuda_devices != "all":
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    proc = run_with_oom_retry(
        cmd,
        downscale_arg="inference.num_designs=",
        initial_value=num_designs,
        factor=0.5,
        max_retries=3,
        min_value=1,
        logger=log,
        env=env,
    )
    if proc.stdout:
        log.info("RFdiffusion stdout (tail):\n%s", proc.stdout[-2000:])

    # RFdiffusion writes <prefix>_<n>.pdb; normalize to <name>_<n>.pdb in
    # out_dir (already the case because output_prefix uses out_dir/name).
    generated = sorted(out_dir.glob(f"{cfg.name}_*.pdb"))
    log.info("RFdiffusion produced %d backbone PDB(s).", len(generated))
    mark_stage_done(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
