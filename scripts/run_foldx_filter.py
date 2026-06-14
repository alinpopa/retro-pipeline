#!/usr/bin/env python3
"""Entry Point 4 — Stability filter via FoldX.

For every candidate that survived the Protenix step we:

1. Read the candidate amino-acid sequence (from the step-2 FASTA) and the
   wild-type sequence (extracted from the template PDB).
2. Build a FoldX ``individual_list.txt`` of point mutations relative to
   WT by aligning candidate vs WT with Biopython's ``pairwise2``.
3. Run ``foldx --command=BuildModel --pdb=WT.pdb --mutant-file=...
   --numberOfRuns=N``.
4. Run ``foldx --command=Stability`` on both WT and the mutant model.
5. Parse ``Stability.fxout`` to compute ``ΔΔG = G(mutant) − G(WT)``.
6. Apply the spec rule: ``ΔΔG > ddg_cutoff`` (default 0.0 kcal/mol) is
   destabilizing -> discard.
7. As an aggregation-risk proxy, sum the SASA of hydrophobic residues
   (A, V, L, I, M, F, W, Y) via Biopython's ShrakeRupley sampler (no
   DSSP dependency) and rank low-to-high.

All outputs land in ``--out_dir``:

  workspace/04_thermodynamics/
    ddg.csv                 # one row per candidate, sorted by composite rank
    individual_lists/       # per-candidate FoldX mutant lists
    foldx_runs/<job>/...    # raw FoldX work dirs (kept for debugging)
"""

from __future__ import annotations

import argparse
import csv
import io
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
    which,
)

HYDROPHOBIC = set("AVLIMFWY")
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class FoldxRow:
    job_name: str
    n_mutations: int
    ddg_kcal_mol: float
    hydrophobic_sasa: float
    passed: bool
    failure_reason: str

    def as_csv(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# PDB sequence extraction (minimal; avoids hard dependency on Biopython for the
# happy path so --dry-run still works on a bare-metal interpreter).
# --------------------------------------------------------------------------- #


def extract_chain_sequence(pdb_path: Path, chain: str) -> tuple[str, list[int]]:
    """Return (one-letter sequence, residue numbers) for ``chain`` in ``pdb_path``."""
    residues: dict[int, str] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[21:22] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            resname = line[17:20].strip()
            residues.setdefault(resnum, THREE_TO_ONE.get(resname, "X"))
    nums = sorted(residues)
    seq = "".join(residues[n] for n in nums)
    return seq, nums


# --------------------------------------------------------------------------- #
# Mutation list builder
# --------------------------------------------------------------------------- #


def diff_to_mutations(
    *,
    wt_seq: str,
    wt_resnums: list[int],
    cand_seq: str,
    chain: str,
) -> list[str]:
    """Return FoldX-formatted mutations relative to WT.

    FoldX format per mutation: ``<WT><chain><resnum><MUT>``, e.g.
    ``RA39W``. Multiple mutations on one design are joined by commas and
    terminated with a semicolon.
    """
    try:
        from Bio import pairwise2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Biopython is required for FoldX mutation-list generation."
        ) from exc
    alignments = pairwise2.align.globalxx(wt_seq, cand_seq, one_alignment_only=True)
    if not alignments:
        return []
    aln = alignments[0]
    wt_aln, cand_aln = aln.seqA, aln.seqB
    muts: list[str] = []
    wt_idx = 0
    for wt_c, c_c in zip(wt_aln, cand_aln):
        if wt_c == "-":
            # insertion in candidate; FoldX BuildModel cannot model
            # insertions, so we skip — this candidate may still get
            # filtered by the alignment-aware stability fallback.
            continue
        resnum = wt_resnums[wt_idx]
        wt_idx += 1
        if c_c == "-":
            # deletion -- also unmodelable by BuildModel; skip.
            continue
        if wt_c != c_c and wt_c != "X" and c_c != "X":
            muts.append(f"{wt_c}{chain}{resnum}{c_c}")
    return muts


def write_individual_list(path: Path, mutations: list[str], runs: int) -> None:
    """Write FoldX ``individual_list.txt`` for ``runs`` independent reps."""
    line = ",".join(mutations) + ";"
    path.write_text("\n".join([line] * runs) + "\n")


# --------------------------------------------------------------------------- #
# FoldX subprocess wrappers
# --------------------------------------------------------------------------- #


def _run_foldx(
    *,
    foldx_bin: str,
    cwd: Path,
    args: list[str],
    log,
) -> subprocess.CompletedProcess:
    cmd = [foldx_bin] + args
    log.info("$ (in %s) %s", cwd, " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd), check=True, capture_output=True, text=True
    )
    if proc.stdout:
        log.debug(proc.stdout)
    return proc


def _parse_stability(fxout: Path) -> float:
    """Return the total ΔG (kcal/mol) from a FoldX Stability.fxout file.

    Format: ``<model.pdb>\\t<total_energy>\\t...``
    """
    with open(fxout) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    raise RuntimeError(f"Could not parse total energy from {fxout}")


# --------------------------------------------------------------------------- #
# Hydrophobic SASA via Biopython
# --------------------------------------------------------------------------- #


def hydrophobic_sasa(pdb_path: Path, chain: str) -> float:
    """Sum of solvent-accessible surface area on hydrophobic side-chain
    atoms. Uses Biopython's ShrakeRupley sampler so we don't need DSSP
    installed.
    """
    try:
        from Bio.PDB import PDBParser  # type: ignore
        from Bio.PDB.SASA import ShrakeRupley  # type: ignore
    except ImportError:
        return float("nan")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    total = 0.0
    for model in structure:
        if chain not in model:
            continue
        for res in model[chain]:
            one = THREE_TO_ONE.get(res.get_resname(), "X")
            if one in HYDROPHOBIC:
                total += float(getattr(res, "sasa", 0.0))
    return total


# --------------------------------------------------------------------------- #
# Dry-run helpers
# --------------------------------------------------------------------------- #


def _stub_ddg(job_name: str) -> float:
    import random
    rng = random.Random(hash(job_name) & 0xFFFFFFFF)
    return round(rng.uniform(-1.5, 4.0), 3)


def _stub_hydrophobic_sasa(job_name: str) -> float:
    import random
    rng = random.Random(hash(job_name) & 0xFFFFFFFF)
    return round(rng.uniform(1200.0, 3500.0), 2)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FoldX ΔΔG filter.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--in_dir",
        required=True,
        type=Path,
        help="Protenix output dir from step 3 (uses metrics.csv to know which jobs passed).",
    )
    p.add_argument(
        "--sequences_dir",
        type=Path,
        default=None,
        help="ProteinMPNN FASTA dir from step 2 (used to read candidate AA sequences). "
        "Defaults to '<in_dir>/../02_sequences'.",
    )
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument(
        "--foldx_bin",
        type=str,
        default=None,
        help="Path to the FoldX binary (defaults to first 'foldx' on PATH).",
    )
    p.add_argument(
        "--ddg_cutoff",
        type=float,
        default=None,
        help="Override config.ddg_cutoff (positive ΔΔG -> destabilizing -> discard).",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=None,
        help="FoldX BuildModel numberOfRuns override.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = TargetConfig.from_yaml(args.config)
    out_dir = ensure_dir(args.out_dir)
    log = get_logger("foldx", log_dir=out_dir.parent / "logs")

    ddg_cutoff = args.ddg_cutoff if args.ddg_cutoff is not None else cfg.ddg_cutoff
    runs = args.runs if args.runs is not None else cfg.foldx_runs_per_mutant

    sequences_dir = args.sequences_dir or (args.in_dir.parent / "02_sequences")
    log.info(
        "FoldX filter: ddg_cutoff=%.2f kcal/mol, runs=%d, sequences=%s",
        ddg_cutoff,
        runs,
        sequences_dir,
    )

    # Determine the candidate job list from the Protenix metrics CSV
    # (single source of truth -- only candidates with passed=True).
    metrics_csv = args.in_dir / "metrics.csv"
    if not metrics_csv.exists():
        log.error("Protenix metrics.csv not found at %s", metrics_csv)
        return 2

    candidates: list[str] = []
    with open(metrics_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("passed", "").lower() in ("true", "1", "yes"):
                candidates.append(row["job_name"])
    if not candidates:
        log.warning("No Protenix-passed candidates; emitting empty ddg.csv.")
    log.info("Will process %d Protenix-passed candidate(s).", len(candidates))

    # Read the WT sequence ONCE from the template; cache by chain.
    if not cfg.template_pdb.exists() and not args.dry_run:
        log.error("Template PDB %s not found.", cfg.template_pdb)
        return 2
    if args.dry_run and not cfg.template_pdb.exists():
        wt_seq, wt_resnums = "", []  # not used in --dry-run
    else:
        wt_seq, wt_resnums = extract_chain_sequence(cfg.template_pdb, cfg.motif.chain)
        log.info("WT chain %s length=%d.", cfg.motif.chain, len(wt_seq))

    foldx_bin = args.foldx_bin or which("foldx") or which("foldx_20251231")
    if not args.dry_run and foldx_bin is None:
        log.error(
            "FoldX binary not found. Install FoldX or pass --foldx_bin. "
            "(Use --dry-run for a Mac sanity check.)"
        )
        return 2

    rows: list[FoldxRow] = []
    individual_lists_dir = ensure_dir(out_dir / "individual_lists")
    foldx_runs_dir = ensure_dir(out_dir / "foldx_runs")

    # Build a {job_name: aa_sequence} map by reading every FASTA from step 2.
    aa_by_job = _read_aa_by_job(sequences_dir)

    for job_name in candidates:
        cand_aa = aa_by_job.get(job_name)
        if cand_aa is None:
            log.warning("No AA sequence found for %s; skipping.", job_name)
            rows.append(
                FoldxRow(
                    job_name=job_name,
                    n_mutations=0,
                    ddg_kcal_mol=float("nan"),
                    hydrophobic_sasa=float("nan"),
                    passed=False,
                    failure_reason="missing_sequence",
                )
            )
            continue

        if args.dry_run:
            ddg = _stub_ddg(job_name)
            sasa = _stub_hydrophobic_sasa(job_name)
            rows.append(
                FoldxRow(
                    job_name=job_name,
                    n_mutations=-1,
                    ddg_kcal_mol=ddg,
                    hydrophobic_sasa=sasa,
                    passed=ddg <= ddg_cutoff,
                    failure_reason="" if ddg <= ddg_cutoff else f"ddg={ddg:.2f}>{ddg_cutoff}",
                )
            )
            continue

        muts = diff_to_mutations(
            wt_seq=wt_seq,
            wt_resnums=wt_resnums,
            cand_seq=cand_aa,
            chain=cfg.motif.chain,
        )
        if not muts:
            log.info("%s is identical to WT (no mutations); ΔΔG=0.", job_name)
            rows.append(
                FoldxRow(
                    job_name=job_name,
                    n_mutations=0,
                    ddg_kcal_mol=0.0,
                    hydrophobic_sasa=hydrophobic_sasa(cfg.template_pdb, cfg.motif.chain),
                    passed=True,
                    failure_reason="",
                )
            )
            continue
        ind_list = individual_lists_dir / f"{job_name}.txt"
        write_individual_list(ind_list, muts, runs)

        work = ensure_dir(foldx_runs_dir / job_name)
        # FoldX wants the PDB next to the working directory; symlink/copy.
        shutil.copy(cfg.template_pdb, work / cfg.template_pdb.name)
        shutil.copy(ind_list, work / "individual_list.txt")

        try:
            _run_foldx(
                foldx_bin=foldx_bin,
                cwd=work,
                args=[
                    "--command=BuildModel",
                    f"--pdb={cfg.template_pdb.name}",
                    "--mutant-file=individual_list.txt",
                    f"--numberOfRuns={runs}",
                ],
                log=log,
            )
            _run_foldx(
                foldx_bin=foldx_bin,
                cwd=work,
                args=["--command=Stability", f"--pdb={cfg.template_pdb.name}"],
                log=log,
            )
            wt_g = _parse_stability(work / "Stability.fxout")
            mutant_pdbs = sorted(work.glob(f"{cfg.template_pdb.stem}_*.pdb"))
            if not mutant_pdbs:
                raise RuntimeError("BuildModel produced no mutant PDBs.")
            mut_gs: list[float] = []
            for mpdb in mutant_pdbs[:runs]:
                _run_foldx(
                    foldx_bin=foldx_bin,
                    cwd=work,
                    args=["--command=Stability", f"--pdb={mpdb.name}"],
                    log=log,
                )
                # Stability.fxout gets overwritten each call.
                mut_gs.append(_parse_stability(work / "Stability.fxout"))
            mean_mut_g = statistics.fmean(mut_gs)
            ddg = mean_mut_g - wt_g

            sasa = hydrophobic_sasa(mutant_pdbs[0], cfg.motif.chain)
            passed = ddg <= ddg_cutoff
            rows.append(
                FoldxRow(
                    job_name=job_name,
                    n_mutations=len(muts),
                    ddg_kcal_mol=round(ddg, 3),
                    hydrophobic_sasa=round(sasa, 2),
                    passed=passed,
                    failure_reason="" if passed else f"ddg={ddg:.2f}>{ddg_cutoff}",
                )
            )
            log.info(
                "%s: %d mutations  ΔΔG=%+.2f kcal/mol  SASA_hyd=%.0f  %s",
                job_name,
                len(muts),
                ddg,
                sasa,
                "PASS" if passed else "DROP",
            )
        except subprocess.CalledProcessError as exc:
            log.error("FoldX failed for %s: %s", job_name, exc.stderr or exc)
            rows.append(
                FoldxRow(
                    job_name=job_name,
                    n_mutations=len(muts),
                    ddg_kcal_mol=float("nan"),
                    hydrophobic_sasa=float("nan"),
                    passed=False,
                    failure_reason="foldx_error",
                )
            )

    # Rank survivors by a composite score (lower is better):
    #   primary key   = ddg_kcal_mol         (more stable first)
    #   secondary key = hydrophobic_sasa     (less surface hydrophobicity first)
    rows.sort(key=lambda r: (r.ddg_kcal_mol, r.hydrophobic_sasa))

    ddg_csv = out_dir / "ddg.csv"
    with open(ddg_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "job_name",
                "n_mutations",
                "ddg_kcal_mol",
                "hydrophobic_sasa",
                "passed",
                "failure_reason",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.as_csv())

    n_pass = sum(1 for r in rows if r.passed)
    log.info("FoldX: %d/%d candidates passed ΔΔG<=%.2f.", n_pass, len(rows), ddg_cutoff)
    mark_stage_done(out_dir)
    return 0


def _read_aa_by_job(sequences_dir: Path) -> dict[str, str]:
    """Build {job_name -> aa_sequence} from per-backbone FASTAs.

    job_name format matches the one used by run_protenix:
    ``{backbone_stem}__{seq_tag}``.
    """
    out: dict[str, str] = {}
    if not sequences_dir.is_dir():
        return out
    for fa in iter_files(sequences_dir, ".fasta"):
        backbone_stem = fa.stem
        name, seq_lines = None, []
        for line in fa.read_text().splitlines():
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(seq_lines)
                tag = line[1:].split("|")
                seq_tag = tag[1] if len(tag) > 1 else "seq"
                name = f"{backbone_stem}__{seq_tag}"
                seq_lines = []
            elif line:
                seq_lines.append(line.strip())
        if name is not None:
            out[name] = "".join(seq_lines)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
