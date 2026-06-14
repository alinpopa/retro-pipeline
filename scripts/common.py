"""Shared utilities for the retro_pipeline scripts.

This module is intentionally dependency-light so that `--dry-run` paths
work even on a developer Mac with only the standard library + PyYAML +
Biopython installed.

Provides:
- structured logging (console + per-stage rotating file under workspace/logs/)
- ``run_with_oom_retry`` for shelling out to CUDA-bound subprocesses with
  graceful array downscaling on `CUDA out of memory` errors
- DNA helpers (reverse complement, palindrome validation)
- PDB residue-range helpers (length inference, motif residue list expansion)
- Dry-run stub writers that emit schema-valid placeholders for each stage
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Return a configured logger.

    Console handler is always installed; if ``log_dir`` is supplied a
    rotating file handler (5 MB x 3) is attached as well so each stage
    leaves an audit trail in ``workspace/logs/<stage>.log``.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(_LOG_FMT)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MotifSpec:
    chain: str
    ranges: tuple[tuple[int, int], ...]

    def residue_indices(self) -> list[int]:
        out: list[int] = []
        for start, end in self.ranges:
            out.extend(range(start, end + 1))
        return out

    def total_length(self) -> int:
        return sum(end - start + 1 for start, end in self.ranges)


@dataclass(frozen=True)
class TargetConfig:
    """Strongly-typed view of a configs/<target>.yaml file."""

    name: str
    template_pdb: Path
    protein_chain: str
    protein_length: int
    motif: MotifSpec
    dna_target: str
    partial_T: int
    num_backbones: int
    mpnn_seqs_per_backbone: int
    mpnn_temp: float
    plddt_cutoff: float
    interface_pae_cutoff: float
    iptm_warn_threshold: float
    ddg_cutoff: float
    foldx_runs_per_mutant: int
    top_n: int

    @classmethod
    def from_yaml(cls, path: Path) -> "TargetConfig":
        data = yaml.safe_load(Path(path).read_text())
        motif = MotifSpec(
            chain=data["motif"]["chain"],
            ranges=tuple(tuple(r) for r in data["motif"]["ranges"]),
        )
        return cls(
            name=data["name"],
            template_pdb=Path(data["template_pdb"]),
            protein_chain=data["protein_chain"],
            protein_length=int(data["protein_length"]),
            motif=motif,
            dna_target=data["dna_target"].upper(),
            partial_T=int(data["partial_T"]),
            num_backbones=int(data["num_backbones"]),
            mpnn_seqs_per_backbone=int(data["mpnn_seqs_per_backbone"]),
            mpnn_temp=float(data["mpnn_temp"]),
            plddt_cutoff=float(data["plddt_cutoff"]),
            interface_pae_cutoff=float(data["interface_pae_cutoff"]),
            iptm_warn_threshold=float(data["iptm_warn_threshold"]),
            ddg_cutoff=float(data["ddg_cutoff"]),
            foldx_runs_per_mutant=int(data["foldx_runs_per_mutant"]),
            top_n=int(data["top_n"]),
        )


# --------------------------------------------------------------------------- #
# DNA helpers
# --------------------------------------------------------------------------- #

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Return the Watson-Crick reverse complement of ``seq``."""
    return seq.translate(_COMPLEMENT)[::-1]


def is_palindrome(seq: str) -> bool:
    return seq.upper() == reverse_complement(seq.upper())


# --------------------------------------------------------------------------- #
# PDB / motif helpers
# --------------------------------------------------------------------------- #


def expand_motif_residues(motif: MotifSpec) -> list[int]:
    """Return all 1-indexed residue numbers covered by the motif."""
    return motif.residue_indices()


def contig_from_motif(
    *,
    protein_length: int,
    motif: MotifSpec,
    dna_length: int | None = None,
) -> str:
    """Build an RFdiffusion partial-diffusion contig string.

    Partial diffusion requires the contig total length to exactly equal
    the protein length of the input PDB. We build it as:

        [N_flank-N_flank/<motif chain & ranges>/C_flank-C_flank/0 B1-<dna_len>]

    The motif segments are emitted with their literal chain prefix so the
    HMG-box (or zinc-finger) coordinates are held fixed. The DNA chain is
    appended only if ``dna_length`` is supplied (it is for nucleosome
    templates that include the wrapped DNA).
    """
    parts: list[str] = []
    chain = motif.chain
    sorted_ranges = sorted(motif.ranges)
    cursor = 1
    for start, end in sorted_ranges:
        if start > cursor:
            flank = start - cursor
            parts.append(f"{flank}-{flank}")
        parts.append(f"{chain}{start}-{end}")
        cursor = end + 1
    if cursor <= protein_length:
        flank = protein_length - cursor + 1
        parts.append(f"{flank}-{flank}")
    contig_protein = "/".join(parts)
    if dna_length is not None and dna_length > 0:
        return f"[{contig_protein}/0 B1-{dna_length}]"
    return f"[{contig_protein}]"


_PDB_ATOM_RE = re.compile(r"^(ATOM|HETATM)")


def count_residues_in_pdb(pdb_path: Path, chain: str) -> int:
    """Count unique residue numbers in ``chain`` of ``pdb_path``."""
    seen: set[int] = set()
    with open(pdb_path) as fh:
        for line in fh:
            if not _PDB_ATOM_RE.match(line):
                continue
            if line[21:22] != chain:
                continue
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            seen.add(resnum)
    return len(seen)


# --------------------------------------------------------------------------- #
# CUDA-OOM retry wrapper
# --------------------------------------------------------------------------- #

_OOM_PATTERNS = (
    "CUDA out of memory",
    "cuda runtime error: out of memory",
    "OutOfMemoryError",
    "cuBLAS_STATUS_ALLOC_FAILED",
    # Apple Metal (MPS) backend out-of-memory signatures.
    "MPS backend out of memory",
    "MPSNDArray error: product of dimension sizes",
)


class OOMError(RuntimeError):
    """Raised after exhausting OOM downscaling retries."""


def _looks_like_oom(text: str) -> bool:
    return any(p.lower() in text.lower() for p in _OOM_PATTERNS)


def run_with_oom_retry(
    cmd: Sequence[str],
    *,
    downscale_arg: str,
    initial_value: int,
    factor: float = 0.5,
    max_retries: int = 3,
    min_value: int = 1,
    logger: logging.Logger | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` and retry with halved array size on CUDA OOM.

    ``downscale_arg`` is a *prefix* of the argument that controls the batch
    size, e.g. ``inference.num_designs=`` for RFdiffusion or
    ``--batch_size`` for ProteinMPNN. The most recent occurrence of
    ``downscale_arg`` in ``cmd`` is replaced with the new value on retry.

    Returns the completed process on success (exit code 0). Raises
    ``OOMError`` if the process keeps OOM'ing after ``max_retries``, or
    ``CalledProcessError`` on any non-OOM non-zero exit.
    """
    log = logger or get_logger("oom_retry")
    current = initial_value
    last_stderr = ""
    for attempt in range(max_retries + 1):
        mutated_cmd = _replace_arg(cmd, downscale_arg, current)
        log.info(
            "Launch attempt %d/%d (%s=%d): %s",
            attempt + 1,
            max_retries + 1,
            downscale_arg,
            current,
            " ".join(mutated_cmd),
        )
        proc = subprocess.run(
            mutated_cmd,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        if proc.returncode == 0:
            log.info("Subprocess succeeded after %d attempt(s).", attempt + 1)
            return proc
        last_stderr = (proc.stderr or "") + (proc.stdout or "")
        if not _looks_like_oom(last_stderr):
            log.error("Subprocess failed with non-OOM error:\n%s", last_stderr[-2000:])
            raise subprocess.CalledProcessError(
                proc.returncode, mutated_cmd, proc.stdout, proc.stderr
            )
        next_value = max(min_value, int(current * factor))
        if next_value == current:
            break
        log.warning(
            "CUDA OOM detected on attempt %d; downscaling %s %d -> %d",
            attempt + 1,
            downscale_arg,
            current,
            next_value,
        )
        current = next_value
    raise OOMError(
        f"Subprocess kept OOM-ing after {max_retries + 1} attempts; "
        f"last {downscale_arg}={current}\n{last_stderr[-2000:]}"
    )


def _replace_arg(cmd: Sequence[str], prefix: str, value: int) -> list[str]:
    """Return a copy of ``cmd`` with the last token starting with ``prefix``
    replaced (Hydra-style ``key=value`` or argparse-style ``--key value``).
    """
    out = list(cmd)
    if prefix.endswith("=") or "=" in prefix:
        # Hydra-style: replace the whole token.
        for i in range(len(out) - 1, -1, -1):
            if out[i].startswith(prefix):
                key = prefix
                out[i] = f"{key}{value}" if prefix.endswith("=") else f"{prefix.split('=')[0]}={value}"
                return out
        out.append(f"{prefix}{value}" if prefix.endswith("=") else f"{prefix.split('=')[0]}={value}")
        return out
    # argparse-style: replace token after the flag.
    for i in range(len(out) - 1, -1, -1):
        if out[i] == prefix and i + 1 < len(out):
            out[i + 1] = str(value)
            return out
    out.extend([prefix, str(value)])
    return out


# --------------------------------------------------------------------------- #
# Dry-run stub writers
# --------------------------------------------------------------------------- #

# Small but biologically plausible SOX2 HMG-box derived sequence used only
# as a placeholder so ProteinMPNN-style FASTAs and Protenix JSONs are well
# formed under --dry-run on a Mac.
_STUB_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _deterministic_rng(seed_key: str) -> random.Random:
    return random.Random(hash(seed_key) & 0xFFFFFFFF)


def _stub_sequence(length: int, seed_key: str) -> str:
    rng = _deterministic_rng(seed_key)
    return "".join(rng.choice(_STUB_AA_ALPHABET) for _ in range(length))


def write_stub_backbone_pdb(path: Path, *, num_residues: int, seed_key: str) -> None:
    """Write a minimal placeholder PDB so downstream stages can iterate
    without a real RFdiffusion install. CA-only, glycine residues, spaced
    along the X axis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"REMARK   1 DRY-RUN STUB BACKBONE (seed={seed_key})"]
    for i in range(1, num_residues + 1):
        x = i * 3.8
        lines.append(
            f"ATOM  {i:5d}  CA  GLY A{i:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def write_stub_fasta(
    path: Path,
    *,
    backbone_id: str,
    num_sequences: int,
    sequence_length: int,
) -> None:
    """Emit a ProteinMPNN-style multi-FASTA for the given backbone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for i in range(num_sequences):
        rng = _deterministic_rng(f"{backbone_id}|{i}")
        score = round(rng.uniform(0.8, 1.5), 4)
        recovery = round(rng.uniform(0.3, 0.7), 4)
        seq = _stub_sequence(sequence_length, f"{backbone_id}|{i}")
        blocks.append(
            f">{backbone_id}|seq{i:03d}|score={score}|recovery={recovery}\n{seq}"
        )
    path.write_text("\n".join(blocks) + "\n")


def write_stub_protenix_summary(
    path: Path,
    *,
    name: str,
    plddt_mean: float | None = None,
    iptm: float | None = None,
    interface_pae: float | None = None,
) -> None:
    """Emit a minimal Protenix ``summary_confidences.json``."""
    rng = _deterministic_rng(name)
    if plddt_mean is None:
        plddt_mean = rng.uniform(70.0, 92.0)
    if iptm is None:
        iptm = rng.uniform(0.45, 0.85)
    if interface_pae is None:
        interface_pae = rng.uniform(4.0, 14.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "ranking_score": round(plddt_mean / 100.0, 4),
        "iptm": round(iptm, 4),
        "chain_pair_iptm": [[1.0, iptm], [iptm, 1.0]],
        "chain_pair_pae_min": [[1.5, interface_pae], [interface_pae, 1.5]],
        "plddt_mean": round(plddt_mean, 2),
        "plddt_designed_region_mean": round(plddt_mean, 2),
        "interface_pae_mean": round(interface_pae, 3),
    }
    path.write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# Sentinels (used by the orchestrator's --resume-from)
# --------------------------------------------------------------------------- #


def stage_done(out_dir: Path) -> Path:
    return Path(out_dir) / "_DONE"


def mark_stage_done(out_dir: Path) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stage_done(out_dir).write_text("ok\n")


def already_done(out_dir: Path) -> bool:
    return stage_done(out_dir).exists()


# --------------------------------------------------------------------------- #
# Misc small helpers
# --------------------------------------------------------------------------- #


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def iter_files(directory: Path, suffix: str) -> Iterable[Path]:
    if not Path(directory).is_dir():
        return []
    return sorted(p for p in Path(directory).iterdir() if p.suffix == suffix)


def which(executable: str) -> str | None:
    """Best-effort PATH lookup that also handles user-vendored binaries."""
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for d in paths:
        candidate = Path(d) / executable
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None
