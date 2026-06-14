#!/usr/bin/env python3
"""Final stage — join Protenix and FoldX metrics, Pareto-rank, and copy
the top-N candidates into ``workspace/final_top/``.

This script is the deliverable the user actually consumes. It reads:

* ``<predictions>/metrics.csv``  -- Protenix-stage metrics
* ``<thermo>/ddg.csv``           -- FoldX-stage metrics

joins on ``job_name``, keeps rows that passed BOTH filters, computes a
non-dominated Pareto front over four objectives, and writes:

* ``ranked_designs.csv``  -- full ordered table with composite scores
* ``final_top/*.cif``     -- top-N predicted structures, renamed by rank
* ``pareto_front.csv``    -- the subset on the first non-dominated front
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from .common import ensure_dir, get_logger


@dataclass
class Design:
    job_name: str
    iptm: float
    plddt_designed: float
    interface_pae: float
    ddg_kcal_mol: float
    hydrophobic_sasa: float

    def objective_vector(self) -> tuple[float, float, float, float]:
        """Objective tuple (all 'lower is better' after sign flip):

        - ddg_kcal_mol   : minimize (more stable)
        - -iptm          : minimize -> maximize iptm
        - -plddt_designed: minimize -> maximize plddt
        - interface_pae  : minimize
        """
        return (self.ddg_kcal_mol, -self.iptm, -self.plddt_designed, self.interface_pae)


def _read_csv_dict(path: Path, key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[row[key]] = row
    return out


def _to_float(x: str, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def pareto_front(designs: list[Design]) -> list[Design]:
    """Return the subset of ``designs`` on the first non-dominated front.

    A design ``a`` dominates ``b`` iff ``a`` is no worse in every
    objective and strictly better in at least one.
    """
    vecs = [d.objective_vector() for d in designs]
    front: list[Design] = []
    for i, di in enumerate(designs):
        vi = vecs[i]
        dominated = False
        for j, vj in enumerate(vecs):
            if i == j:
                continue
            if all(vj[k] <= vi[k] for k in range(len(vi))) and any(
                vj[k] < vi[k] for k in range(len(vi))
            ):
                dominated = True
                break
        if not dominated:
            front.append(di)
    return front


def composite_score(d: Design) -> float:
    """Simple convex combination used to linearize the ranking. Lower is
    better. Coefficients chosen so each term lands roughly in [0, 1] for
    typical inputs."""
    if any(math.isnan(x) for x in (d.ddg_kcal_mol, d.iptm, d.plddt_designed, d.interface_pae)):
        return float("inf")
    return (
        0.40 * d.ddg_kcal_mol
        - 0.25 * d.iptm * 5.0           # iptm in [0,1] -> scaled
        - 0.20 * (d.plddt_designed - 80.0) / 20.0
        + 0.15 * d.interface_pae / 10.0
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Join Protenix + FoldX metrics, Pareto-rank, copy top-N.")
    p.add_argument("--predictions", required=True, type=Path, help="workspace/03_predictions")
    p.add_argument("--thermo", required=True, type=Path, help="workspace/04_thermodynamics")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Defaults to <predictions>/../final_top",
    )
    p.add_argument("--top_n", type=int, default=100)
    args = p.parse_args(argv)

    log = get_logger("rank", log_dir=args.predictions.parent / "logs")

    protenix_rows = _read_csv_dict(args.predictions / "metrics.csv", key="job_name")
    foldx_rows = _read_csv_dict(args.thermo / "ddg.csv", key="job_name")
    log.info(
        "Loaded %d Protenix rows and %d FoldX rows.",
        len(protenix_rows),
        len(foldx_rows),
    )

    out_dir = args.out_dir or (args.predictions.parent / "final_top")
    ensure_dir(out_dir)

    designs: list[Design] = []
    for job_name, p_row in protenix_rows.items():
        f_row = foldx_rows.get(job_name)
        if f_row is None:
            continue
        if p_row.get("passed", "").lower() not in ("true", "1", "yes"):
            continue
        if f_row.get("passed", "").lower() not in ("true", "1", "yes"):
            continue
        designs.append(
            Design(
                job_name=job_name,
                iptm=_to_float(p_row.get("iptm", "nan")),
                plddt_designed=_to_float(p_row.get("plddt_designed_region_mean", "nan")),
                interface_pae=_to_float(p_row.get("interface_pae_mean", "nan")),
                ddg_kcal_mol=_to_float(f_row.get("ddg_kcal_mol", "nan")),
                hydrophobic_sasa=_to_float(f_row.get("hydrophobic_sasa", "nan")),
            )
        )

    log.info("%d candidate(s) passed both Protenix and FoldX filters.", len(designs))

    front = pareto_front(designs) if designs else []
    log.info("Pareto front size: %d", len(front))

    designs_sorted = sorted(designs, key=composite_score)

    ranked_csv = out_dir.parent / "ranked_designs.csv"
    with open(ranked_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rank",
                "job_name",
                "composite_score",
                "ddg_kcal_mol",
                "iptm",
                "plddt_designed",
                "interface_pae",
                "hydrophobic_sasa",
                "on_pareto_front",
            ]
        )
        front_ids = {d.job_name for d in front}
        for rank, d in enumerate(designs_sorted, 1):
            w.writerow(
                [
                    rank,
                    d.job_name,
                    f"{composite_score(d):.4f}",
                    f"{d.ddg_kcal_mol:.3f}",
                    f"{d.iptm:.3f}",
                    f"{d.plddt_designed:.2f}",
                    f"{d.interface_pae:.3f}",
                    f"{d.hydrophobic_sasa:.2f}",
                    d.job_name in front_ids,
                ]
            )
    log.info("Wrote %s (%d rows).", ranked_csv, len(designs_sorted))

    front_csv = out_dir.parent / "pareto_front.csv"
    with open(front_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "job_name",
                "ddg_kcal_mol",
                "iptm",
                "plddt_designed",
                "interface_pae",
                "hydrophobic_sasa",
            ]
        )
        for d in front:
            w.writerow(
                [
                    d.job_name,
                    f"{d.ddg_kcal_mol:.3f}",
                    f"{d.iptm:.3f}",
                    f"{d.plddt_designed:.2f}",
                    f"{d.interface_pae:.3f}",
                    f"{d.hydrophobic_sasa:.2f}",
                ]
            )
    log.info("Wrote %s (%d rows).", front_csv, len(front))

    # Copy top-N predicted structures into final_top/.
    passed_dir = args.predictions / "passed"
    n_copied = 0
    for rank, d in enumerate(designs_sorted[: args.top_n], 1):
        src = passed_dir / f"{d.job_name}.cif"
        if not src.exists():
            # In --dry-run we wrote model.cif in the per-job shard dir
            # rather than the 'passed' directory.
            candidates = list(args.predictions.rglob(f"{d.job_name}/model.cif"))
            if candidates:
                src = candidates[0]
            else:
                continue
        dest = out_dir / f"rank{rank:04d}__{d.job_name}.cif"
        shutil.copy(src, dest)
        n_copied += 1
    log.info("Copied %d top structure(s) to %s.", n_copied, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
