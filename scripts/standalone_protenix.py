#!/usr/bin/env python3
"""
Standalone Protenix pipeline — zero dependencies on retro_pipeline.
Runs directly from the Protenix venv. Auto-resume via shard caching.

Test first:  python standalone_protenix.py --fasta_dir ... --out_dir ... --dna CTTTGTTCT --max_fasta 10 --dry-run
Then real:  python standalone_protenix.py --fasta_dir ... --out_dir ... --dna CTTTGTTCT
"""
from __future__ import annotations

import argparse, csv, json, os, shutil, statistics, sys
from dataclasses import dataclass, asdict
from pathlib import Path

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rc(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def read_fasta(p: Path) -> list[tuple[str, str]]:
    out, name, lines = [], None, []
    for line in p.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(lines)))
            name, lines = line[1:].strip(), []
        elif line:
            lines.append(line.strip())
    if name is not None:
        out.append((name, "".join(lines)))
    return out


def build_jobs(fasta_dir: Path, dna: str, max_fasta: int = 0) -> list[dict]:
    jobs = []
    files = sorted(fasta_dir.glob("*.fasta"))
    if max_fasta:
        files = files[:max_fasta]
    for fa in files:
        for hdr, aa in read_fasta(fa):
            tag = hdr.split("|", 2)
            name = f"{fa.stem}__{tag[1] if len(tag) > 1 else 'seq'}"
            jobs.append({
                "name": name,
                "sequences": [
                    {"proteinChain": {"sequence": aa, "count": 1}},
                    {"dnaSequence": {"sequence": dna.upper(), "count": 1}},
                    {"dnaSequence": {"sequence": rc(dna.upper()), "count": 1}},
                ],
            })
    return jobs


@dataclass
class M:
    job_name: str; iptm: float; plddt_mean: float
    plddt_designed_region_mean: float; interface_pae_mean: float
    ranking_score: float; passed: bool; failure_reason: str

    def csv(self) -> dict:
        return asdict(self)


def mean_off_diag(mat: list[list[float]]) -> float:
    n = len(mat)
    if n < 2:
        return float("inf")
    v = [float(mat[i][j]) for i in range(n) for j in range(n) if i != j]
    return statistics.fmean(v) if v else float("inf")


def evaluate(job_name: str, summary: dict, plddt_cutoff: float,
             interface_pae_cutoff: float, iptm_warn: float) -> M:
    iptm = float(summary.get("iptm", 0))
    plddt = float(summary.get("plddt_mean", summary.get("plddt", 0)))
    plddt_des = float(summary.get("plddt_designed_region_mean", plddt))
    score = float(summary.get("ranking_score", 0))

    ipae = summary.get("interface_pae_mean")
    if ipae is not None:
        ipae = float(ipae)
    elif "chain_pair_pae_min" in summary:
        ipae = mean_off_diag(summary["chain_pair_pae_min"])
    else:
        ipae = float("inf")

    reasons = []
    if plddt_des < plddt_cutoff:
        reasons.append(f"plddt={plddt_des:.1f}<{plddt_cutoff}")
    if ipae > interface_pae_cutoff:
        reasons.append(f"ipae={ipae:.2f}>{interface_pae_cutoff}")
    return M(job_name, iptm, plddt, plddt_des, ipae, score, not reasons, ";".join(reasons))


def shard_done(inp: Path, out: Path) -> bool:
    if not out.exists():
        return False
    try:
        jobs = json.loads(inp.read_text())
    except Exception:
        return False
    return all((out / j["name"] / "summary_confidences.json").exists() for j in jobs)


def run_pt(model: str, inp: Path, out: Path) -> None:
    """Run Protenix locally — bypasses colab/web-service queue entirely."""
    # Force local inference env vars before importing protenix
    os.environ["LAYERNORM_TYPE"] = "torch"
    os.environ["PYTHONUNBUFFERED"] = "1"
    # These env vars prevent Protenix from trying to queue to remote services
    os.environ.pop("PROTENIX_SERVICE_URL", None)  # remove if set
    os.environ["PROTENIX_LOCAL_ONLY"] = "1"

    print(f"Running Protenix locally (model={model})", flush=True)
    print(f"  input : {inp}", flush=True)
    print(f"  output: {out}", flush=True)

    from runner.batch_inference import build_inference_config
    from protenix.web_service.colab_request_utils import run_inference_locally

    cfg = build_inference_config(
        input_json=str(inp),
        output_dir=str(out),
        model_name=model,
    )
    run_inference_locally(cfg)
    print("  Done.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone Protenix pipeline")
    ap.add_argument("--fasta_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--dna", required=True, type=str, help="DNA forward strand")
    ap.add_argument("--model", default="protenix_base_default_v1.0.0")
    ap.add_argument("--max_jobs_per_shard", type=int, default=64)
    ap.add_argument("--plddt_cutoff", type=float, default=80.0)
    ap.add_argument("--interface_pae_cutoff", type=float, default=10.0)
    ap.add_argument("--iptm_warn", type=float, default=0.60)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max_fasta", type=int, default=0, help="0=all")
    ap.add_argument("--from_shard", type=int, default=0)
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(a.fasta_dir, a.dna, a.max_fasta)
    if not jobs:
        print(f"No .fasta in {a.fasta_dir}", file=sys.stderr); sys.exit(2)
    print(f"{len(jobs)} job(s)")

    shards = [jobs[i:i + a.max_jobs_per_shard] for i in range(0, len(jobs), a.max_jobs_per_shard)]
    print(f"{len(shards)} shard(s)")
    if a.from_shard:
        shards = shards[a.from_shard:]

    metrics: list[M] = []
    for idx, shard in enumerate(shards, start=a.from_shard):
        sin = a.out_dir / f"in_shard_{idx:04d}.json"
        sout = a.out_dir / f"out_shard_{idx:04d}"
        sin.write_text(json.dumps(shard, indent=2))

        if shard_done(sin, sout):
            print(f"Shard {idx}: skip (done)")
        else:
            sout.mkdir(parents=True, exist_ok=True)
            if a.dry_run:
                print(f"Shard {idx}: DRY ({len(shard)} jobs)")
                for job in shard:
                    d = sout / job["name"]; d.mkdir(parents=True, exist_ok=True)
                    (d / "summary_confidences.json").write_text(json.dumps({
                        "name": job["name"], "ranking_score": 0.85, "iptm": 0.72,
                        "plddt_mean": 88.0, "plddt_designed_region_mean": 86.0,
                        "interface_pae_mean": 5.2}, indent=2))
                    (d / "model.cif").write_text(f"# STUB {job['name']}\n")
            else:
                print(f"Shard {idx}: {len(shard)} jobs")
                run_pt(a.model, sin, sout)

        for job in shard:
            sp = sout / job["name"] / "summary_confidences.json"
            if not sp.exists():
                print(f"  missing {job['name']}"); continue
            m = evaluate(job["name"], json.loads(sp.read_text()),
                         a.plddt_cutoff, a.interface_pae_cutoff, a.iptm_warn)
            metrics.append(m)
            if m.iptm < a.iptm_warn:
                print(f"  low iptm={m.iptm:.3f} {m.job_name}")
            if m.passed:
                for cif in (sout / job["name"]).glob("*.cif"):
                    d = a.out_dir / "passed" / f"{m.job_name}.cif"
                    d.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(cif, d); break

    csv_path = a.out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job_name","iptm","plddt_mean",
            "plddt_designed_region_mean","interface_pae_mean",
            "ranking_score","passed","failure_reason"])
        w.writeheader()
        for r in metrics: w.writerow(r.csv())
    print(f"\n{sum(1 for m in metrics if m.passed)}/{len(metrics)} passed -> {csv_path}")
    (a.out_dir / "_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
