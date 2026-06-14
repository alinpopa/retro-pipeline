#!/usr/bin/env python3
"""Visualisation CLI for the retro_pipeline.

Generates static (matplotlib/seaborn) plots and interactive (plotly)
charts from each stage's output, plus standalone HTML 3D viewers for
backbone and predicted-structure files.

Usage
-----

    # The shortest path: a single interactive HTML report combining
    # metrics, rankings, Pareto front, and 3D structure viewers.
    retro-visualize report --open

    # All static plots (PNG) for stages 1-5
    retro-visualize all

    # Per-stage plots (static PNG)
    retro-visualize funnel       # how many designs survive each stage
    retro-visualize metrics      # Protenix confidence metrics
    retro-visualize ddg          # FoldX ΔΔG distribution
    retro-visualize ranking      # Pareto front + ranking

    # Standalone 3D structure viewer (HTML)
    retro-visualize structures
    retro-visualize backbones
    retro-visualize design sox2_00002__seq000   # one specific design

All commands default to the workspace at retro_pipeline/workspace,
override with --workspace <path>. Add --open to launch in browser.

Or via the module path:
    python -m scripts.visualize <command>
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import webbrowser
from pathlib import Path

import pandas as pd

# Optional imports — gracefully degraded if missing.
# ----------------------------------------------------------------------

_HAS_MPL: bool
_HAS_SNS: bool
try:
    import matplotlib  # noqa: F401

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

_HAS_PLOTLY: bool
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

_HAS_PY3DMOL: bool
try:
    import py3Dmol  # noqa: F401

    _HAS_PY3DMOL = True
except ImportError:
    _HAS_PY3DMOL = False

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_workspace(ws: Path | None) -> Path:
    """Default to the canonical workspace dir next to scripts/."""
    if ws is not None:
        return Path(ws).resolve()
    # Walk up from scripts/visualize.py -> retro_pipeline/
    here = Path(__file__).resolve().parent.parent
    return here / "workspace"


def _require_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {label} not found at {path}", file=sys.stderr)
        print(f"  Run the corresponding pipeline stage first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if df.empty:
        print(f"WARNING: {label} at {path} is empty.", file=sys.stderr)
    return df


# ---------------------------------------------------------------------------
# Matplotlib / Seaborn helpers (static PNG)
# ---------------------------------------------------------------------------


def _ensure_mpl():
    if not _HAS_MPL:
        print(
            "ERROR: matplotlib is required for static plots. "
            "Install with: pip install 'retro-pipeline[viz]'",
            file=sys.stderr,
        )
        sys.exit(1)
    # Direct matplotlib's cache to a writable temp dir if the default
    # ($HOME/.matplotlib) is not writable (e.g. sandboxed CI runners).
    if "MPLCONFIGDIR" not in os.environ:
        default = Path.home() / ".matplotlib"
        if not default.exists() and not os.access(Path.home(), os.W_OK):
            os.environ["MPLCONFIGDIR"] = "/tmp/retro_pipeline_mpl"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "font.size": 10,
    })
    return plt


def _ensure_sns():
    _ensure_mpl()
    if not _HAS_SNS:
        print(
            "ERROR: seaborn is required for static plots. "
            "Install with: pip install 'retro-pipeline[viz]'",
            file=sys.stderr,
        )
        sys.exit(1)
    import matplotlib.pyplot as plt

    sns.set_style("whitegrid")
    return plt


def _save_or_show(fig, path: Path | None, stem: str, fmt: str = "png"):
    if path is None:
        fig.tight_layout()
        fig.show()
    else:
        dest = path / f"{stem}.{fmt}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, format=fmt)
        print(f"  → saved {dest}")


def _figsize(num_cols: int) -> tuple[int, int]:
    return (5 * num_cols, 4)


# ---------------------------------------------------------------------------
# Plot: Protenix metrics  (stage 3)
# ---------------------------------------------------------------------------


def plot_metrics(
    workspace: Path,
    save_dir: Path | None = None,
    fmt: str = "png",
) -> dict[str, Path]:
    """Histograms + scatter matrix of Protenix confidence metrics.

    Returns a dict of {plot_name: Path} for generated files.
    """
    plt = _ensure_sns()
    metrics_csv = workspace / "03_predictions" / "metrics.csv"
    df = _require_csv(metrics_csv, "Protenix metrics")

    passed = df[df["passed"] == True] if "passed" in df.columns else df
    failed = df[df["passed"] == False] if "passed" in df.columns else pd.DataFrame()

    out_paths: dict[str, Path] = {}

    # --- 2×2 panel: pLDDT, ipTM, interface PAE, ranking score ---
    fig, axes = plt.subplots(2, 2, figsize=_figsize(2))
    col_map = {
        "pLDDT (designed)": ("plddt_designed_region_mean", 0, 0),
        "ipTM": ("iptm", 0, 1),
        "Interface PAE (Å)": ("interface_pae_mean", 1, 0),
        "Ranking score": ("ranking_score", 1, 1),
    }
    for label, (col, r, c) in col_map.items():
        if col not in df.columns:
            axes[r, c].set_title(f"{label}\n(column not found)")
            continue
        ax = axes[r, c]
        if not failed.empty and col in failed.columns:
            sns.histplot(failed[col], color="tomato", alpha=0.5, label="fail", ax=ax)
        sns.histplot(passed[col], color="steelblue", alpha=0.6, label="pass", ax=ax)
        ax.set_title(label)
        ax.set_ylabel("")
        if r == 0 and c == 1:
            ax.legend(fontsize=8)
    fig.suptitle("Protenix confidence metrics", fontsize=13, y=1.02)
    _save_or_show(fig, save_dir, "01_metrics_histograms", fmt)
    out_paths["histograms"] = save_dir / f"01_metrics_histograms.{fmt}"

    # --- Scatter matrix (pass/fail) ---
    scatter_cols = ["iptm", "plddt_designed_region_mean", "interface_pae_mean"]
    scatter_cols = [c for c in scatter_cols if c in df.columns]
    if len(scatter_cols) >= 2:
        fig2, ax2 = plt.subplots(figsize=_figsize(1))
        hue_data = "passed" if "passed" in df.columns else None
        sns.scatterplot(
            data=df,
            x=scatter_cols[0],
            y=scatter_cols[1],
            hue=hue_data,
            alpha=0.6,
            ax=ax2,
        )
        ax2.set_title(f"{scatter_cols[0]} vs {scatter_cols[1]}")
        _save_or_show(fig2, save_dir, "02_metrics_scatter", fmt)
        out_paths["scatter"] = save_dir / f"02_metrics_scatter.{fmt}"

    return out_paths


# ---------------------------------------------------------------------------
# Plot: FoldX ΔΔG  (stage 4)
# ---------------------------------------------------------------------------


def plot_ddg(
    workspace: Path,
    save_dir: Path | None = None,
    fmt: str = "png",
) -> dict[str, Path]:
    """ΔΔG distribution + hydrophobic SASA scatter."""
    plt = _ensure_sns()
    ddg_csv = workspace / "04_thermodynamics" / "ddg.csv"
    df = _require_csv(ddg_csv, "FoldX ΔΔG")

    out_paths: dict[str, Path] = {}

    # --- ΔΔG histogram ---
    fig, axes = plt.subplots(1, 2, figsize=_figsize(2))
    if "ddg_kcal_mol" in df.columns:
        hue = "passed" if "passed" in df.columns else None
        sns.histplot(df, x="ddg_kcal_mol", hue=hue, ax=axes[0])
        axes[0].axvline(0, color="red", ls="--", lw=1, label="ΔΔG=0")
        axes[0].set_title("ΔΔG (kcal/mol)")
    if "hydrophobic_sasa" in df.columns:
        xcol = "ddg_kcal_mol" if "ddg_kcal_mol" in df.columns else None
        if xcol:
            sns.scatterplot(
                df, x=xcol, y="hydrophobic_sasa", hue=hue, alpha=0.6, ax=axes[1]
            )
        axes[1].set_title("ΔΔG vs hydrophobic SASA")
    fig.suptitle("FoldX thermodynamic stability", fontsize=13, y=1.02)
    _save_or_show(fig, save_dir, "03_ddg_distribution", fmt)
    out_paths["histograms"] = save_dir / f"03_ddg_distribution.{fmt}"

    return out_paths


# ---------------------------------------------------------------------------
# Plot: ranked designs + Pareto front  (stage 5)
# ---------------------------------------------------------------------------


def plot_ranking(
    workspace: Path,
    save_dir: Path | None = None,
    fmt: str = "png",
) -> dict[str, Path]:
    """Pareto front + composite-score overview."""
    plt = _ensure_sns()
    ranking_csv = workspace / "ranked_designs.csv"
    df = _require_csv(ranking_csv, "ranked designs")

    pareto_csv = workspace / "pareto_front.csv"
    pareto = _require_csv(pareto_csv, "Pareto front")

    out_paths: dict[str, Path] = {}

    # --- 2D scatter: pLDDT vs ipTM, coloured by ΔΔG ---
    if all(c in df.columns for c in ("plddt_designed", "iptm", "ddg_kcal_mol")):
        fig, ax = plt.subplots(figsize=_figsize(1))
        sc = ax.scatter(
            df["plddt_designed"],
            df["iptm"],
            c=df["ddg_kcal_mol"],
            cmap="viridis_r",
            alpha=0.7,
            s=20,
        )
        ax.scatter(
            pareto["plddt_designed"],
            pareto["iptm"],
            marker="D",
            facecolors="none",
            edgecolors="red",
            s=80,
            label="Pareto front",
        )
        ax.set_xlabel("pLDDT (designed)")
        ax.set_ylabel("ipTM")
        ax.set_title("Design ranking — pLDDT vs ipTM (color = ΔΔG)")
        plt.colorbar(sc, ax=ax, label="ΔΔG (kcal/mol)")
        ax.legend()
        _save_or_show(fig, save_dir, "04_ranking_scatter", fmt)
        out_paths["scatter"] = save_dir / f"04_ranking_scatter.{fmt}"

    # --- Top-N composite score bar ---
    if "composite_score" in df.columns and "rank" in df.columns:
        n = min(20, len(df))
        top = df.head(n)
        fig, ax = plt.subplots(figsize=(8, max(3, n * 0.3)))
        colors = ["gold" if r else "steelblue" for r in top.get("on_pareto_front", False)]
        ax.barh(
            range(n),
            top["composite_score"],
            color=colors,
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels(top["job_name"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Composite score (lower is better)")
        ax.set_title(f"Top-{n} designs by composite score")
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="gold", label="On Pareto front"),
                Patch(color="steelblue", label="Off Pareto front"),
            ],
            fontsize=8,
        )
        _save_or_show(fig, save_dir, "05_top_ranking", fmt)
        out_paths["top_bars"] = save_dir / f"05_top_ranking.{fmt}"

    return out_paths


# ---------------------------------------------------------------------------
# Plot: pipeline funnel  (cross-stage survival)
# ---------------------------------------------------------------------------


def _count_stage_outputs(workspace: Path) -> list[tuple[str, int]]:
    """Count surviving candidates at each pipeline stage."""
    counts: list[tuple[str, int]] = []

    bb = workspace / "01_backbones"
    counts.append(("1. Backbones (RFdiffusion)", len(list(bb.glob("*.pdb"))) if bb.is_dir() else 0))

    seq = workspace / "02_sequences"
    fasta_files = list(seq.glob("*.fasta")) if seq.is_dir() else []
    n_seqs = 0
    for f in fasta_files:
        try:
            n_seqs += sum(1 for line in f.read_text().splitlines() if line.startswith(">"))
        except OSError:
            pass
    counts.append(("2. Sequences (ProteinMPNN)", n_seqs))

    metrics_csv = workspace / "03_predictions" / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        counts.append(("3a. Predicted (Protenix, all)", len(df)))
        if "passed" in df.columns:
            counts.append(("3b. Passed PAE/pLDDT filter", int(df["passed"].astype(bool).sum())))
    else:
        counts.append(("3. Predicted (Protenix)", 0))

    ddg_csv = workspace / "04_thermodynamics" / "ddg.csv"
    if ddg_csv.exists():
        df = pd.read_csv(ddg_csv)
        if "passed" in df.columns:
            counts.append(("4. Passed ΔΔG filter (FoldX)", int(df["passed"].astype(bool).sum())))
        else:
            counts.append(("4. FoldX evaluated", len(df)))

    ranked_csv = workspace / "ranked_designs.csv"
    if ranked_csv.exists():
        df = pd.read_csv(ranked_csv)
        counts.append(("5. Final ranked designs", len(df)))
        pareto = workspace / "pareto_front.csv"
        if pareto.exists():
            pdf = pd.read_csv(pareto)
            counts.append(("6. Pareto front", len(pdf)))

    return counts


def plot_funnel(
    workspace: Path,
    save_dir: Path | None = None,
    fmt: str = "png",
) -> dict[str, Path]:
    """Horizontal funnel showing how many candidates survive each stage."""
    plt = _ensure_sns()
    stages = _count_stage_outputs(workspace)
    if not stages:
        print("No pipeline outputs found.", file=sys.stderr)
        return {}

    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(9, max(3, len(stages) * 0.5)))
    colors = sns.color_palette("viridis_r", n_colors=len(stages))
    bars = ax.barh(range(len(stages)), counts, color=colors)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Candidates surviving")
    ax.set_title("Pipeline funnel — design attrition across stages")

    # Use the largest stage count as the denominator so the funnel reads as
    # a percentage of the overall candidate pool (sequences fan out from
    # backbones, so the first stage isn't necessarily the largest).
    denom = max(counts) if counts and max(counts) > 0 else 1
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = 100.0 * count / denom
        if i == 0:
            label = f"  {count:,}  ({pct:.1f}%)"
        else:
            prev = counts[i - 1] if counts[i - 1] > 0 else 1
            survival = 100.0 * count / prev
            label = f"  {count:,}  ({pct:.1f}% of max, {survival:.1f}% of prev)"
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
        )
    ax.margins(x=0.35)

    out_paths: dict[str, Path] = {}
    _save_or_show(fig, save_dir, "00_pipeline_funnel", fmt)
    if save_dir is not None:
        out_paths["funnel"] = save_dir / f"00_pipeline_funnel.{fmt}"
    return out_paths


# ---------------------------------------------------------------------------
# Interactive Plotly HTML: full dashboard
# ---------------------------------------------------------------------------


def _plotly_metrics_dashboard(df_metrics: pd.DataFrame) -> str:
    """Return Plotly HTML div for the metrics panel."""
    if not _HAS_PLOTLY:
        return "<p>Install plotly to see interactive charts.</p>"

    color_map = {True: "steelblue", False: "tomato"}

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "pLDDT (designed)",
            "ipTM",
            "Interface PAE (Å)",
            "Ranking score",
            "pLDDT vs ipTM",
            "pLDDT vs Interface PAE",
        ),
    )

    col_positions = [
        ("plddt_designed_region_mean", 1, 1),
        ("iptm", 1, 2),
        ("interface_pae_mean", 1, 3),
        ("ranking_score", 2, 1),
    ]
    for col, r, c in col_positions:
        if col not in df_metrics.columns:
            continue
        for passed_val, color in color_map.items():
            subset = df_metrics[df_metrics.get("passed", False) == passed_val]
            if subset.empty:
                continue
            fig.add_trace(
                go.Histogram(
                    x=subset[col],
                    name="pass" if passed_val else "fail",
                    marker_color=color,
                    opacity=0.5,
                    legendgroup="pass" if passed_val else "fail",
                    showlegend=(r == 1 and c == 1),
                ),
                row=r,
                col=c,
            )

    # Scatter plots
    for col_x, col_y, r, c in [
        ("plddt_designed_region_mean", "iptm", 2, 2),
        ("plddt_designed_region_mean", "interface_pae_mean", 2, 3),
    ]:
        if col_x not in df_metrics.columns or col_y not in df_metrics.columns:
            continue
        for passed_val, color in color_map.items():
            subset = df_metrics[df_metrics.get("passed", False) == passed_val]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset[col_x],
                    y=subset[col_y],
                    mode="markers",
                    name="pass" if passed_val else "fail",
                    marker=dict(color=color, size=4, opacity=0.5),
                    legendgroup="pass" if passed_val else "fail",
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

    fig.update_layout(
        height=650,
        title_text="Protenix confidence metrics (interactive)",
        hovermode="closest",
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="metrics")


def _plotly_ranking_dashboard(df_ranked: pd.DataFrame) -> str:
    """Return Plotly HTML div for the ranking panel."""
    if not _HAS_PLOTLY:
        return "<p>Install plotly to see interactive charts.</p>"

    # Parallel coordinates
    dims = [
        "ddg_kcal_mol",
        "iptm",
        "plddt_designed",
        "interface_pae",
        "hydrophobic_sasa",
    ]
    dims = [d for d in dims if d in df_ranked.columns]
    if "composite_score" in df_ranked.columns:
        color_col = "composite_score"
    else:
        color_col = dims[0] if dims else None

    if dims and color_col:
        fig = px.parallel_coordinates(
            df_ranked,
            dimensions=dims,
            color=color_col,
            color_continuous_scale="viridis_r",
            title="Design ranking — parallel coordinates",
        )
        fig.update_layout(height=500, margin=dict(t=50))
        html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="ranking")
    else:
        html = ""

    return html


def _plotly_pareto_3d(df_ranked: pd.DataFrame, pareto_ids: set[str]) -> str:
    """Return Plotly HTML div for a 3D Pareto scatter."""
    if not _HAS_PLOTLY:
        return "<p>Install plotly to see interactive charts.</p>"

    dims = ["ddg_kcal_mol", "iptm", "plddt_designed"]
    dims = [d for d in dims if d in df_ranked.columns]
    if len(dims) < 3:
        return ""

    pareto_df = df_ranked[df_ranked["job_name"].isin(pareto_ids)] if "job_name" in df_ranked.columns else pd.DataFrame()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df_ranked[dims[0]],
            y=df_ranked[dims[1]],
            z=df_ranked[dims[2]],
            mode="markers",
            marker=dict(
                size=3,
                color=df_ranked.get("composite_score", df_ranked[dims[0]]),
                colorscale="Viridis_r",
                showscale=True,
                colorbar=dict(title="Score"),
            ),
            text=df_ranked.get("job_name", df_ranked.index),
            name="All designs",
        )
    )
    if not pareto_df.empty:
        fig.add_trace(
            go.Scatter3d(
                x=pareto_df[dims[0]],
                y=pareto_df[dims[1]],
                z=pareto_df[dims[2]],
                mode="markers",
                marker=dict(size=8, color="red", symbol="diamond"),
                text=pareto_df.get("job_name", pareto_df.index),
                name="Pareto front",
            )
        )
    fig.update_layout(
        height=600,
        title="3D objective space — Pareto front highlighted",
        scene=dict(
            xaxis_title=dims[0],
            yaxis_title=dims[1],
            zaxis_title=dims[2],
        ),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="pareto3d")


# ---------------------------------------------------------------------------
# 3D structure viewer (standalone HTML via py3Dmol)
# ---------------------------------------------------------------------------


def _viewer_body_fragment(
    structures: list[Path],
    title: str,
    container_id: str = "viewer",
    height: str = "95vh",
) -> str:
    """Return an HTML fragment (no <html>/<body> tags) with a py3Dmol viewer.

    Safe to embed inside another HTML page. The fragment includes its own
    script tag for 3Dmol.js (loaded from CDN) and an isolated container id.
    """
    blocks: list[str] = []
    for path in structures:
        b64 = base64.b64encode(path.read_bytes()).decode()
        fmt = path.suffix.lstrip(".").lower() or "pdb"
        blocks.append(
            f'viewer_{container_id}.addModel(atob("{b64}"), "{fmt}");'
        )
    add_models_js = "\n".join(blocks)

    return f"""
<div class="viewer-wrap" style="border:1px solid #ddd; border-radius:6px; overflow:hidden;">
  <div class="viewer-info" style="padding:6px 12px; background:#f5f5f5; border-bottom:1px solid #ddd; font-size:13px;">
    <strong>{title}</strong> &mdash; {len(structures)} structure(s) shown.
  </div>
  <div id="{container_id}" style="width:100%; height:{height}; position:relative;"></div>
</div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
(function() {{
  function init() {{
    if (typeof $3Dmol === "undefined") {{ setTimeout(init, 100); return; }}
    var viewer_{container_id} = $3Dmol.createViewer("{container_id}", {{backgroundColor: "white"}});
    {add_models_js}
    viewer_{container_id}.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
    viewer_{container_id}.zoomTo();
    viewer_{container_id}.render();
  }}
  init();
}})();
</script>
"""


def _make_viewer_html(
    structure_paths: list[Path],
    title: str = "Predicted structures",
    max_structures: int = 50,
) -> str:
    """Generate a standalone HTML page wrapping the viewer body fragment.

    Accepts any mix of .pdb and .cif files.
    """
    if not _HAS_PY3DMOL:
        return (
            "<!DOCTYPE html><html><body><p>Install py3Dmol to view 3D structures: "
            "pip install 'retro-pipeline[viz]'</p></body></html>"
        )

    structures = structure_paths[:max_structures]
    if not structures:
        return "<!DOCTYPE html><html><body><p>No structure files found.</p></body></html>"

    body = _viewer_body_fragment(structures, title=title, container_id="viewer_main", height="92vh")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def _maybe_open(path: Path, args: argparse.Namespace) -> None:
    if getattr(args, "open", False):
        try:
            webbrowser.open(path.resolve().as_uri())
        except Exception as exc:  # noqa: BLE001
            print(f"Could not open browser: {exc}", file=sys.stderr)


def _cmd_metrics(args: argparse.Namespace) -> int:
    save = args.save or args.workspace
    plot_metrics(args.workspace, save_dir=save, fmt=args.format)
    return 0


def _cmd_ddg(args: argparse.Namespace) -> int:
    save = args.save or args.workspace
    plot_ddg(args.workspace, save_dir=save, fmt=args.format)
    return 0


def _cmd_ranking(args: argparse.Namespace) -> int:
    save = args.save or args.workspace
    plot_ranking(args.workspace, save_dir=save, fmt=args.format)
    return 0


def _cmd_funnel(args: argparse.Namespace) -> int:
    save = args.save or args.workspace
    plot_funnel(args.workspace, save_dir=save, fmt=args.format)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    """Run funnel + metrics + ddg + ranking in one shot."""
    save = args.save or args.workspace
    save.mkdir(parents=True, exist_ok=True)

    print("-- Pipeline funnel ---------------------------")
    plot_funnel(args.workspace, save_dir=save, fmt=args.format)
    print("-- Protenix metrics --------------------------")
    plot_metrics(args.workspace, save_dir=save, fmt=args.format)
    print("-- FoldX ΔΔG ---------------------------------")
    plot_ddg(args.workspace, save_dir=save, fmt=args.format)
    print("-- Ranking + Pareto front --------------------")
    plot_ranking(args.workspace, save_dir=save, fmt=args.format)

    print(f"\nAll plots saved to {save}/")
    return 0


def _cmd_design(args: argparse.Namespace) -> int:
    """Generate a standalone HTML 3D viewer for a single named design."""
    name = args.name
    if not name:
        print("ERROR: --name (positional) is required for the 'design' command", file=sys.stderr)
        return 2

    candidates: list[Path] = []
    for sub in ("03_predictions/passed", "final_top", "03_predictions"):
        base = args.workspace / sub
        if not base.is_dir():
            continue
        for ext in ("*.cif", "*.pdb"):
            candidates.extend([p for p in base.rglob(ext) if name in p.stem])
        if candidates:
            break

    if not candidates:
        print(f"ERROR: no structure file found for design '{name}'", file=sys.stderr)
        print(f"  Searched under {args.workspace}/03_predictions/ and {args.workspace}/final_top/")
        return 1

    output = args.output or (args.workspace / f"viz_design_{name}.html")
    html = _make_viewer_html(candidates[:1], title=f"Design: {name}")
    output.write_text(html)
    print(f"  -> saved {output}  ({candidates[0]})")
    _maybe_open(output, args)
    return 0


def _cmd_structures(args: argparse.Namespace) -> int:
    """Generate a standalone HTML 3D viewer for passed CIF files."""
    passed_dir = args.workspace / "03_predictions" / "passed"
    if not passed_dir.is_dir():
        print(
            f"ERROR: passed structures directory not found at {passed_dir}",
            file=sys.stderr,
        )
        return 1

    cif_paths = sorted(passed_dir.glob("*.cif"))
    if not cif_paths:
        print(f"No .cif files found in {passed_dir}")
        return 1

    output = args.output or (args.workspace / "viz_structures.html")
    html = _make_viewer_html(cif_paths, title="Predicted structures (passed)", max_structures=args.max_structs)
    output.write_text(html)
    print(f"  -> saved {output}  ({min(len(cif_paths), args.max_structs)}/{len(cif_paths)} structures)")
    _maybe_open(output, args)
    return 0


def _cmd_backbones(args: argparse.Namespace) -> int:
    """Generate a standalone HTML 3D viewer overlaying backbone PDBs."""
    bb_dir = args.workspace / "01_backbones"
    if not bb_dir.is_dir():
        print(f"ERROR: backbones directory not found at {bb_dir}", file=sys.stderr)
        return 1

    pdb_paths = sorted(bb_dir.glob("*.pdb"))
    if not pdb_paths:
        print(f"No .pdb files found in {bb_dir}")
        return 1

    output = args.output or (args.workspace / "viz_backbones.html")
    html = _make_viewer_html(
        pdb_paths,
        title=f"Backbone overlay ({len(pdb_paths)} structures)",
        max_structures=args.max_structs,
    )
    output.write_text(html)
    print(f"  -> saved {output}  ({min(len(pdb_paths), args.max_structs)}/{len(pdb_paths)} backbones)")
    _maybe_open(output, args)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Generate a single interactive HTML report combining all plots + viewers."""
    ws = args.workspace
    output = args.output or (ws / "viz_report.html")
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics_csv = ws / "03_predictions" / "metrics.csv"
    ranking_csv = ws / "ranked_designs.csv"
    pareto_csv = ws / "pareto_front.csv"

    metrics_df = _require_csv(metrics_csv, "Protenix metrics") if metrics_csv.exists() else pd.DataFrame()
    ranked_df = _require_csv(ranking_csv, "ranked designs") if ranking_csv.exists() else pd.DataFrame()
    pareto_df = _require_csv(pareto_csv, "Pareto front") if pareto_csv.exists() else pd.DataFrame()
    pareto_ids = set(pareto_df.get("job_name", [])) if not pareto_df.empty else set()

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>retro_pipeline — visualisation report</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin:20px; background:#fafafa; }",
        "  h1 { color:#333; border-bottom:2px solid #4a90d9; padding-bottom:8px; }",
        "  h2 { color:#555; margin-top:30px; }",
        "  .plot { background:white; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.1); padding:12px; margin:12px 0; }",
        "  .note { color:#888; font-style:italic; }",
        "</style></head><body>",
        "<h1>retro_pipeline — visualisation report</h1>",
        f"<p class='note'>Generated from workspace: {ws}</p>",
    ]

    # Section 1: metrics
    parts.append("<h2>1. Protenix confidence metrics</h2><div class='plot'>")
    if not metrics_df.empty:
        parts.append(_plotly_metrics_dashboard(metrics_df))
    else:
        parts.append("<p class='note'>No metrics.csv found — run stage 3 (Protenix) first.</p>")
    parts.append("</div>")

    # Section 2: ranking
    parts.append("<h2>2. Design ranking</h2><div class='plot'>")
    if not ranked_df.empty:
        parts.append(_plotly_ranking_dashboard(ranked_df))
    else:
        parts.append("<p class='note'>No ranked_designs.csv found — run stage 5 (score_and_rank) first.</p>")
    parts.append("</div>")

    # Section 3: Pareto 3D
    parts.append("<h2>3. Pareto front (3D)</h2><div class='plot'>")
    if not ranked_df.empty:
        parts.append(_plotly_pareto_3d(ranked_df, pareto_ids))
    else:
        parts.append("<p class='note'>Run ranking stage first.</p>")
    parts.append("</div>")

    # Section 4: 3D structures (passed predictions)
    parts.append("<h2>4. Predicted structures</h2><div class='plot'>")
    passed_dir = ws / "03_predictions" / "passed"
    cif_paths = sorted(passed_dir.glob("*.cif")) if passed_dir.is_dir() else []
    if cif_paths:
        parts.append(
            _viewer_body_fragment(
                cif_paths[: args.max_structs],
                title="Predicted structures (passed)",
                container_id="viewer_predictions",
                height="500px",
            )
        )
    else:
        parts.append("<p class='note'>No passed .cif files found.</p>")
    parts.append("</div>")

    # Section 5: backbones (RFdiffusion outputs)
    parts.append("<h2>5. Backbones (RFdiffusion)</h2><div class='plot'>")
    bb_dir = ws / "01_backbones"
    pdb_paths = sorted(bb_dir.glob("*.pdb")) if bb_dir.is_dir() else []
    if pdb_paths:
        parts.append(
            _viewer_body_fragment(
                pdb_paths[: args.max_structs],
                title="Backbones overlay",
                container_id="viewer_backbones",
                height="500px",
            )
        )
    else:
        parts.append("<p class='note'>No backbone .pdb files found — run stage 1 first.</p>")
    parts.append("</div>")

    parts.append("</body></html>")

    output.write_text("\n".join(parts))
    print(f"  -> saved {output}")
    _maybe_open(output, args)
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualisation tools for the retro_pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Path to the workspace directory (default: retro_pipeline/workspace)",
    )
    p.add_argument("--save", type=Path, default=None, help="Output directory for plots (default: workspace)")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Plot file format (default: png)")
    p.add_argument(
        "--max-structs",
        type=int,
        default=50,
        help="Max structures to include in 3D viewer (default: 50)",
    )
    p.add_argument("--output", type=Path, default=None, help="Output file path (for HTML-generating commands)")
    p.add_argument("--open", action="store_true", help="Open the generated HTML file in your default browser")

    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("funnel", help="Plot a stage-by-stage survival funnel")
    sp.set_defaults(func=_cmd_funnel)

    sp = sub.add_parser("metrics", help="Plot Protenix confidence metrics")
    sp.set_defaults(func=_cmd_metrics)

    sp = sub.add_parser("ddg", help="Plot FoldX ΔΔG distribution")
    sp.set_defaults(func=_cmd_ddg)

    sp = sub.add_parser("ranking", help="Plot ranked designs + Pareto front")
    sp.set_defaults(func=_cmd_ranking)

    sp = sub.add_parser("all", help="Generate all static plots (funnel + metrics + ddg + ranking)")
    sp.set_defaults(func=_cmd_all)

    sp = sub.add_parser(
        "report",
        help="Generate a single interactive HTML report combining all plots + 3D viewer",
    )
    sp.set_defaults(func=_cmd_report)

    sp = sub.add_parser("structures", help="Standalone HTML 3D viewer for passed CIF files")
    sp.set_defaults(func=_cmd_structures)

    sp = sub.add_parser("backbones", help="Standalone HTML 3D viewer overlaying backbone PDBs")
    sp.set_defaults(func=_cmd_backbones)

    sp = sub.add_parser("design", help="View a single design's structure by name")
    sp.add_argument("name", help="Design name (e.g. sox2_00002__seq000)")
    sp.set_defaults(func=_cmd_design)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.workspace = _resolve_workspace(args.workspace)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
