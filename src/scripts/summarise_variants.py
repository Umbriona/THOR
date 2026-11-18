#!/usr/bin/env python
"""
Turn the FASTA that generate_thor.py writes into a table and a figure.

This is the Colab notebook's post-processing as a script, so a cluster run and
a notebook run produce the same CSV columns and the same plot. Import it:

    from summarise_variants import parse_variant_fasta, plot_variants

or run it over one or more variant FASTAs:

    python summarise_variants.py out/*/\\*_variants.fasta --combined all.csv
"""
import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # cluster nodes have no display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

NUMERIC_FIELDS = [
    "predicted_OGT", "G_global_nll", "G_sub_nll",
    "ESM_global_nll", "ESM_sub_nll", "Identity_to_WT(%)", "temp",
    "filter_threshold", "MPNN_nll",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
WARMER = "#eb6834"   # positive dT
COOLER = "#2a78d6"   # negative dT
SCORE_1 = "#2a78d6"  # ESM
SCORE_2 = "#1baf7a"  # generator
SCORE_3 = "#6250d6"  # ProteinMPNN


def parse_variant_fasta(path) -> pd.DataFrame:
    """Read the FASTA written by generate_thor.py into a DataFrame.

    Headers look like:
    >label | type=sampled | predicted_OGT=67.3 | ... | reverted_pos=M1L,M50V
    """
    rows, rec, seq = [], None, []

    def flush():
        if rec is not None:
            rec["seq"] = "".join(seq)
            rec["length"] = len(rec["seq"])
            rows.append(rec)

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                parts = [p.strip() for p in line[1:].split("|")]
                rec, seq = {"label": parts[0]}, []
                for part in parts[1:]:
                    if "=" in part:
                        key, val = part.split("=", 1)
                        rec[key.strip()] = val.strip()
            else:
                seq.append(line)
    flush()

    df = pd.DataFrame(rows)
    for col in NUMERIC_FIELDS:
        if col in df:
            df[col] = pd.to_numeric(df[col].replace("NA", pd.NA), errors="coerce")
    if "reverted_pos" in df:
        df["n_reverted"] = df["reverted_pos"].fillna("").apply(
            lambda s: len([m for m in s.split(",") if m])
        )
    df = df.rename(columns={"type": "kind", "Identity_to_WT(%)": "identity_to_wt"})

    # Predicted temperature shift relative to the wildtype of the same input.
    wt_ogt = df.loc[df["kind"] == "wt", "predicted_OGT"]
    df["delta_T"] = df["predicted_OGT"] - (wt_ogt.iloc[0] if len(wt_ogt) else float("nan"))

    # ProteinMPNN scores sit far from zero and close together, so the useful
    # quantity is the shift against the wildtype on the same backbone.
    if "MPNN_nll" in df:
        wt_mpnn = df.loc[df["kind"] == "wt", "MPNN_nll"]
        df["delta_MPNN"] = df["MPNN_nll"] - (wt_mpnn.iloc[0] if len(wt_mpnn) else float("nan"))
    return df


def short_label(label: str) -> str:
    """query1_variant_rep0_filtered -> rep0 filtered"""
    parts = label.split("_")
    if "variant" in parts:
        parts = parts[parts.index("variant") + 1:]
    elif len(parts) > 1:
        parts = parts[1:]
    return " ".join(parts)


def plot_variants(df: pd.DataFrame, out_path, sequence_id: str = "", sort_by: str = "delta_T"):
    """Horizontal bar panels: predicted shift plus the scores behind it."""
    plot_df = df[df["kind"] != "wt"].copy()
    if plot_df.empty:
        print(f"[warn] no variants to plot for {sequence_id or out_path}")
        return None
    plot_df = plot_df.sort_values(sort_by, ascending=False)  # best on top (y axis inverted)
    y = range(len(plot_df))
    labels = [short_label(l) for l in plot_df["label"]]

    panels = [
        ("delta_T", "Predicted ΔT vs wildtype  (°C)", None),
        ("ESM_sub_nll", "ESM NLL at substitutions  (lower = more plausible)", SCORE_1),
        ("G_sub_nll", "Generator NLL at substitutions  (lower = more likely)", SCORE_2),
        ("delta_MPNN", "ProteinMPNN NLL vs wildtype  (negative = fits the fold better)", SCORE_3),
    ]
    panels = [p for p in panels if p[0] in plot_df and plot_df[p[0]].notna().any()]

    fig, axes = plt.subplots(
        1, len(panels), sharey=True,
        figsize=(4.3 * len(panels), 0.42 * len(plot_df) + 2.2),
        facecolor=SURFACE,
    )
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (col, title, color) in zip(axes, panels):
        vals = plot_df[col].fillna(0.0).to_numpy()
        colors = [WARMER if v >= 0 else COOLER for v in vals] if color is None else color
        ax.barh(list(y), vals, height=0.62, color=colors, zorder=3)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
        ax.set_facecolor(SURFACE)
        ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
        if color is None or vals.min() < 0:
            ax.axvline(0, color=INK_SOFT, lw=1.0, zorder=4)

        span = max(abs(vals.max()), abs(vals.min())) or 1.0
        pad = span * 0.03
        for yi, v in zip(y, vals):
            ax.text(
                v + (pad if v >= 0 else -pad), yi, f"{v:,.1f}" if col == "delta_T" else f"{v:,.2f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=INK_SOFT,
            )
        lo, hi = min(0, vals.min()), max(0, vals.max())
        ax.set_xlim(lo - span * 0.18, hi + span * 0.18)

    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels, fontsize=9, color=INK)
    axes[0].invert_yaxis()

    wt_ogt = df.loc[df["kind"] == "wt", "predicted_OGT"]
    wt_txt = f"wildtype predicted OGT {wt_ogt.iloc[0]:.1f} °C" if len(wt_ogt) else "no wildtype record"
    fig.suptitle(
        f"THOR variants for {sequence_id}  ({wt_txt})",
        fontsize=12.5, color=INK, x=0.012, ha="left", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarise(
    fasta_paths: List[Path],
    combined_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    no_plot: bool = False,
) -> pd.DataFrame:
    """Write a CSV and a figure beside each FASTA, and optionally one combined CSV."""
    frames = []
    for fasta in fasta_paths:
        fasta = Path(fasta)
        # The run's id is the output prefix generate_thor.py was given.
        seq_id = fasta.stem[:-len("_variants")] if fasta.stem.endswith("_variants") else fasta.stem
        dest = Path(out_dir) if out_dir else fasta.parent
        dest.mkdir(parents=True, exist_ok=True)

        df = parse_variant_fasta(fasta)
        csv_path = dest / f"{seq_id}_variants.csv"
        df.to_csv(csv_path, index=False)
        kinds = df["kind"].value_counts().to_dict() if "kind" in df else {}
        print(f"[ok] {seq_id}: {len(df)} sequences {kinds} -> {csv_path}")

        if not no_plot:
            png = plot_variants(df, dest / f"{seq_id}_variants.png", sequence_id=seq_id)
            if png:
                print(f"[ok] {seq_id}: figure -> {png}")

        df.insert(0, "input_id", seq_id)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined_path and not combined.empty:
        Path(combined_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"[ok] combined table ({len(combined)} rows) -> {combined_path}")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fasta", nargs="+", type=Path, help="Variant FASTA(s) from generate_thor.py")
    parser.add_argument("--combined", type=Path, default=None,
                        help="Also write one CSV with every run stacked, keyed by input_id")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Where to write CSV/PNG (default: beside each FASTA)")
    parser.add_argument("--no_plot", action="store_true", help="Skip the figures")
    args = parser.parse_args()
    summarise(args.fasta, args.combined, args.out_dir, args.no_plot)


if __name__ == "__main__":
    main()
