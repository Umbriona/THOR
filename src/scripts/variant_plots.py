#!/usr/bin/env python
"""Figures for the variant scoring in `variant_ranking.py`.

Two views, because the scoring has two layers:

`substitution_plane` is one variant's substitutions, plotted as
p(wildtype residue) against p(substituted residue) under each model. In that
plane the log-likelihood ratio is a diagonal and the plausibility floor is a
horizontal, so the two grounds on which a model objects are two different cuts
and the gap between them is visible.

`score_overview` is the whole run: where each variant sits in the crossfade
between the average and the count, and in the temperature-against-risk plane
where the selection is made.

Both return a matplotlib figure and draw nothing by themselves, so the notebooks
can show them and `save_planes` can write them into a bundle.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

INK = "#1c1b19"
INK_2 = "#55534e"
INK_3 = "#8c8a86"
RULE = "#d6d4d0"
C_OK = "#9b9894"
C_WORSE = "#2f6fb3"
C_IMPL = "#e08a1e"
C_BOTH = "#a3242a"
# Kept variants in `score_overview`. Deliberately not C_WORSE, which means
# "worse than wildtype" in the plane and would read backwards here.
C_KEEP = "#2e7d4f"

FLAG_COLOUR = {"ok": C_OK, "worse_than_wt": C_WORSE,
               "implausible": C_IMPL, "worse+implausible": C_BOTH, "n/a": C_OK}
FLAG_LABEL = {"ok": "accepted", "worse_than_wt": "worse than wildtype",
              "implausible": "implausible", "worse+implausible": "both grounds"}

MODELS = (("esm", "ESM-1v"), ("mpnn", "ProteinMPNN"))


def _tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.9, colors=INK_2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(RULE)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _plane_panel(ax, rows, model, floor, bad_llr, annotate, lo):
    p_wt = np.array([r[f"p_{model}_wt"] for r in rows], dtype=float)
    p_var = np.array([r[f"p_{model}_var"] for r in rows], dtype=float)
    flags = [r[f"{model}_flag"] for r in rows]

    hi = 1.0
    xg = np.logspace(math.log10(lo), 0.0, 400)
    ratio = xg * math.exp(bad_llr)
    lower = np.full_like(xg, lo)

    if floor is not None:
        floor_line = np.full_like(xg, floor)
        ax.fill_between(xg, lower, np.minimum(ratio, floor_line),
                        color=C_BOTH, alpha=0.11, lw=0)
        ax.fill_between(xg, floor_line, ratio, where=ratio > floor_line,
                        color=C_WORSE, alpha=0.11, lw=0, interpolate=True)
        ax.fill_between(xg, ratio, floor_line, where=floor_line > ratio,
                        color=C_IMPL, alpha=0.11, lw=0, interpolate=True)
        ax.axhline(floor, color=C_IMPL, lw=2, zorder=3)
        ax.annotate(f"floor  $p = {floor:.3f}$", xy=(lo * 1.5, floor * 1.3),
                    color=C_IMPL, fontsize=9, ha="left", va="bottom", fontweight="bold")
    else:
        ax.fill_between(xg, lower, ratio, color=C_WORSE, alpha=0.11, lw=0)

    ax.plot(xg, xg, ls=(0, (1, 2)), color=INK_3, lw=1.2, zorder=2)
    ax.plot(xg, ratio, color=C_WORSE, lw=2, zorder=3)

    # Accepted points first, so an objection is never hidden underneath one.
    order = ["ok", "n/a", "worse_than_wt", "implausible", "worse+implausible"]
    for flag in order:
        sel = np.array([f == flag for f in flags])
        if not sel.any():
            continue
        ax.scatter(p_wt[sel], p_var[sel], s=46, zorder=4, color=FLAG_COLOUR[flag],
                   alpha=0.9, linewidths=0.5, edgecolors="white")

    # The positions both models reject are what the repair step targets, so
    # ring them in each panel even where only one model objected here.
    both = np.array([r.get("disagrees_with") == "both" for r in rows])
    if both.any():
        ax.scatter(p_wt[both], p_var[both], s=150, facecolors="none",
                   edgecolors=INK, linewidths=1.1, zorder=5)

    if annotate:
        # Name the objections, the repair targets first, then the worst of the
        # rest, and only as many as stay legible.
        bad = [(r, p, b) for r, f, p, b in zip(rows, flags, p_var, both)
               if f not in ("ok", "n/a")]
        bad.sort(key=lambda t: (not t[2], t[1]))
        places = [(7, 6, "left", "bottom"), (-7, -8, "right", "top"),
                  (7, -8, "left", "top"), (-7, 6, "right", "bottom")]
        for i, (r, p, _) in enumerate(bad[:annotate]):
            dx, dy, ha, va = places[i % len(places)]
            ax.annotate(r["substitution"], xy=(r[f"p_{model}_wt"], p),
                        xytext=(dx, dy), textcoords="offset points", fontsize=8,
                        ha=ha, va=va, color=INK, zorder=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("$p$(wildtype residue)")
    _tidy(ax)

    corner = lo * 2.4
    ax.annotate("no change", xy=(corner, corner), color=INK_3, fontsize=8.5,
                rotation=45, ha="center", va="bottom", rotation_mode="anchor")
    ax.annotate(f"$\\mathrm{{llr}} = {bad_llr:g}$", xy=(0.78, 0.78 * math.exp(bad_llr)),
                color=C_WORSE, fontsize=9, rotation=45, ha="center", va="bottom",
                rotation_mode="anchor", fontweight="bold")


def substitution_plane(rows: Sequence[Dict], label: str,
                       esm_floor: Optional[float] = None,
                       mpnn_floor: Optional[float] = None,
                       bad_llr: float = -2.0,
                       metrics: Optional[Dict] = None,
                       annotate: int = 10,
                       figsize=None):
    """One variant's substitutions, in the plane where each model decides.

    `rows` are that variant's rows from `variant_ranking.substitution_rows`;
    `metrics` is its entry from `variant_ranking.rank`, used only for the
    subtitle. Pass `annotate=0` to leave the objections unnamed.
    """
    rows = [r for r in rows if r.get("label", label) == label]
    if not rows:
        raise ValueError(f"No substitution rows for {label}")

    panels = [(m, title, floor) for (m, title), floor
              in zip(MODELS, (esm_floor, mpnn_floor))
              if any(r.get(f"p_{m}_var") is not None for r in rows)]
    if not panels:
        raise ValueError(f"No model probabilities for {label}")

    # A shared lower bound keeps the two panels comparable.
    seen = [r[f"p_{m}_{which}"] for r in rows for m, _, _ in panels
            for which in ("var", "wt") if r.get(f"p_{m}_{which}") is not None]
    floors = [f for _, _, f in panels if f is not None]
    lo = min(list(seen) + floors) * 0.4 if seen else 1e-4
    lo = min(max(lo, 1e-6), 1e-2)

    figsize = figsize or (5.6 * len(panels) + 0.6, 6.1)
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    axes = np.atleast_1d(axes)

    for ax, (model, title, floor) in zip(axes, panels):
        _plane_panel(ax, rows, model, floor, bad_llr, annotate, lo)
        n_bad = sum(1 for r in rows if r[f"{model}_flag"] not in ("ok", "n/a"))
        ax.set_title(f"{title}\n", fontsize=12, color=INK, loc="left",
                     fontweight="bold", pad=8)
        ax.text(0, 1.012, f"objects to {n_bad} of {len(rows)} substitutions",
                transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")
    axes[0].set_ylabel("$p$(substituted residue)")

    head = label
    if metrics:
        gain = "n/a" if metrics.get("delta_T") is None else f"{metrics['delta_T']:+.1f} °C"
        head = (f"{label}        score {metrics['score']:+.2f}        "
                f"$\\Delta T$ {gain}        bad: both {int(metrics['n_bad_both'])}, "
                f"one {int(metrics['n_bad_one'])}")
    fig.suptitle(head, fontsize=12.5, color=INK, x=0.012, ha="left", y=0.995)

    handles = [Line2D([], [], marker="o", ls="", markersize=7, color=FLAG_COLOUR[f],
                      markeredgecolor="white", markeredgewidth=0.5, label=FLAG_LABEL[f])
               for f in ("ok", "worse_than_wt", "implausible", "worse+implausible")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5,
               columnspacing=2.2, handletextpad=0.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.09, right=0.98, wspace=0.18)
    return fig


def score_overview(metrics: Sequence[Dict], selected: Optional[Sequence[str]] = None,
                   softness: float = 1.0, label_prefix: str = "",
                   max_labels: int = 8, figsize=(11.6, 5.2)):
    """Every variant of the run: the crossfade, and the plane it is ranked in."""
    metrics = list(metrics)
    if not metrics:
        raise ValueError("No variants to plot")
    chosen = set(selected if selected is not None
                 else [d["label"] for d in metrics if d.get("selected")])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)

    # ---- the crossfade -----------------------------------------------------
    mg = np.linspace(-3.2, 3.2, 400)
    ax_a.fill_between(mg, 0, 1, where=mg < 0, color=C_WORSE, alpha=0.07, lw=0)
    ax_a.fill_between(mg, 0, 1, where=mg >= 0, color=C_BOTH, alpha=0.07, lw=0)
    ax_a.axvline(0, color=RULE, lw=1)
    ax_a.plot(mg, 1.0 / (1.0 + np.exp(-mg / softness)), color=INK, lw=2.2, zorder=4)

    m = np.array([d["mean_llr"] if d["mean_llr"] is not None else 0.0 for d in metrics])
    w = np.array([d["count_weight"] for d in metrics], dtype=float)
    keep = np.array([d["label"] in chosen for d in metrics])
    ax_a.scatter(m[~keep], w[~keep], s=52, facecolors="white", edgecolors=INK_3,
                 linewidths=1.3, zorder=5)
    ax_a.scatter(m[keep], w[keep], s=52, color=C_KEEP, edgecolors="white",
                 linewidths=0.8, zorder=6)

    ax_a.set_xlim(-3.2, 3.2)
    ax_a.set_ylim(0, 1)
    ax_a.set_yticks([0, 0.5, 1])
    ax_a.set_xlabel("mean llr over the substitutions,  $m$  (nats)")
    ax_a.set_ylabel("weight on the count of bad positions,  $w$")
    _tidy(ax_a)
    ax_a.text(-3.0, 0.93, "worse than wildtype\n$\\rightarrow$ the average is\nthe complaint",
              fontsize=9, color=C_WORSE, va="top", linespacing=1.5)
    ax_a.text(3.0, 0.30, "better than wildtype\n$\\rightarrow$ the bad positions\nare the complaint",
              fontsize=9, color=C_BOTH, va="top", ha="right", linespacing=1.5)
    ax_a.set_title("How much the count matters", fontsize=12, color=INK,
                   loc="left", fontweight="bold", pad=8)

    # ---- the plane the ranking happens in ----------------------------------
    lam = float(metrics[0].get("penalty_per_risk") or 0.0)
    risk = np.array([d["risk"] for d in metrics], dtype=float)
    gain = np.array([d["delta_T"] if d["delta_T"] is not None else 0.0
                     for d in metrics], dtype=float)

    x1 = max(risk.max() * 1.20, 1.0)
    y0 = min(gain.min() - 2.0, 0.0)
    y1 = max(gain.max() * 1.18, y0 + 1.0)
    xg = np.array([0.0, x1])
    step = max(1.0, round((y1 - y0) / 7.0))
    for s in np.arange(math.floor(y0 - lam * x1), y1 + step, step):
        ax_b.plot(xg, s + lam * xg, color=RULE, lw=0.9, zorder=1)
    if chosen:
        cut = min(d["score"] for d in metrics if d["label"] in chosen)
        ax_b.plot(xg, cut + lam * xg, color=INK, lw=1.7, ls=(0, (5, 3)), zorder=3)

    ax_b.scatter(risk[~keep], gain[~keep], s=82, facecolors="white", edgecolors=INK_3,
                 linewidths=1.4, zorder=5)
    ax_b.scatter(risk[keep], gain[keep], s=82, color=C_KEEP, edgecolors="white",
                 linewidths=0.9, zorder=6)

    # Only the kept variants are named, with leader lines, since they cluster.
    places = [(24, 20, "left", "bottom"), (-24, -24, "right", "top"),
              (24, -24, "left", "top"), (-24, 20, "right", "bottom")]
    named = sorted(((x, y, d["label"]) for x, y, d, k in zip(risk, gain, metrics, keep) if k),
                   key=lambda t: t[0])[:max_labels]
    for i, (x, y, lab) in enumerate(named):
        dx, dy, ha, va = places[i % len(places)]
        short = lab[len(label_prefix):] if label_prefix and lab.startswith(label_prefix) else lab
        ax_b.annotate(short, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                      fontsize=8.5, ha=ha, va=va, color=INK, zorder=7,
                      arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.8,
                                      shrinkA=1, shrinkB=5))

    ax_b.set_xlim(0, x1)
    ax_b.set_ylim(y0, y1)
    ax_b.set_xlabel("risk   (bad positions, both models counting hardest)")
    ax_b.set_ylabel("predicted temperature shift,  $\\Delta T$  (°C)")
    _tidy(ax_b)
    ax_b.text(x1 * 0.04, y1 * 0.98 if y1 > 0 else y1,
              f"grey lines: equal score\nslope $\\lambda = {lam:.2f}$ °C per unit of risk",
              fontsize=9, color=INK_2, va="top", linespacing=1.5)
    ax_b.set_title("What the gain costs", fontsize=12, color=INK,
                   loc="left", fontweight="bold", pad=8)
    if chosen:
        ax_b.text(x1 * 0.985, y0 + (y1 - y0) * 0.03,
                  f"{len(chosen)} kept (filled), above the dashed cut",
                  fontsize=9, color=C_KEEP, ha="right", fontweight="bold")

    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.07, right=0.98, wspace=0.26)
    return fig


def save_planes(sub_rows: Sequence[Dict], labels: Sequence[str], out_dir: Path,
                esm_floor=None, mpnn_floor=None, bad_llr: float = -2.0,
                metrics: Optional[Sequence[Dict]] = None, dpi: int = 150,
                **kwargs) -> List[Path]:
    """One `substitution_plane` per label, written as PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_label: Dict[str, List[Dict]] = {}
    for row in sub_rows:
        by_label.setdefault(row["label"], []).append(row)
    by_metric = {d["label"]: d for d in (metrics or [])}

    written = []
    for label in labels:
        rows = by_label.get(label)
        if not rows:
            continue
        fig = substitution_plane(rows, label, esm_floor, mpnn_floor, bad_llr,
                                 metrics=by_metric.get(label), **kwargs)
        path = out_dir / f"{label}_substitution_plane.png"
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
