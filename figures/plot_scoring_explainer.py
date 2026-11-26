#!/usr/bin/env python
"""Two slide figures explaining how variant_ranking.py scores a variant.

Figure 1 is the per-substitution layer: the two independent grounds on which a
model objects, drawn in the plane the decision is actually made in.
Figure 2 is the per-variant layer: how those objections turn into one number.

Both are drawn from a real run's `*_substitutions.csv` and `*_ranking.csv`, so
the point clouds are data rather than illustration.

    python plot_scoring_explainer.py BUNDLE_DIR [--out_dir .]
"""
import argparse
import math
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Matching variant_ranking.py.
BAD_LLR = -2.0
SOFTNESS = 1.0
WEIGHT_ONE = 0.35

INK = "#1c1b19"
INK_2 = "#55534e"
INK_3 = "#8c8a86"
RULE = "#d6d4d0"
C_OK = "#9b9894"
C_WORSE = "#2f6fb3"
C_IMPL = "#e08a1e"
C_BOTH = "#a3242a"
# Figure 2 only: the kept variants. Deliberately not C_WORSE, which means
# "worse than wildtype" in figure 1 and would read backwards here.
C_KEEP = "#2e7d4f"

FLAG_COLOUR = {"ok": C_OK, "worse_than_wt": C_WORSE,
               "implausible": C_IMPL, "worse+implausible": C_BOTH}
FLAG_LABEL = {"ok": "accepted",
              "worse_than_wt": "worse than wildtype",
              "implausible": "implausible",
              "worse+implausible": "both grounds"}


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.9,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.9)


def save(fig, out_dir: Path, stem: str):
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.18)
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figure 1 - the two grounds, in the plane where the decision is made
# ---------------------------------------------------------------------------
def panel_model(ax, subs, model, floor, show_examples=False):
    """p(wildtype residue) against p(substituted residue), both on log axes.

    In this plane the log-likelihood ratio is a diagonal - it only sees how far
    a point sits below the parity line - and the plausibility floor is a
    horizontal one. Drawn together, the gap between them is the thing the ratio
    alone cannot catch.
    """
    p_wt = subs[f"p_{model}_wt"].to_numpy(float)
    p_var = subs[f"p_{model}_var"].to_numpy(float)
    flags = subs[f"{model}_flag"].to_numpy()

    lo, hi = 2e-4, 1.0
    xg = np.logspace(math.log10(lo), math.log10(hi), 400)
    ratio = xg * math.exp(BAD_LLR)
    floor_line = np.full_like(xg, floor)
    lower = np.full_like(xg, lo)

    both = np.minimum(ratio, floor_line)
    ax.fill_between(xg, lower, both, color=C_BOTH, alpha=0.11, lw=0)
    ax.fill_between(xg, floor_line, ratio, where=ratio > floor_line,
                    color=C_WORSE, alpha=0.11, lw=0, interpolate=True)
    ax.fill_between(xg, ratio, floor_line, where=floor_line > ratio,
                    color=C_IMPL, alpha=0.11, lw=0, interpolate=True)

    ax.plot(xg, xg, ls=(0, (1, 2)), color=INK_3, lw=1.2, zorder=2)
    ax.plot(xg, ratio, color=C_WORSE, lw=2, zorder=3)
    ax.axhline(floor, color=C_IMPL, lw=2, zorder=3)

    # Draw the accepted points first so objections are never hidden under them.
    for flag in ("ok", "worse_than_wt", "implausible", "worse+implausible"):
        sel = flags == flag
        if not sel.any():
            continue
        ax.scatter(p_wt[sel], p_var[sel], s=26, zorder=4,
                   color=FLAG_COLOUR[flag], alpha=0.85,
                   linewidths=0.4, edgecolors="white")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("$p$(wildtype residue)")
    tidy(ax)

    ax.annotate("no change", xy=(0.0013, 0.0013), xycoords="data", color=INK_3,
                fontsize=9, rotation=45, ha="center", va="bottom",
                rotation_mode="anchor")
    ax.annotate(f"$\\mathrm{{llr}} = {BAD_LLR:.0f}$", xy=(0.78, 0.78 * math.exp(BAD_LLR)),
                color=C_WORSE, fontsize=9.5, rotation=45, ha="center", va="bottom",
                rotation_mode="anchor", fontweight="bold")
    ax.annotate(f"floor  $p = {floor:.3f}$", xy=(lo * 1.5, floor * 1.25),
                color=C_IMPL, fontsize=9.5, ha="left", va="bottom", fontweight="bold")

    if show_examples:
        # The case that motivated the floor: a big drop from a confident
        # wildtype is survivable, a small drop from an unconfident one is not.
        examples = (((0.5, 0.3), "accepted", "right", 0.80, 0.64),
                    ((0.1, 0.01), "rejected on both grounds", "left", 1.45, 0.52))
        for (x, y), verdict, ha, dx, dy in examples:
            llr = math.log(y) - math.log(x)
            ax.scatter([x], [y], s=92, facecolors="none", edgecolors=INK,
                       linewidths=1.6, zorder=6)
            ax.annotate(f"{x:g} $\\rightarrow$ {y:g},  "
                        f"$\\mathrm{{llr}} = {llr:+.2f}$\n{verdict}",
                        xy=(x, y), xytext=(x * dx, y * dy), fontsize=9,
                        color=INK, ha=ha, va="top", zorder=6, linespacing=1.4,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=RULE, lw=0.8, alpha=0.94))


def figure_one(subs, esm_floor, mpnn_floor, out_dir, stem):
    fig = plt.figure(figsize=(11.6, 7.4))
    # The panels, their legend and the equation band are placed in figure
    # fractions rather than stacked rows: `bbox_inches="tight"` crops whatever
    # slack a gridspec leaves, and the three blocks need different gaps.
    gs = fig.add_gridspec(1, 2, wspace=0.16,
                          top=0.98, bottom=0.40, left=0.07, right=0.98)
    ax_esm = fig.add_subplot(gs[0, 0])
    ax_mpnn = fig.add_subplot(gs[0, 1])

    panel_model(ax_esm, subs, "esm", esm_floor, show_examples=True)
    panel_model(ax_mpnn, subs, "mpnn", mpnn_floor)
    ax_esm.set_ylabel("$p$(substituted residue)")

    n = len(subs)
    for ax, model, title in ((ax_esm, "esm", "ESM-1v"),
                             (ax_mpnn, "mpnn", "ProteinMPNN")):
        bad = int((subs[f"{model}_flag"] != "ok").sum())
        ax.set_title(f"{title}\n", fontsize=13, color=INK, loc="left",
                     fontweight="bold", pad=10)
        ax.text(0, 1.015, f"objects to {bad} of {n} substitutions",
                transform=ax.transAxes, fontsize=10, color=INK_2, va="bottom")

    handles = [Line2D([], [], marker="o", ls="", markersize=7.5, color=FLAG_COLOUR[f],
                      markeredgecolor="white", markeredgewidth=0.5, label=FLAG_LABEL[f])
               for f in ("ok", "worse_than_wt", "implausible", "worse+implausible")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.345),
               ncol=4, fontsize=10.5, columnspacing=2.6, handletextpad=0.5,
               frameon=False)

    fig.add_artist(Line2D([0.02, 0.98], [0.262, 0.262], color=RULE, lw=0.9))
    fig.text(0.02, 0.212,
             r"$\mathrm{llr}_i \;=\; \ln p(\mathrm{substituted}_i)\;-\;"
             r"\ln p(\mathrm{wildtype}_i)$        in nats, per model, "
             r"at the position in the variant's own context",
             fontsize=12.5, color=INK, va="top")
    fig.text(0.02, 0.142,
             r"a model $\mathbf{objects}$ when   $\mathrm{llr}_i \leq -2$   "
             r"$\mathbf{or}$   $p(\mathrm{substituted}_i) < \mathrm{floor}$,      "
             r"$\mathrm{floor} = P_{5}\{\,p(\mathrm{wildtype}_i)\,\}$ over the wildtype",
             fontsize=12.5, color=INK, va="top")
    fig.text(0.02, 0.074,
             "The ratio is scale-free, so it cannot tell a survivable drop from a confident "
             "residue apart from a fatal one from an unconfident residue.\nThe floor can: it is "
             "set by the wildtype's own weakest positions, a sequence that demonstrably folds "
             "and works.",
             fontsize=10.5, color=INK_2, va="top", linespacing=1.5)

    return save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Figure 2 - from objections to one number
# ---------------------------------------------------------------------------
def panel_crossfade(ax, ranking):
    mg = np.linspace(-3.2, 3.2, 400)
    w = 1.0 / (1.0 + np.exp(-mg / SOFTNESS))

    ax.fill_between(mg, 0, 1, where=mg < 0, color=C_WORSE, alpha=0.07, lw=0)
    ax.fill_between(mg, 0, 1, where=mg >= 0, color=C_BOTH, alpha=0.07, lw=0)
    ax.axvline(0, color=RULE, lw=1)
    ax.plot(mg, w, color=INK, lw=2.4, zorder=4)

    m = ranking["mean_llr"].to_numpy(float)
    wv = ranking["count_weight"].to_numpy(float)
    sel = ranking["selected"].to_numpy(bool)
    ax.scatter(m[~sel], wv[~sel], s=58, facecolors="white", edgecolors=INK_3,
               linewidths=1.4, zorder=5)
    ax.scatter(m[sel], wv[sel], s=58, color=C_KEEP, edgecolors="white",
               linewidths=0.8, zorder=6)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean llr over the variant's substitutions,  $m$  (nats)")
    ax.set_ylabel("weight on the count of bad positions,  $w$")
    ax.set_yticks([0, 0.5, 1])
    tidy(ax)

    ax.text(-3.0, 0.90, "worse than wildtype on average\n"
                        "$\\rightarrow$ the average is the complaint",
            fontsize=9.5, color=C_WORSE, va="top", linespacing=1.5)
    ax.text(3.0, 0.22, "better than wildtype on average\n"
                       "$\\rightarrow$ the average says nothing more,\nthe bad positions do",
            fontsize=9.5, color=C_BOTH, va="top", ha="right", linespacing=1.5)
    ax.annotate("this run's 12 variants", xy=(float(np.median(m)), float(np.median(wv))),
                xytext=(-1.95, 0.29), fontsize=9.5, color=INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.9,
                                connectionstyle="arc3,rad=0.25"))


def panel_selection(ax, ranking, lam):
    risk = ranking["risk"].to_numpy(float)
    gain = ranking["delta_T"].to_numpy(float)
    sel = ranking["selected"].to_numpy(bool)
    labels = [s.replace("dsPETase05_variant_", "") for s in ranking["label"]]

    x0, x1 = 0, max(risk.max() * 1.18, 1)
    y0, y1 = min(gain.min() - 2.5, 0), gain.max() * 1.16
    xg = np.array([x0, x1])

    # score = delta_T - lambda * risk, so a constant score is a line of slope
    # lambda: everything above it is preferred, everything below rejected.
    for s in np.arange(-10, 16, 2.5):
        ax.plot(xg, s + lam * xg, color=RULE, lw=0.9, zorder=1)
    cut = float(ranking.loc[sel, "score"].min())
    ax.plot(xg, cut + lam * xg, color=INK, lw=1.8, ls=(0, (5, 3)), zorder=3)

    ax.scatter(risk[~sel], gain[~sel], s=95, facecolors="white", edgecolors=INK_3,
               linewidths=1.5, zorder=5)
    ax.scatter(risk[sel], gain[sel], s=95, color=C_KEEP, edgecolors="white",
               linewidths=0.9, zorder=6)
    # Only the kept variants are named; twelve labels in this space collide, and
    # the rejected ones are not what the slide is about.
    kept = sorted(((x, y, lab) for x, y, lab, k in zip(risk, gain, labels, sel) if k),
                  key=lambda t: t[0])
    # The kept variants cluster, so each label gets a leader line and the
    # placements cycle through four directions.
    places = [(26, 22, "left", "bottom"), (-26, -26, "right", "top"),
              (26, -26, "left", "top"), (-26, 22, "right", "bottom")]
    for i, (x, y, lab) in enumerate(kept):
        dx, dy, ha, va = places[i % len(places)]
        ax.annotate(lab, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, ha=ha, va=va, color=INK, zorder=7,
                    arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.8,
                                    shrinkA=1, shrinkB=5))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("risk   (bad positions, both models counting hardest)")
    ax.set_ylabel("predicted temperature shift,  $\\Delta T$  (°C)")
    tidy(ax)

    ax.text(x1 * 0.045, y1 * 0.98,
            "grey lines: equal score\n"
            f"slope $\\lambda = {lam:.2f}$ °C per unit of risk",
            fontsize=9.5, color=INK_2, va="top", linespacing=1.5)
    ax.annotate("the cut: everything above it is kept",
                xy=(x1 * 0.12, cut + lam * x1 * 0.12),
                xytext=(x1 * 0.30, y0 + (y1 - y0) * 0.055), fontsize=9.5,
                color=INK, va="bottom", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))


def figure_two(ranking, lam, tradeoff_note, out_dir, stem):
    fig = plt.figure(figsize=(11.6, 7.0))
    gs = fig.add_gridspec(1, 2, wspace=0.26, top=0.96, bottom=0.33,
                          left=0.07, right=0.98)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_crossfade(ax_a, ranking)
    panel_selection(ax_b, ranking, lam)

    ax_a.set_title("1.  How much the count matters\n", fontsize=13, color=INK,
                   loc="left", fontweight="bold", pad=10)
    ax_a.text(0, 1.015, "an average hides three ruinous substitutions "
                        "among thirty harmless ones",
              transform=ax_a.transAxes, fontsize=10, color=INK_2, va="bottom")
    ax_b.set_title("2.  What the gain costs\n", fontsize=13, color=INK,
                   loc="left", fontweight="bold", pad=10)
    ax_b.text(0, 1.015, tradeoff_note, transform=ax_b.transAxes,
              fontsize=10, color=INK_2, va="bottom")

    fig.add_artist(Line2D([0.02, 0.98], [0.250, 0.250], color=RULE, lw=0.9))
    fig.text(0.02, 0.200,
             r"$w \;=\; \frac{1}{1+e^{-m/s}}$         "
             r"$\mathrm{risk} \;=\; (1-w)\,\max(0,\,-m)\;+\;"
             r"w\left(n_{\mathrm{both}} + 0.35\,n_{\mathrm{one}}\right)$         "
             r"$\mathbf{score} \;=\; \Delta T \;-\; \lambda\;\mathrm{risk}$",
             fontsize=13, color=INK, va="top")
    fig.text(0.02, 0.125,
             "The first term of the risk is the average, the second the count of "
             "individually bad positions; $w$ hands over from one to the other.",
             fontsize=10.5, color=INK_2, va="top")
    fig.text(0.02, 0.072,
             r"$\lambda \;=\; \mathrm{tradeoff}\,\times\,"
             r"\mathrm{spread}(\Delta T)\,/\,\mathrm{spread}(\mathrm{risk})$"
             "  — the two have no common scale, so the exchange rate comes from the "
             "run's own 5–95 spread in each.\nThe score is then in degrees: the "
             "predicted gain, minus what that gain costs in model-implausibility.",
             fontsize=10.5, color=INK_2, va="top", linespacing=1.6)
    return save(fig, out_dir, stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path, help="Directory holding *_ranking.csv and *_substitutions.csv")
    ap.add_argument("--out_dir", type=Path, default=Path("."))
    args = ap.parse_args()

    ranking = pd.read_csv(sorted(args.bundle.glob("*_ranking.csv"))[0])
    subs = pd.read_csv(sorted(args.bundle.glob("*_substitutions.csv"))[0])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    esm_floor = float(ranking["esm_floor"].iloc[0])
    mpnn_floor = float(ranking["mpnn_floor"].iloc[0])
    lam = float(ranking["penalty_per_risk"].iloc[0])
    note = (f"{int(ranking['selected'].sum())} of {len(ranking)} variants kept "
            f"(filled); open circles were not")

    style()
    written = figure_one(subs, esm_floor, mpnn_floor, args.out_dir,
                         "thor_scoring_1_substitutions")
    written += figure_two(ranking, lam, note, args.out_dir,
                          "thor_scoring_2_variants")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
