#!/usr/bin/env python
"""
Rank THOR variants, and say which substitutions the scoring models dislike.

Three numbers describe a variant: the predicted temperature shift, and how
plausible ESM and ProteinMPNN find the residues it put in. The first is what you
want; the other two are what you risk to get it.

Per substituted position, each model gives a log-likelihood ratio against the
wildtype residue in the same position and context:

    llr = log p(variant residue) - log p(wildtype residue)

Negative means the model would rather have kept the wildtype. Averaging those
over a variant's substitutions says whether it is better or worse than the
wildtype overall - but an average hides the case that matters, a variant that
looks fine on average because thirty harmless substitutions outvote three
ruinous ones.

So the penalty moves between two readings as the variant improves:

    m     = mean llr over substitutions, averaged across the models (nats)
    w     = 1 / (1 + exp(-m / softness))      0 when far worse than wildtype, 1 when better
    risk  = (1 - w) * max(0, -m)  +  w * (n_bad_both + 0.35 * n_bad_one)
    score = delta_T - penalty * risk

While the variant is worse than the wildtype on average, the average is the
problem and drives the penalty. Once it reaches parity the average stops being
informative and the count of individually bad positions takes over, with the
ones both models reject counting hardest.

`penalty` is not a constant. Predicted shifts range over tens of degrees while
risk ranges over a handful of substitutions, so any fixed factor would let one
term decide the ranking by itself; instead it is set from the run's own spread
in each, via `tradeoff` - the share of the temperature spread that the whole
risk spread is worth. `score` is then in degrees: the predicted gain minus what
that gain costs in model-implausibility.

As a script, over a THOR output directory or an unpacked bundle:

    python variant_ranking.py OUTPUT_DIR --top 5

writes `<name>_ranking.csv`, `<name>_substitutions.csv` and
`<name>_selected.fasta`.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# A substitution is "bad" under a model when it is this much less likely than
# the wildtype residue, in nats. -2.0 is about a sevenfold drop.
BAD_LLR = -2.0
# Both models rejecting a position is worse than one; see the module docstring.
WEIGHT_ONE = 0.35
# Nats of mean llr over which the penalty shifts from the average to the counts.
SOFTNESS = 1.0
# How much of the run's spread in predicted temperature the run's spread in risk
# is worth. 0 ranks on temperature alone, 1 weighs the two equally, above 1
# prefers a plausible variant to a hot one. The degrees-per-risk factor is
# derived from this and the run's own spread, because the two quantities have no
# common scale: predicted shifts here range over tens of degrees while risk
# ranges over a handful of substitutions, so any fixed factor would make one
# term decide the ranking on its own.
TRADEOFF = 0.5


def _lookup(probs: Optional[np.ndarray], pos: int, aa: str,
            aa_order: str = AA20, floor: float = 1e-9) -> Optional[float]:
    """p(aa) at this position, or None when the model has nothing to say."""
    if probs is None or pos >= len(probs) or aa not in aa_order:
        return None
    return max(float(probs[pos][aa_order.index(aa)]), floor)


def _llr(probs: Optional[np.ndarray], pos: int, var_aa: str, wt_aa: str,
         aa_order: str = AA20) -> Optional[float]:
    p_var = _lookup(probs, pos, var_aa, aa_order)
    p_wt = _lookup(probs, pos, wt_aa, aa_order)
    if p_var is None or p_wt is None:
        return None
    return math.log(p_var) - math.log(p_wt)


def substitution_rows(label: str, var_seq: str, wt_seq: str,
                      esm_var: Optional[np.ndarray] = None,
                      esm_wt: Optional[np.ndarray] = None,
                      mpnn: Optional[np.ndarray] = None,
                      bad_llr: float = BAD_LLR,
                      aa_order: str = AA20) -> List[Dict]:
    """One row per substituted position, with each model's verdict on it."""
    rows = []
    for pos, (var_aa, wt_aa) in enumerate(zip(var_seq, wt_seq)):
        if var_aa == wt_aa:
            continue
        esm_llr = _llr(esm_var, pos, var_aa, wt_aa, aa_order)
        mpnn_llr = _llr(mpnn, pos, var_aa, wt_aa, aa_order)
        bad_esm = esm_llr is not None and esm_llr < bad_llr
        bad_mpnn = mpnn_llr is not None and mpnn_llr < bad_llr
        if bad_esm and bad_mpnn:
            verdict = "both"
        elif bad_esm:
            verdict = "esm"
        elif bad_mpnn:
            verdict = "proteinmpnn"
        else:
            verdict = "ok"
        rows.append({
            "label": label,
            "position": pos + 1,
            "wt_aa": wt_aa,
            "var_aa": var_aa,
            "substitution": f"{wt_aa}{pos + 1}{var_aa}",
            "p_esm_var": _lookup(esm_var, pos, var_aa, aa_order),
            "p_esm_wt": _lookup(esm_var, pos, wt_aa, aa_order),
            "esm_llr": esm_llr,
            # The same substitution judged in the wildtype's context: a gap
            # between the two is other substitutions rescuing or spoiling it.
            "esm_llr_wt_context": _llr(esm_wt, pos, var_aa, wt_aa, aa_order),
            "p_mpnn_var": _lookup(mpnn, pos, var_aa, aa_order),
            "p_mpnn_wt": _lookup(mpnn, pos, wt_aa, aa_order),
            "mpnn_llr": mpnn_llr,
            "disagrees_with": verdict,
        })
    return rows


def variant_metrics(rows: Sequence[Dict], delta_t: Optional[float],
                    softness: float = SOFTNESS,
                    weight_one: float = WEIGHT_ONE,
                    use_delta_t: bool = True,
                    use_esm: bool = True,
                    use_mpnn: bool = True) -> Dict:
    """Aggregate one variant's substitution rows into the ranking metrics.

    Everything here is per-variant. `score` needs the whole run, because the
    weight between temperature and risk comes from their spreads, so it is added
    by `rank`.
    """
    esm = [r["esm_llr"] for r in rows if r["esm_llr"] is not None] if use_esm else []
    mpnn = [r["mpnn_llr"] for r in rows if r["mpnn_llr"] is not None] if use_mpnn else []

    means = [float(np.mean(v)) for v in (esm, mpnn) if v]
    m = float(np.mean(means)) if means else 0.0

    n_bad_both = n_bad_one = 0
    for r in rows:
        bad_esm = use_esm and r["esm_llr"] is not None and r["esm_llr"] < BAD_LLR
        bad_mpnn = use_mpnn and r["mpnn_llr"] is not None and r["mpnn_llr"] < BAD_LLR
        if bad_esm and bad_mpnn:
            n_bad_both += 1
        elif bad_esm or bad_mpnn:
            n_bad_one += 1

    # Far below the wildtype the average is the complaint; at or above it the
    # average says nothing more and the individually bad positions take over.
    w = 1.0 / (1.0 + math.exp(-m / softness)) if means else 1.0
    risk = (1.0 - w) * max(0.0, -m) + w * (n_bad_both + weight_one * n_bad_one)

    gain = float(delta_t) if (use_delta_t and delta_t is not None
                              and not (isinstance(delta_t, float) and math.isnan(delta_t))) else 0.0

    return {
        "n_substitutions": len(rows),
        "esm_mean_llr": float(np.mean(esm)) if esm else None,
        "mpnn_mean_llr": float(np.mean(mpnn)) if mpnn else None,
        "mean_llr": m if means else None,
        "n_bad_both": n_bad_both,
        "n_bad_one": n_bad_one,
        "worst_substitution": min(
            (r for r in rows if r["esm_llr"] is not None or r["mpnn_llr"] is not None),
            key=lambda r: min(x for x in (r["esm_llr"], r["mpnn_llr"]) if x is not None),
            default={"substitution": None})["substitution"],
        "count_weight": round(w, 4),
        "risk": round(risk, 4),
        "delta_T": gain if use_delta_t else None,
    }


def _spread(values: Sequence[float]) -> float:
    """The 5th-to-95th percentile span, which outliers cannot stretch."""
    if len(values) < 2:
        return 0.0
    lo, hi = np.percentile(np.asarray(values, dtype=float), [5, 95])
    return float(hi - lo)


def score_metrics(metrics: List[Dict], tradeoff: float = TRADEOFF,
                  penalty_per_risk: Optional[float] = None) -> float:
    """Add `score` to each variant, in degrees, and return the factor used.

    `penalty_per_risk` overrides the trade-off with an explicit degrees-per-risk
    figure, for comparing runs on one fixed scale.
    """
    risks = [d["risk"] for d in metrics]
    gains = [d["delta_T"] or 0.0 for d in metrics]
    if penalty_per_risk is None:
        risk_spread = _spread(risks)
        # Without a spread in risk there is nothing to trade off, and without one
        # in temperature the risk is all there is to rank on.
        gain_spread = _spread(gains) or 1.0
        penalty_per_risk = tradeoff * gain_spread / risk_spread if risk_spread else 0.0
    for d in metrics:
        d["penalty_per_risk"] = round(penalty_per_risk, 4)
        d["score"] = round((d["delta_T"] or 0.0) - penalty_per_risk * d["risk"], 4)
    return penalty_per_risk


def rank(by_label: Dict[str, Dict], wt_seq: str, wt_label: str,
         esm: Dict[str, np.ndarray], mpnn: Optional[np.ndarray],
         delta_t: Dict[str, Optional[float]],
         tradeoff: float = TRADEOFF, penalty_per_risk: Optional[float] = None,
         bad_llr: float = BAD_LLR, **kwargs):
    """Metrics and substitution rows for every variant, best score first."""
    sub_rows, metrics = [], []
    esm_wt = esm.get(wt_label)
    for label, rec in by_label.items():
        if label == wt_label:
            continue
        rows = substitution_rows(label, rec["seq"], wt_seq, esm.get(label), esm_wt, mpnn,
                                 bad_llr=bad_llr)
        sub_rows.extend(rows)
        metrics.append({"label": label, **variant_metrics(rows, delta_t.get(label), **kwargs)})
    score_metrics(metrics, tradeoff=tradeoff, penalty_per_risk=penalty_per_risk)
    metrics.sort(key=lambda d: d["score"], reverse=True)
    for i, d in enumerate(metrics, 1):
        d["rank"] = i
    return metrics, sub_rows


# --------------------------------------------------------------------------
# Reading a THOR output directory, for use as a script
# --------------------------------------------------------------------------
def load_bundle(data_dir: Path):
    """Sequences and probabilities from an output directory or unpacked bundle."""
    fastas = sorted(data_dir.rglob("*_variants.fasta"))
    if not fastas:
        raise FileNotFoundError(f"No *_variants.fasta under {data_dir}")
    records, label, meta, seq = [], None, {}, []

    def flush():
        if label is not None:
            records.append({"label": label, "seq": "".join(seq), **meta})

    for line in fastas[0].read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            parts = [p.strip() for p in line[1:].split("|")]
            label, seq, meta = parts[0], [], {}
            for part in parts[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    meta[k.strip()] = v.strip()
        else:
            seq.append(line)
    flush()

    esm, mpnn = {}, None
    for f in data_dir.rglob("*_scoring_probs.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            probs = np.asarray(d["probs"], dtype=np.float32)
            which = d.get("Scoring_Model", d.get("model"))
            if which == "esm":
                esm[d["label"]] = probs
            elif which == "proteinmpnn":
                mpnn = probs

    wt = next((r for r in records if r.get("type") == "wt"), records[0])
    return records, wt, esm, mpnn, delta_t_map(records, wt), fastas[0]


def _as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def delta_t_map(records: Sequence[Dict], wt: Dict) -> Dict[str, Optional[float]]:
    """Predicted temperature shift against the wildtype, per label.

    Records may come from a parsed FASTA, where everything is a string, or from
    the table the notebooks build, where it is already numeric.
    """
    wt_ogt = _as_float(wt.get("predicted_OGT"))
    out = {}
    for r in records:
        ogt = _as_float(r.get("predicted_OGT"))
        out[r["label"]] = None if (ogt is None or wt_ogt is None) else ogt - wt_ogt
    return out


def write_csv(path: Path, rows: Sequence[Dict]):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_selection(out_dir: Path, prefix: str, wt: Dict, by_label: Dict[str, Dict],
                    metrics: List[Dict], sub_rows: List[Dict],
                    chosen: Sequence[str]) -> List[Path]:
    """The three files a selection produces, in one place for every caller.

    The CSVs cover every variant with a `selected` column, not only the chosen
    ones, so the choice can be revisited without re-running anything.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = list(chosen)
    for d in metrics:
        d["selected"] = d["label"] in chosen
    for r in sub_rows:
        r["selected"] = r["label"] in chosen

    written = []
    ranking_path = out_dir / f"{prefix}_ranking.csv"
    write_csv(ranking_path, metrics)
    written.append(ranking_path)
    if sub_rows:
        subs_path = out_dir / f"{prefix}_substitutions.csv"
        write_csv(subs_path, sub_rows)
        written.append(subs_path)

    if chosen:
        fasta_path = out_dir / f"{prefix}_selected.fasta"
        with open(fasta_path, "w") as fh:
            fh.write(f">{wt['label']} | type=wt\n{wt['seq']}\n")
            for label in chosen:
                d = next(x for x in metrics if x["label"] == label)
                gain = "NA" if d["delta_T"] is None else f"{d['delta_T']:.4g}"
                fh.write(f">{label} | score={d['score']} | delta_T={gain} | "
                         f"n_bad_both={d['n_bad_both']} | n_bad_one={d['n_bad_one']}\n"
                         f"{by_label[label]['seq']}\n")
        written.append(fasta_path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="THOR output directory or unpacked bundle")
    parser.add_argument("--top", type=int, default=5, help="How many variants to save (0 = none)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Save these labels instead of the top-scoring ones")
    parser.add_argument("--bad_llr", type=float, default=BAD_LLR,
                        help="Log-likelihood ratio below which a substitution counts as bad")
    parser.add_argument("--tradeoff", type=float, default=TRADEOFF,
                        help="How much of the run's temperature spread its risk spread is worth: "
                             "0 ranks on temperature alone, 1 weighs them equally")
    parser.add_argument("--penalty_per_risk", type=float, default=None,
                        help="Fixed degrees per unit of risk, instead of --tradeoff")
    parser.add_argument("--softness", type=float, default=SOFTNESS)
    parser.add_argument("--no_delta_t", action="store_true", help="Rank on the models alone")
    parser.add_argument("--no_esm", action="store_true")
    parser.add_argument("--no_mpnn", action="store_true")
    parser.add_argument("--out_dir", type=Path, default=None, help="Default: data_dir")
    args = parser.parse_args()

    records, wt, esm, mpnn, delta_t, fasta = load_bundle(args.data_dir)
    by_label = {r["label"]: r for r in records}
    metrics, sub_rows = rank(
        by_label, wt["seq"], wt["label"], esm, mpnn, delta_t,
        bad_llr=args.bad_llr, tradeoff=args.tradeoff,
        penalty_per_risk=args.penalty_per_risk, softness=args.softness,
        use_delta_t=not args.no_delta_t, use_esm=not args.no_esm, use_mpnn=not args.no_mpnn)

    chosen = (list(args.only) if args.only
              else [d["label"] for d in metrics[:max(0, args.top)]])
    missing = [c for c in chosen if c not in by_label]
    if missing:
        parser.error(f"--only names labels not in {fasta.name}: {missing}")
    out_dir = args.out_dir or args.data_dir
    prefix = fasta.name.split("_variants")[0]
    written = write_selection(out_dir, prefix, wt, by_label, metrics, sub_rows, chosen)
    print(f"Ranked {len(metrics)} variants, {len(sub_rows)} substitutions:")
    for path in written:
        print(f"  {path}")

    if chosen:
        for d in metrics[:len(chosen)] if not args.only else []:
            gain = f"{d['delta_T']:+.1f}" if d["delta_T"] is not None else "n/a"
            print(f"  {d['rank']:>2}. {d['label']}  score {d['score']:+.2f}  "
                  f"dT {gain}  bad both {d['n_bad_both']}  one {d['n_bad_one']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
