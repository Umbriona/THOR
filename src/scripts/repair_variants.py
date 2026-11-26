#!/usr/bin/env python
"""Repair the substitutions both scoring models reject, then score the result.

`variant_ranking.py` flags a substitution as bad under a model on either of two
grounds, and counts hardest the positions where *both* models object. Those are
the positions this script changes. Everything else the generator proposed is
left alone: the point is to keep a variant's gain while removing the handful of
residues neither model can account for.

Two rules are available for what to put there instead:

    model_best  the residue the models jointly like best at that position,
                scored as the mean of their log-probabilities over all 20. The
                wildtype residue is among the candidates, so this reverts when
                reverting is what the models prefer.
    wildtype    always the wildtype residue.

The joint-best residue is chosen from the probabilities computed for the
*unrepaired* variant, so it is a proposal, not a verdict: changing a residue
changes the context its neighbours are judged in. That is why the repaired
sequences are then run through ESM and the temperature predictor again and
re-ranked beside their parents - the comparison, not the proposal, is what says
whether a repair helped.

ProteinMPNN is not re-run. Its probabilities are a property of the backbone,
not of the sequence threaded through it, so they are unchanged by a repair and
the same matrix scores parent and repair alike.

    python repair_variants.py BUNDLE_DIR --scope top --top 5 \
        --ogt_config ... --ogt_weights w1.h5 w2.h5 w3.h5

writes `<name>_repaired.fasta`, `<name>_repaired_scoring_probs.jsonl` and
`<name>_repairs.csv`.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import variant_ranking as vr

AA20 = vr.AA20
LOG_EPS = 1e-9


def repair_targets(rows: Sequence[Dict], require_both: bool = True) -> List[Dict]:
    """The substitutions a repair should change.

    Normally the positions both models reject. With one model available nothing
    can disagree with both, so `require_both=False` falls back to that model's
    own objections - stated by the caller, never assumed here.
    """
    if require_both:
        return [r for r in rows if r.get("disagrees_with") == "both"]
    return [r for r in rows if r.get("bad_esm") or r.get("bad_mpnn")]


def _joint_log_probs(pos: int, probs_list: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Mean log-probability over the models that have an opinion here."""
    rows = [np.asarray(p[pos], dtype=float) for p in probs_list
            if p is not None and pos < len(p)]
    if not rows:
        return None
    return np.mean([np.log(np.maximum(r, LOG_EPS)) for r in rows], axis=0)


def repaired_sequence(var_seq: str, wt_seq: str, targets: Sequence[Dict],
                      esm_probs: Optional[np.ndarray] = None,
                      mpnn_probs: Optional[np.ndarray] = None,
                      mode: str = "model_best",
                      aa_order: str = AA20) -> Tuple[str, List[Dict]]:
    """Apply the repair rule at each target position.

    Returns the new sequence and one record per position changed. A target the
    rule cannot improve - no model opinion, or the best candidate is the residue
    already there - is left alone and not reported as a change.
    """
    seq = list(var_seq)
    changes: List[Dict] = []
    for row in targets:
        pos = int(row["position"]) - 1
        if pos < 0 or pos >= len(seq):
            continue
        current = seq[pos]
        wt_aa = wt_seq[pos] if pos < len(wt_seq) else current

        if mode == "wildtype":
            new_aa = wt_aa
        else:
            joint = _joint_log_probs(pos, [esm_probs, mpnn_probs])
            if joint is None:
                continue
            order = np.argsort(joint)[::-1]
            new_aa = next((aa_order[i] for i in order if aa_order[i] != current), current)

        if new_aa == current:
            continue
        seq[pos] = new_aa
        changes.append({
            "position": pos + 1,
            "wt_aa": wt_aa,
            "from_aa": current,
            "to_aa": new_aa,
            "substitution": f"{wt_aa}{pos + 1}{current}",
            "repair": f"{current}{pos + 1}{new_aa}",
            "reverted_to_wt": new_aa == wt_aa,
            "esm_llr": row.get("esm_llr"),
            "mpnn_llr": row.get("mpnn_llr"),
        })
    return "".join(seq), changes


def plan_repairs(labels: Sequence[str], by_label: Dict[str, Dict], wt_seq: str,
                 sub_rows: Sequence[Dict], esm: Dict[str, np.ndarray],
                 mpnn: Optional[np.ndarray], mode: str = "model_best",
                 require_both: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """One repaired record per label that had something to repair."""
    rows_by_label: Dict[str, List[Dict]] = {}
    for row in sub_rows:
        rows_by_label.setdefault(row["label"], []).append(row)

    records, change_rows = [], []
    for label in labels:
        rec = by_label.get(label)
        if rec is None:
            continue
        targets = repair_targets(rows_by_label.get(label, []), require_both)
        if not targets:
            continue
        seq, changes = repaired_sequence(rec["seq"], wt_seq, targets,
                                         esm.get(label), mpnn, mode)
        if not changes:
            continue
        new_label = f"{label}_repaired"
        records.append({
            "label": new_label,
            "parent": label,
            "kind": "repaired",
            "seq": seq,
            "repairs": [c["repair"] for c in changes],
            "n_repaired": len(changes),
        })
        for c in changes:
            change_rows.append({"label": new_label, "parent": label, **c})
    return records, change_rows


# --------------------------------------------------------------------------
# Re-scoring. These pull in TensorFlow and Torch, so they load on demand.
# --------------------------------------------------------------------------
def score_records(records: List[Dict], wt_seq: str, args) -> Dict[str, np.ndarray]:
    """Run ESM and the temperature predictor over the repaired sequences.

    Fills `esm_probs`, `esm_global_nll`, `esm_sub_nll`, `identity_to_wt` and
    `ogt` on each record, and returns the ESM probabilities by label.
    """
    from types import SimpleNamespace

    import generate_thor as gt

    print(f"[info] Loading ESM ({args.hf_model})...")
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(args.hf_model)
    if hasattr(model, "esm") and hasattr(model.esm, "contact_head"):
        model.esm.contact_head = None
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    model = model.to(device)
    print(f"[info] ESM on {device}; scoring {len(records)} repaired sequence(s)...")

    esm_args = SimpleNamespace(max_length=args.max_length,
                               batch_tokens=args.batch_tokens,
                               esm_bf16=args.esm_bf16)
    results = gt.run_esm(tokenizer, model,
                         [r["label"] for r in records],
                         [r["seq"] for r in records],
                         [None] * len(records), esm_args)

    probs_by_label = {}
    for rec in records:
        probs = results[rec["label"]]["probs"]
        g, sub = gt.compute_esm_nll(rec["seq"], probs, wt_seq)
        rec["esm_probs"] = probs
        rec["esm_global_nll"] = g
        rec["esm_sub_nll"] = sub
        rec["identity_to_wt"] = gt.compute_identity(rec["seq"], wt_seq)
        probs_by_label[rec["label"]] = np.asarray(probs, dtype=np.float32)

    del model, tokenizer
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    if args.skip_ogt:
        print("[info] Skipping the temperature predictor as requested.")
    else:
        print("[info] Predicting OGT for the repaired sequences...")
        gt.predict_ogt_for_records(records, args.ogt_config, args.ogt_weights,
                                   args.ogt_batch_size, args.ogt_max_len)
    return probs_by_label


def write_repairs(out_dir: Path, prefix: str, records: Sequence[Dict],
                  change_rows: Sequence[Dict]) -> List[Path]:
    """The FASTA, the ESM probabilities and the table of what changed.

    The probabilities file carries the `_scoring_probs.jsonl` suffix on purpose:
    anything that reads a bundle picks the repaired distributions up with the
    originals, and the labels keep them apart.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def metric(val):
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return "NA"
        return f"{val:.4f}"

    fasta_path = out_dir / f"{prefix}_repaired.fasta"
    with open(fasta_path, "w") as fh:
        for rec in records:
            header = [
                rec["label"],
                "type=repaired",
                f"predicted_OGT={metric(rec.get('ogt'))}",
                f"ESM_global_nll={metric(rec.get('esm_global_nll'))}",
                f"ESM_sub_nll={metric(rec.get('esm_sub_nll'))}",
                f"Identity_to_WT(%)={metric(rec.get('identity_to_wt'))}",
                f"parent={rec['parent']}",
                f"n_repaired={rec['n_repaired']}",
                f"repairs={','.join(rec['repairs'])}",
            ]
            fh.write(">" + " | ".join(header) + "\n" + rec["seq"] + "\n")
    written.append(fasta_path)

    probs_path = out_dir / f"{prefix}_repaired_scoring_probs.jsonl"
    with open(probs_path, "w") as fh:
        for rec in records:
            probs = rec.get("esm_probs")
            if probs is None:
                continue
            trimmed = [[round(float(v), 6) for v in row]
                       for row in np.asarray(probs)[:len(rec["seq"])]]
            fh.write(json.dumps({
                "Scoring_Model": "esm",
                "label": rec["label"],
                "kind": "repaired",
                "aa_order": AA20,
                "probs": trimmed,
            }) + "\n")
    written.append(probs_path)

    if change_rows:
        changes_path = out_dir / f"{prefix}_repairs.csv"
        vr.write_csv(changes_path, list(change_rows))
        written.append(changes_path)
    return written


def load_repaired(data_dir: Path, prefix: str):
    """Read back what `write_repairs` wrote: (records, esm probabilities)."""
    data_dir = Path(data_dir)
    fasta = data_dir / f"{prefix}_repaired.fasta"
    probs_file = data_dir / f"{prefix}_repaired_scoring_probs.jsonl"
    if not fasta.exists():
        return [], {}

    records, label, meta, seq = [], None, {}, []

    def flush():
        if label is not None:
            records.append({"label": label, "seq": "".join(seq), "kind": "repaired", **meta})

    for line in fasta.read_text().splitlines():
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

    esm = {}
    if probs_file.exists():
        for line in probs_file.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            esm[d["label"]] = np.asarray(d["probs"], dtype=np.float32)
    return records, esm


def compare(metrics: Sequence[Dict]) -> List[Dict]:
    """Each repaired variant beside the parent it came from."""
    by_label = {d["label"]: d for d in metrics}
    rows = []
    for d in metrics:
        if not d["label"].endswith("_repaired"):
            continue
        parent = by_label.get(d["label"][: -len("_repaired")])
        if parent is None:
            continue
        rows.append({
            "parent": parent["label"],
            "score_before": parent["score"],
            "score_after": d["score"],
            "score_change": round(d["score"] - parent["score"], 4),
            "delta_T_before": parent["delta_T"],
            "delta_T_after": d["delta_T"],
            "risk_before": parent["risk"],
            "risk_after": d["risk"],
            "bad_both_before": parent["n_bad_both"],
            "bad_both_after": d["n_bad_both"],
            "bad_one_before": parent["n_bad_one"],
            "bad_one_after": d["n_bad_one"],
            "rank_before": parent["rank"],
            "rank_after": d["rank"],
            "improved": d["score"] > parent["score"],
        })
    rows.sort(key=lambda r: r["score_change"], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="THOR output directory or unpacked bundle")
    parser.add_argument("--scope", choices=["none", "top", "all"], default="top",
                        help="Which variants to repair: none, the top --top by score, or all")
    parser.add_argument("--top", type=int, default=5, help="How many, for --scope top")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Repair these labels instead of choosing by scope")
    parser.add_argument("--mode", choices=["model_best", "wildtype"], default="model_best",
                        help="What to put at a rejected position")
    parser.add_argument("--out_dir", type=Path, default=None, help="Default: data_dir")
    # Ranking, so the flagged positions match what the notebooks show.
    parser.add_argument("--bad_llr", type=float, default=vr.BAD_LLR)
    parser.add_argument("--implausible_percentile", type=float, default=vr.IMPLAUSIBLE_PERCENTILE)
    parser.add_argument("--tradeoff", type=float, default=vr.TRADEOFF)
    # ESM.
    parser.add_argument("--hf_model", default="facebook/esm1v_t33_650M_UR90S_1")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch_tokens", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=1022)
    parser.add_argument("--esm_bf16", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")
    # Temperature predictor.
    parser.add_argument("--ogt_config", default="/ThermalGAN/config/Classifier/config_classifier1.yaml")
    parser.add_argument("--ogt_weights", nargs=3, default=[])
    parser.add_argument("--ogt_batch_size", type=int, default=128)
    parser.add_argument("--ogt_max_len", type=int, default=512)
    parser.add_argument("--skip_ogt", action="store_true")
    args = parser.parse_args()

    if args.scope == "none" and not args.labels:
        print("Nothing to do (--scope none).")
        return 0

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    records, wt, esm, mpnn, delta_t, fasta = vr.load_bundle(args.data_dir)
    by_label = {r["label"]: r for r in records}
    metrics, sub_rows = vr.rank(
        by_label, wt["seq"], wt["label"], esm, mpnn, delta_t,
        tradeoff=args.tradeoff, bad_llr=args.bad_llr,
        implausible_percentile=args.implausible_percentile)

    if args.labels:
        labels = list(args.labels)
        missing = [lab for lab in labels if lab not in by_label]
        if missing:
            parser.error(f"--labels names sequences not in {fasta.name}: {missing}")
    elif args.scope == "all":
        labels = [d["label"] for d in metrics]
    else:
        labels = [d["label"] for d in metrics[:max(0, args.top)]]

    require_both = mpnn is not None
    if not require_both:
        print("[info] No ProteinMPNN probabilities in this bundle, so nothing can be "
              "rejected by both models; repairing what ESM alone rejects.")

    repaired, change_rows = plan_repairs(labels, by_label, wt["seq"], sub_rows,
                                         esm, mpnn, args.mode, require_both)
    if not repaired:
        print(f"Nothing to repair in {len(labels)} variant(s): no position is rejected "
              f"by both models.")
        return 0

    print(f"Repairing {len(repaired)} of {len(labels)} variant(s), "
          f"{len(change_rows)} position(s), rule '{args.mode}'.")
    probs_by_label = score_records(repaired, wt["seq"], args)

    out_dir = args.out_dir or args.data_dir
    prefix = fasta.name.split("_variants")[0]
    written = write_repairs(out_dir, prefix, repaired, change_rows)

    # Rank parents and repairs together: the comparison is the point.
    for rec in repaired:
        by_label[rec["label"]] = {"label": rec["label"], "seq": rec["seq"],
                                  "predicted_OGT": rec.get("ogt")}
        esm[rec["label"]] = probs_by_label[rec["label"]]
    wt_ogt = vr._as_float(wt.get("predicted_OGT"))
    for rec in repaired:
        ogt = vr._as_float(rec.get("ogt"))
        delta_t[rec["label"]] = None if (ogt is None or wt_ogt is None) else ogt - wt_ogt

    metrics, sub_rows = vr.rank(
        by_label, wt["seq"], wt["label"], esm, mpnn, delta_t,
        tradeoff=args.tradeoff, bad_llr=args.bad_llr,
        implausible_percentile=args.implausible_percentile)
    rows = compare(metrics)
    if rows:
        comparison_path = Path(out_dir) / f"{prefix}_repair_comparison.csv"
        vr.write_csv(comparison_path, rows)
        written.append(comparison_path)

    for path in written:
        print(f"  {path}")
    print()
    better = sum(1 for r in rows if r["improved"])
    print(f"{better} of {len(rows)} repair(s) scored better than their parent:")
    for r in rows:
        arrow = "+" if r["improved"] else " "
        print(f"  {arrow} {r['parent']}  score {r['score_before']:+.2f} -> "
              f"{r['score_after']:+.2f}   bad both {r['bad_both_before']} -> "
              f"{r['bad_both_after']}   rank {r['rank_before']} -> {r['rank_after']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
