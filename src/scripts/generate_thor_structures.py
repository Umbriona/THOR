#!/usr/bin/env python
"""
Run generate_thor.py once per FASTA record, pairing each with its structure.

generate_thor.py takes one backbone at a time: ProteinMPNN features describe a
single chain, so --structure and --fasta cannot be combined. This driver closes
that gap for a cluster run - give it a FASTA and a directory of structures named
after the record ids, and it runs each pair in turn, then writes the per-run
tables and figures plus one combined CSV.

    python generate_thor_structures.py \\
        --run_dir  RUN --fasta seqs.fasta --structure_dir structures/ \\
        --output_dir out/ --mpnn_dir /opt/ProteinMPNN --mpnn_weight 0.3 \\
        -- --replicates 10 --esm_filter_threshold 3.0

Anything after `--` is passed straight to generate_thor.py, so every flag that
script accepts is available without being restated here.

A record whose run fails is reported and skipped; the remaining records still
run, and the exit status is non-zero if any failed.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STRUCTURE_SUFFIXES = (".cif", ".pdb", ".mmcif", ".ent")


def parse_fasta(path: Path) -> List[Tuple[str, str, Optional[float]]]:
    """(id, sequence, temperature) per record; temperature is the trailing number if any."""
    records: List[Tuple[str, str, Optional[float]]] = []
    header, chunks = None, []

    def flush():
        if header is None:
            return
        rid = header.split()[0]
        temp = None
        tail = header.split()
        if len(tail) > 1:
            try:
                temp = float(tail[-1])
            except ValueError:
                temp = None
        records.append((rid, "".join(chunks), temp))

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header, chunks = line[1:], []
            else:
                chunks.append(line.replace(" ", "").upper())
    flush()
    return records


def find_structure(structure_dir: Optional[Path], rid: str) -> Optional[Path]:
    """The structure file named after this record, whichever supported suffix it uses."""
    if structure_dir is None:
        return None
    for suffix in STRUCTURE_SUFFIXES:
        candidate = structure_dir / f"{rid}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_command(args, rid: str, seq: str, temp: Optional[float],
                  structure: Optional[Path], out_dir: Path, passthrough: List[str]) -> List[str]:
    cmd = [
        sys.executable, str(HERE / "generate_thor.py"),
        "--run_dir", str(args.run_dir),
        "--seq_id", rid,
        "--name", rid,
        "--output_dir", str(out_dir),
    ]
    # With --seq_source structure the chain's own sequence is used, which is what
    # you want when the model has unresolved residues the FASTA still lists.
    if not (structure and args.seq_source == "structure"):
        cmd += ["--sequence", seq]
    if temp is not None:
        cmd += ["--seq_temp", str(temp)]
    if structure:
        cmd += ["--structure", str(structure), "--mpnn_weight", str(args.mpnn_weight)]
        if args.mpnn_dir:
            cmd += ["--mpnn_dir", str(args.mpnn_dir)]
        if args.mpnn_chain:
            cmd += ["--mpnn_chain", args.mpnn_chain]
    return cmd + passthrough


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Passthrough flags must not be swallowed by a prefix match on ours.
        allow_abbrev=False,
    )
    parser.add_argument("--run_dir", required=True, help="Trained run directory (config.yaml + weights/)")
    parser.add_argument("--fasta", required=True, type=Path, help="Input sequences")
    parser.add_argument("--structure_dir", type=Path, default=None,
                        help=f"Directory of structures named <record_id>{{{','.join(STRUCTURE_SUFFIXES)}}}")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Results root; each record gets a subdirectory")
    parser.add_argument("--mpnn_dir", default=None, help="ProteinMPNN checkout")
    parser.add_argument("--mpnn_weight", type=float, default=0.3,
                        help="Blend weight for structure features (0 = pure ESM)")
    parser.add_argument("--mpnn_chain", default=None, help="Chain to use in every structure")
    parser.add_argument("--seq_source", choices=["fasta", "structure"], default="fasta",
                        help="Which sequence to generate from when a structure is present. "
                             "'fasta' fails loudly if the two disagree; 'structure' uses the "
                             "resolved chain")
    parser.add_argument("--require_structure", action="store_true",
                        help="Fail a record that has no structure instead of running it ESM-only")
    parser.add_argument("--only", nargs="+", default=None, help="Restrict to these record ids")
    parser.add_argument("--no_summary", action="store_true", help="Skip the tables and figures")
    parser.add_argument("--dry_run", action="store_true", help="Print the commands and stop")
    args, passthrough = parser.parse_known_args()
    # argparse leaves the separator in when it appears before unknown args.
    passthrough = [a for a in passthrough if a != "--"]

    records = parse_fasta(args.fasta)
    if args.only:
        wanted = set(args.only)
        missing = wanted - {r[0] for r in records}
        if missing:
            parser.error(f"--only names ids not in {args.fasta}: {sorted(missing)}")
        records = [r for r in records if r[0] in wanted]
    if not records:
        parser.error(f"No sequences in {args.fasta}")

    print(f"[info] {len(records)} record(s) from {args.fasta}")
    if args.structure_dir and not args.structure_dir.is_dir():
        parser.error(f"--structure_dir is not a directory: {args.structure_dir}")

    ok, failed, skipped = [], [], []
    for n, (rid, seq, temp) in enumerate(records, 1):
        structure = find_structure(args.structure_dir, rid)
        where = structure.name if structure else "no structure"
        print(f"\n[{n}/{len(records)}] {rid}  ({len(seq)} aa, {where})", flush=True)

        if structure is None and args.structure_dir:
            if args.require_structure:
                print(f"[fail] {rid}: no structure in {args.structure_dir} "
                      f"(expected {rid}{STRUCTURE_SUFFIXES[0]})")
                failed.append(rid)
                continue
            print(f"[warn] {rid}: no structure found; running ESM-only")

        out_dir = args.output_dir / rid
        cmd = build_command(args, rid, seq, temp, structure, out_dir, passthrough)
        if args.dry_run:
            print("  " + " ".join(cmd))
            skipped.append(rid)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        # Inherit stdout/stderr so progress reaches the SLURM log as it happens.
        result = subprocess.run(cmd)
        if result.returncode == 0:
            ok.append(rid)
        else:
            print(f"[fail] {rid}: generate_thor.py exited {result.returncode}", flush=True)
            failed.append(rid)

    print(f"\n[info] done: {len(ok)} ok, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print(f"[info] failed ids: {', '.join(failed)}")

    if ok and not args.no_summary:
        from summarise_variants import summarise
        fastas = [args.output_dir / rid / f"{rid}_variants.fasta" for rid in ok]
        fastas = [f for f in fastas if f.exists()]
        if fastas:
            print()
            summarise(fastas, combined_path=args.output_dir / "all_variants.csv")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
