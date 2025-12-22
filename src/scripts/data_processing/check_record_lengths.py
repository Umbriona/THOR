#!/usr/bin/env python3
"""
Quick scanner for TFRecord shards to report sequence lengths.

Defaults to THOR_HUGE train/uniref50 shards, but you can point it at any
directory containing *.tfrecord or *.tfrecord.gz written by the quantized
MLM feature pipeline (fields: seq, length, ...).
"""
import argparse
import pathlib
import sys
from typing import Iterable, Tuple

import tensorflow as tf


def iter_lengths(paths: Iterable[pathlib.Path]) -> Iterable[Tuple[int, int]]:
    """
    Yield (seq_len, recorded_length) for each record in the provided TFRecord files.
    recorded_length can be -1 if the field is missing.
    """
    for path in paths:
        compression = "GZIP" if path.suffix == ".gz" else None
        dataset = tf.data.TFRecordDataset(str(path), compression_type=compression)
        for raw in dataset:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            feat = ex.features.feature
            seq_bytes = feat["seq"].bytes_list.value[0]
            seq_str = seq_bytes.decode("utf-8")
            seq_len = len(seq_str.replace(" ", ""))  # ProtBert seqs may be space-separated
            recorded_len = -1
            if "length" in feat and feat["length"].int64_list.value:
                recorded_len = int(feat["length"].int64_list.value[0])
            yield seq_len, recorded_len


def scan_dir(directory: pathlib.Path):
    paths = sorted(directory.glob("*.tfrecord*"))
    if not paths:
        print(f"No TFRecord files found in {directory}", file=sys.stderr)
        sys.exit(1)

    total = 0
    min_len = None
    max_len = None
    sum_len = 0
    mismatches = 0

    for seq_len, recorded_len in iter_lengths(paths):
        total += 1
        sum_len += seq_len
        min_len = seq_len if min_len is None else min(min_len, seq_len)
        max_len = seq_len if max_len is None else max(max_len, seq_len)
        if recorded_len != -1 and recorded_len != seq_len:
            mismatches += 1

    mean_len = sum_len / total if total else 0.0

    print(f"Scanned {total} records from {len(paths)} file(s) in {directory}")
    print(f"Min length: {min_len}")
    print(f"Max length: {max_len}")
    print(f"Mean length: {mean_len:.2f}")
    print(f"Length field mismatches: {mismatches}")


def main():
    default_dir = pathlib.Path(__file__).resolve().parents[3] / "data" / "THOR_HUGE" / "records" / "train" / "uniref50"
    p = argparse.ArgumentParser(description="Scan TFRecord shards and report sequence lengths.")
    p.add_argument("-d", "--directory", type=pathlib.Path, default=default_dir,
                   help="Directory containing TFRecord shards (default: THOR_HUGE/records/train/uniref50)")
    args = p.parse_args()

    scan_dir(args.directory)


if __name__ == "__main__":
    main()
