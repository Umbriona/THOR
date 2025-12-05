#!/usr/bin/env python3
import os, sys
try:
    os.chdir('/ThermalGAN/src/scripts')
except:
     os.chdir('./')
currentdir = os.path.dirname(os.getcwd())

print(os.listdir(os.getcwd()))
print(os.listdir(currentdir))
print(os.listdir(os.path.dirname(currentdir)))
sys.path.append(currentdir)
import argparse
import textwrap

import numpy as np
import tensorflow as tf
import yaml

# Adjust this import to your actual module location
from utils import models_classifyer as models_class


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"  # order given by you
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}


def read_fasta(path):
    """Return list of (header, sequence) from a FASTA file."""
    entries = []
    header = None
    seq_lines = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_lines)))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
    full_sequence = "".join(seq_lines)
    
    if header is not None and len(full_sequence) > 512:
        entries.append((header, full_sequence))
    else:
        print(f"Skipping: {header} sequence longer than 512\n is: {(header, full_sequence)}")
    return entries


def write_fasta_with_tm(path, entries, tms, line_width=60):
    """
    Write FASTA with TM appended to each header.

    entries: list of (header, seq)
    tms: list/array of TM values, same length as entries
    """
    assert len(entries) == len(tms)

    with open(path, "w") as f:
        for (header, seq), tm in zip(entries, tms):
            new_header = f">{header}, TM={float(tm):.4f}"
            f.write(new_header + "\n")
            for chunk in textwrap.wrap(seq, line_width):
                f.write(chunk + "\n")


def one_hot_encode_sequences(seqs, max_len=512, alphabet=ALPHABET):
    """
    RIGHT-pad and one-hot encode sequences.

    Returns: np.ndarray of shape (N, max_len, len(alphabet))
    """
    num_seqs = len(seqs)
    depth = len(alphabet)
    x = np.zeros((num_seqs, max_len, depth), dtype=np.float32)

    for i, seq in enumerate(seqs):
        seq = seq.upper()
        L = len(seq)

        if L > max_len:
            raise ValueError(
                f"Sequence {i} length {L} exceeds max_len={max_len} is {seq}"
            )

        # write sequence starting at index 0 → right padding
        for j, aa in enumerate(seq):
            idx = AA_TO_IDX.get(aa, AA_TO_IDX["X"])
            x[i, j, idx] = 1.0

        # The rest remains zeros = padding

    return x


def build_ensemble_model(config_path, weights_paths):
    """
    Build your 3-model ensemble and average them.

    weights_paths: list of 3 paths to variables/variables
    """
    with open(config_path, "r") as fd:
        config_class = yaml.load(fd, Loader=yaml.FullLoader)

    model_input = tf.keras.layers.Input(shape=(512, 21))
    model1 = models_class.get_classifier(config_class["Classifier"], 21)
    model2 = models_class.get_classifier(config_class["Classifier"], 21)
    model3 = models_class.get_classifier(config_class["Classifier"], 21)

    print("Initialized regression models")

    output1 = model1(model_input)
    output2 = model2(model_input)
    output3 = model3(model_input)

    model1.load_weights(weights_paths[0]).expect_partial()
    model2.load_weights(weights_paths[1]).expect_partial()
    model3.load_weights(weights_paths[2]).expect_partial()

    print("Loaded weights regression models")

    ensemble_output = tf.keras.layers.Average()([output1, output2, output3])
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

    return ensemble_model


def predict_tm_for_fasta(
    fasta_path,
    out_path,
    config_path,
    weights_paths,
    batch_size=128,
    max_len=512,
):
    # 1) Read FASTA
    entries = read_fasta(fasta_path)
    if not entries:
        raise ValueError(f"No sequences found in {fasta_path}")

    headers, seqs = zip(*entries)

    # 2) One-hot encode (left-padded)
    print(f"Encoding {len(seqs)} sequences...")
    x = one_hot_encode_sequences(seqs, max_len=max_len)

    # 3) Build/load model
    print("Building ensemble model...")
    ensemble_model = build_ensemble_model(config_path, weights_paths)

    # 4) Predict in batches on GPU
    print(f"Predicting TM (batch_size={batch_size})...")
    preds = ensemble_model.predict(x, batch_size=batch_size, verbose=1)

    # Flatten in case shape is (N, 1) or (N,)
    tms = preds.reshape((preds.shape[0],))

    # 5) Write output FASTA
    print(f"Writing output FASTA with TM to {out_path}")
    write_fasta_with_tm(out_path, entries, tms)


def main():
    parser = argparse.ArgumentParser(
        description="Predict TM for sequences in a FASTA file and append TM to headers."
    )
    parser.add_argument("fasta_in", help="Input FASTA file with sequences")
    parser.add_argument(
        "fasta_out",
        help="Output FASTA file with TM values appended to headers",
    )
    parser.add_argument(
        "--config",
        default="/ThermalGAN/config/Classifier/config_classifier1.yaml",
        help="Path to classifier YAML config",
    )
    parser.add_argument(
        "--weights",
        nargs=3,
        default=[
            "/ThermalGAN/weights/OGT/Model1/variables/variables",
            "/ThermalGAN/weights/OGT/Model2/variables/variables",
            "/ThermalGAN/weights/OGT/Model3/variables/variables",
        ],
        help="Three paths to weight files (variables/variables) for the ensemble",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for prediction (default: 128)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum sequence length for padding (default: 512)",
    )

    args = parser.parse_args()

    predict_tm_for_fasta(
        fasta_path=args.fasta_in,
        out_path=args.fasta_out,
        config_path=args.config,
        weights_paths=args.weights,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
