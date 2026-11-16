#!/usr/bin/env python
"""
Structure-based per-position features and sequence scores from ProteinMPNN.

The generator consumes an (L, 20) probability matrix in AA20 order, whichever
model produced it. ProteinMPNN's alphabet is ACDEFGHIKLMNPQRSTVWYX, so its
first 20 columns are already in AA20 order and need no remapping.

ProteinMPNN is not a pip package; clone it next to the run and point --mpnn_dir
at the checkout:

    git clone https://github.com/dauparas/ProteinMPNN
"""
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA20 = "ACDEFGHIKLMNPQRSTVWY"

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y", "MSE": "M",
}


def locate_proteinmpnn(mpnn_dir: str) -> Path:
    """Resolve the ProteinMPNN checkout, with an actionable error if it is missing."""
    root = Path(mpnn_dir).expanduser().resolve()
    runner = root / "protein_mpnn_run.py"
    if not runner.exists():
        raise FileNotFoundError(
            f"No protein_mpnn_run.py under {root}. Clone it first:\n"
            f"    git clone https://github.com/dauparas/ProteinMPNN {root}"
        )
    return root


def list_chains(structure_path: str) -> List[str]:
    """Chain IDs that carry CA atoms, in the order they first appear."""
    chains: List[str] = []
    with open(structure_path) as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA":
                chain = line[21].strip() or "A"
                if chain not in chains:
                    chains.append(chain)
    return chains


def extract_chain(structure_path: str, chain_id: str, out_path: Path) -> Tuple[str, List[int]]:
    """Write a single-chain PDB and report the chain's sequence and residue numbering.

    Keeping one chain per structure makes the ProteinMPNN outputs unambiguous:
    every residue in the npz belongs to the designed chain, so feature rows line
    up with sequence positions by construction.
    """
    kept: List[str] = []
    seq: List[str] = []
    resnums: List[int] = []
    seen = set()
    with open(structure_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if (line[21].strip() or "A") != chain_id:
                continue
            if line.startswith("HETATM") and line[17:20].strip() != "MSE":
                continue  # ligands/waters, but keep selenomethionine
            if line[16] not in (" ", "A"):
                continue  # keep only the first altloc
            kept.append("ATOM  " + line[6:])
            if line[12:16].strip() == "CA":
                key = line[22:27]  # resseq + insertion code
                if key in seen:
                    continue
                seen.add(key)
                seq.append(THREE_TO_ONE.get(line[17:20].strip().upper(), "X"))
                resnums.append(int(line[22:26]))

    if not seq:
        raise ValueError(f"Chain {chain_id!r} has no residues in {structure_path}")

    out_path.write_text("".join(kept) + "TER\nEND\n")
    return "".join(seq), resnums


def numbering_gaps(resnums: List[int]) -> List[Tuple[int, int]]:
    """Breaks in residue numbering, which mean unresolved residues in the model.

    A gap does not stop the run - ProteinMPNN scores the residues that exist -
    but every downstream position index refers to the *resolved* chain, so the
    caller must not mix those indices with full-construct numbering.
    """
    return [
        (a, b)
        for a, b in zip(resnums, resnums[1:])
        if b - a != 1
    ]


def _run_mpnn(root: Path, args: List[str]) -> None:
    cmd = [sys.executable, str(root / "protein_mpnn_run.py")] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ProteinMPNN failed:\n"
            + " ".join(cmd)
            + f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _load_single_npz(folder: Path) -> Dict[str, np.ndarray]:
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"ProteinMPNN wrote no npz under {folder}")
    return dict(np.load(files[0]))


def compute_structure_probs(
    structure_path: str,
    chain_id: str,
    mpnn_dir: str,
    model_name: str = "v_48_020",
    use_soluble_model: bool = False,
    backbone_noise: float = 0.0,
    seed: int = 37,
    work_dir: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Per-position amino-acid probabilities given the backbone alone.

    Returns (probs (L, 20) in AA20 order, sequence parsed by ProteinMPNN).

    These are *unconditional* probabilities: they depend on the backbone, not on
    the sequence threaded through it, so one call covers the wildtype and every
    variant built on the same structure.

    The 21st column (X) is dropped without renormalising, so each row sums to
    slightly less than 1 - the same convention as the ESM features, where the
    probability mass on non-amino-acid tokens is likewise left out.
    """
    root = locate_proteinmpnn(mpnn_dir)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        tmp_path = Path(tmp)
        chain_pdb = tmp_path / "chain.pdb"
        seq, _ = extract_chain(structure_path, chain_id, chain_pdb)
        out_dir = tmp_path / "out"
        _run_mpnn(root, [
            "--pdb_path", str(chain_pdb),
            "--pdb_path_chains", chain_id,
            "--out_folder", str(out_dir),
            "--unconditional_probs_only", "1",
            "--model_name", model_name,
            "--backbone_noise", str(backbone_noise),
            "--seed", str(seed),
        ] + (["--use_soluble_model"] if use_soluble_model else []))

        data = _load_single_npz(out_dir / "unconditional_probs_only")

    return probs_from_npz(data, seq), seq


def probs_from_npz(data: Dict[str, np.ndarray], seq: str) -> np.ndarray:
    """Turn a ProteinMPNN probs npz into (L, 20) AA20-ordered probabilities.

    `log_p` covers the padded structure, so designed positions are selected with
    `design_mask` before anything else; the rows that survive are the chain's
    residues in order.
    """
    log_p = data["log_p"]          # [batches, L_padded, 21]
    keep = np.nonzero(data["design_mask"] > 0)[0]
    probs = np.exp(log_p).mean(axis=0)[keep][:, :20].astype(np.float32)

    # ProteinMPNN's own view of the sequence, as a guard against a silent
    # misalignment between the residues it parsed and the ones we extracted.
    mpnn_seq = "".join(MPNN_ALPHABET[i] for i in data["S"][keep])
    if mpnn_seq != seq:
        raise ValueError(
            "ProteinMPNN parsed a different chain sequence than the PDB reader:\n"
            f"  mpnn: {mpnn_seq}\n  ours: {seq}"
        )
    if probs.shape[0] != len(seq):
        raise ValueError(f"ProteinMPNN returned {probs.shape[0]} rows for {len(seq)} residues")
    return probs


def score_sequences(
    structure_path: str,
    chain_id: str,
    sequences: List[str],
    mpnn_dir: str,
    model_name: str = "v_48_020",
    use_soluble_model: bool = False,
    seed: int = 37,
    work_dir: Optional[str] = None,
) -> List[float]:
    """Mean negative log-likelihood of each sequence given the backbone.

    Lower is better: the sequence sits more comfortably on this fold. Every
    sequence must be the same length as the chain.
    """
    if not sequences:
        return []
    root = locate_proteinmpnn(mpnn_dir)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        tmp_path = Path(tmp)
        chain_pdb = tmp_path / "chain.pdb"
        seq, _ = extract_chain(structure_path, chain_id, chain_pdb)
        bad = [i for i, s in enumerate(sequences) if len(s) != len(seq)]
        if bad:
            raise ValueError(
                f"{len(bad)} sequence(s) do not match the chain length ({len(seq)}); "
                f"first offender has length {len(sequences[bad[0]])}"
            )

        fasta = tmp_path / "variants.fa"
        fasta.write_text(
            "".join(f">seq_{i}\n{s}\n" for i, s in enumerate(sequences))
        )
        out_dir = tmp_path / "out"
        _run_mpnn(root, [
            "--pdb_path", str(chain_pdb),
            "--pdb_path_chains", chain_id,
            "--out_folder", str(out_dir),
            "--score_only", "1",
            "--path_to_fasta", str(fasta),
            "--model_name", model_name,
            "--seed", str(seed),
        ] + (["--use_soluble_model"] if use_soluble_model else []))

        score_dir = out_dir / "score_only"
        scores = []
        for i in range(len(sequences)):
            # ProteinMPNN numbers the FASTA outputs from 1 (<name>_fasta_1.npz).
            files = sorted(score_dir.glob(f"*_fasta_{i + 1}.npz"))
            if not files:
                raise FileNotFoundError(f"No ProteinMPNN score for sequence {i} in {score_dir}")
            scores.append(float(np.load(files[0])["global_score"].mean()))
    return scores


def blend_probs(esm_probs: np.ndarray, mpnn_probs: np.ndarray, weight: float) -> np.ndarray:
    """Convex mix of sequence- and structure-derived features.

    weight=0 is pure ESM (what the generator was trained on), weight=1 pure
    ProteinMPNN. Values in between keep the input near the training distribution
    while letting the backbone pull on it.
    """
    if esm_probs.shape != mpnn_probs.shape:
        raise ValueError(
            f"Feature shape mismatch: ESM {esm_probs.shape} vs ProteinMPNN {mpnn_probs.shape}"
        )
    return ((1.0 - weight) * esm_probs + weight * mpnn_probs).astype(np.float32)
