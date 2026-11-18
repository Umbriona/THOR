#!/usr/bin/env python
"""
Structure-based per-position features and sequence scores from ProteinMPNN.

The generator consumes an (L, 20) probability matrix in AA20 order, whichever
model produced it. Structures are read from PDB or mmCIF; ProteinMPNN's own
reader only handles PDB, so the chain we hand it is always written back out as
PDB first. ProteinMPNN's alphabet is ACDEFGHIKLMNPQRSTVWYX, so its
first 20 columns are already in AA20 order and need no remapping.

ProteinMPNN is not a pip package; clone it next to the run and point --mpnn_dir
at the checkout:

    git clone https://github.com/dauparas/ProteinMPNN
"""
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

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


class Atom(NamedTuple):
    """One atom record, from either input format."""
    hetatm: bool
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str


def _parse_pdb_atoms(structure_path: str) -> List[Atom]:
    atoms: List[Atom] = []
    with open(structure_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break  # first model only
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atoms.append(Atom(
                hetatm=line.startswith("HETATM"),
                name=line[12:16].strip(),
                altloc=line[16].strip(),
                resname=line[17:20].strip().upper(),
                chain=line[21].strip() or "A",
                resseq=int(line[22:26]),
                icode=line[26].strip(),
                x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                occupancy=float(line[54:60] or 1.0),
                bfactor=float(line[60:66] or 0.0),
                element=line[76:78].strip() or line[12:16].strip()[:1],
            ))
    return atoms


def _cif_tokens(line: str) -> List[str]:
    """Split one mmCIF data line, honouring the quoting the format allows."""
    tokens: List[str] = []
    i, n = 0, len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        if line[i] in "'\"":
            quote = line[i]
            j = i + 1
            # A quote only closes a value when whitespace (or the end) follows it.
            while j < n and not (line[j] == quote and (j + 1 == n or line[j + 1].isspace())):
                j += 1
            tokens.append(line[i + 1:j])
            i = j + 2
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens


def _parse_cif_atoms(structure_path: str) -> List[Atom]:
    """Read the _atom_site loop of an mmCIF/PDBx file (AlphaFold, Boltz, RCSB).

    Only the columns we need are required, and auth_* numbering is preferred so
    chain IDs and residue numbers match what a PDB of the same model would show.
    """
    cols: Dict[str, int] = {}
    rows: List[List[str]] = []
    with open(structure_path) as fh:
        in_header = False
        in_loop = False
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("_atom_site."):
                cols[stripped.split()[0][len("_atom_site."):]] = len(cols)
                in_header, in_loop = True, False
                continue
            if in_header:
                in_header, in_loop = False, True
            if not in_loop:
                continue
            if not stripped or stripped.startswith("#") or stripped.startswith("_") \
                    or stripped.startswith("loop_"):
                break  # end of the atom loop
            rows.append(_cif_tokens(line))

    if not cols:
        return []

    def pick(*names: str) -> int:
        for name in names:
            if name in cols:
                return cols[name]
        raise ValueError(f"mmCIF _atom_site loop in {structure_path} lacks {names[0]}")

    i_group = pick("group_PDB")
    i_name = pick("auth_atom_id", "label_atom_id")
    i_alt = pick("label_alt_id", "auth_atom_id")
    i_comp = pick("auth_comp_id", "label_comp_id")
    i_chain = pick("auth_asym_id", "label_asym_id")
    i_seq = pick("auth_seq_id", "label_seq_id")
    i_x, i_y, i_z = pick("Cartn_x"), pick("Cartn_y"), pick("Cartn_z")
    i_model = cols.get("pdbx_PDB_model_num")
    i_ins = cols.get("pdbx_PDB_ins_code")
    i_occ = cols.get("occupancy")
    i_b = cols.get("B_iso_or_equiv")
    i_elem = cols.get("type_symbol")
    width = max(cols.values()) + 1

    def value(row: List[str], idx: Optional[int], default: str = "") -> str:
        if idx is None or idx >= len(row):
            return default
        tok = row[idx]
        return default if tok in (".", "?") else tok

    first_model = None
    atoms: List[Atom] = []
    for row in rows:
        if len(row) < width:
            continue
        model = value(row, i_model, "1")
        if first_model is None:
            first_model = model
        if model != first_model:
            continue  # first model only
        name = row[i_name]
        try:
            resseq = int(row[i_seq])
        except ValueError:
            continue  # label_seq_id is "." for ligands
        atoms.append(Atom(
            hetatm=row[i_group].upper() == "HETATM",
            name=name,
            altloc=value(row, i_alt),
            resname=row[i_comp].upper(),
            chain=row[i_chain],
            resseq=resseq,
            icode=value(row, i_ins),
            x=float(row[i_x]), y=float(row[i_y]), z=float(row[i_z]),
            occupancy=float(value(row, i_occ, "1.0")),
            bfactor=float(value(row, i_b, "0.0")),
            element=value(row, i_elem) or name[:1],
        ))
    return atoms


def read_atoms(structure_path: str) -> List[Atom]:
    """Atom records from a .pdb/.ent or .cif/.mmcif file, first model only."""
    suffix = Path(structure_path).suffix.lower()
    if suffix in (".cif", ".mmcif"):
        return _parse_cif_atoms(structure_path)
    if suffix in (".pdb", ".ent"):
        return _parse_pdb_atoms(structure_path)
    # Unknown extension: sniff for the mmCIF atom loop before falling back.
    with open(structure_path) as fh:
        head = fh.read(200_000)
    return (_parse_cif_atoms if "_atom_site." in head else _parse_pdb_atoms)(structure_path)


def pdb_chain_id(chain_id: str) -> str:
    """The single character a chain gets in the PDB we hand to ProteinMPNN.

    mmCIF allows multi-character chain IDs, which do not fit the PDB column, so
    those are renamed. Callers must use this same id in --pdb_path_chains.
    """
    return chain_id if len(chain_id) == 1 else "A"


def _pdb_line(serial: int, atom: Atom, chain: str) -> str:
    # A 1-2 letter atom name is indented by one column; a 4-letter one is not.
    name = atom.name if len(atom.name) >= 4 else f" {atom.name:<3}"
    return (
        f"ATOM  {serial:5d} {name:<4}{atom.altloc[:1] or ' '}{atom.resname:>3} "
        f"{chain}{atom.resseq:4d}{atom.icode[:1] or ' '}   "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{atom.occupancy:6.2f}{atom.bfactor:6.2f}          {atom.element:>2}\n"
    )


def list_chains(structure_path: str) -> List[str]:
    """Chain IDs that carry CA atoms, in the order they first appear."""
    chains: List[str] = []
    for atom in read_atoms(structure_path):
        if atom.name == "CA" and atom.chain not in chains:
            chains.append(atom.chain)
    return chains


def extract_chain(structure_path: str, chain_id: str, out_path: Path) -> Tuple[str, List[int]]:
    """Write a single-chain PDB and report the chain's sequence and residue numbering.

    Keeping one chain per structure makes the ProteinMPNN outputs unambiguous:
    every residue in the npz belongs to the designed chain, so feature rows line
    up with sequence positions by construction. The output is always PDB, since
    ProteinMPNN's own reader does not handle mmCIF.
    """
    out_chain = pdb_chain_id(chain_id)
    kept: List[str] = []
    seq: List[str] = []
    resnums: List[int] = []
    seen = set()
    for atom in read_atoms(structure_path):
        if atom.chain != chain_id:
            continue
        if atom.hetatm and atom.resname != "MSE":
            continue  # ligands/waters, but keep selenomethionine
        if atom.altloc not in ("", "A"):
            continue  # keep only the first altloc
        kept.append(_pdb_line(len(kept) + 1, atom, out_chain))
        if atom.name == "CA":
            key = (atom.resseq, atom.icode)
            if key in seen:
                continue
            seen.add(key)
            seq.append(THREE_TO_ONE.get(atom.resname, "X"))
            resnums.append(atom.resseq)

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
            "--pdb_path_chains", pdb_chain_id(chain_id),
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
            "--pdb_path_chains", pdb_chain_id(chain_id),
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
