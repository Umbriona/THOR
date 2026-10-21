# Dataset construction

How THOR's training data was built from raw IMG downloads, before the feature
pipeline in `../` runs. Kept for provenance: the released datasets are already
built, so these are not needed to reproduce results from the published data —
they document where that data came from.

Run order, roughly:

| Step | Script | What it does |
|---|---|---|
| 1 | `Extract_IMG.py` | Unpacks IMG tarballs and joins sequences to their organism metadata, emitting FASTA with temperature in the header. |
| 2 | `data_clean.py` | Drops sequences with invalid residues or out-of-range lengths. |
| 3 | `remove_virus.py` | Removes viral entries and applies EC-based filtering. |
| 4 | `Combine_data.py` | Merges FASTA files, with `--min_length` / `--max_length` bounds. |
| 5 | `split_fasta.py` | Sorts by temperature and splits into the meso / thermo sets. |
| 6 | `make_data_splits.py` | Produces the train / validation / test splits. |

Redundancy reduction is separate, in `../cluster.sh` (mmseqs `easy-cluster` at
70/60/50/40% identity).

Paths inside these scripts point at the original IMG download locations and
will need editing to re-run.
