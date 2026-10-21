# Archived data-processing scripts

Superseded or unused parts of the data pipeline, kept for provenance. **None of
these are used.** See `../` for the live pipeline and `../../archive/` for
archived training, generation and ET scripts.

## Feature generation, earlier paths

`../compute_MLM_features.py` (formerly `compute_MLM_features_quantized_tfrecord.py`)
is the single live feature step. These preceded it:

| File | Notes |
|---|---|
| `compute_MLM_features_quantized.py` | The pickle-only predecessor, 585 lines against 900. A **strict subset**: the live script contains the same `save_sharded` pickle writer plus `save_grouped_pickle`, and selects between formats with `--format {pickle,tfrecord}`. Pickle output is now `../compute_MLM_features_pickle.sh`. |
| `generate_MLM_probability_features.py` + `bert_class.py` + `compute_features.sh` | The ProtBert-era feature path, replaced by ESM-1v. `generate_MLM_probability_features.py` imports `BertPatcher` / `BertLikelihood` from `bert_class.py`, so the three are one unit. |

## Superseded helpers

Oct 2025, no longer referenced by anything in the live pipeline.

| File | What it did |
|---|---|
| `create_data.py` | Built histogram-normalised training tables. |
| `calc_msa_clust.py` | MSA-based cluster structure. |
| `filter_cogs.py` | COG filtering. |
| `make_test_training.py` | An early train/test split. |
| `reorder_fasta.py` | Reordered FASTA records. |
| `scale_fastas.py` | Upscaled FASTA sets to balance counts. |
| `uniform_distribution.py` | Resampled toward a uniform temperature distribution. |

## TFRecord writers

`compute_MLM_features_quantized_tfrecord.py` computes the ESM features *and*
writes the model-ready TFRecords in one step (`tf.io.TFRecordWriter`,
`--format tfrecord`, gzip). That removed the need for a separate record-writing
stage, so these are obsolete:

| File | Notes |
|---|---|
| `make_record_ThermalGAN.py` | The last of the three to be used. Differs from `make_records_simple.py` by a single line — `nargs="+"` on `--input`, so it accepts several FASTAs rather than one. |
| `make_records_simple.py` | Single-input variant of the above; otherwise byte-identical. |
| `make_record_old.py` | Earlier, shorter version (128 lines). Lacks the `_bytes_feature` / `get_class_sizes` / `readme` helpers the later two share, and scales sequences via a `scale_sequences` step the others dropped. |

## Analysis carried over from earlier work

Not part of THOR. These came from the earlier kcat/BRENDA work and contain
hardcoded `/Users/niktat/...` paths.

| File | Notes |
|---|---|
| `blastsKcats.py` | BLAST / kcat analysis and plotting. |
| `volcano.py` | Near-identical to `blastsKcats.py` — same nine functions, plus `scipy.stats` and `math.log` imports for the volcano plot. |
| `topCogs.py` | Selects top COGs; its module docstring is a commented-out block of an earlier script. |

## Pipeline definition

| File | Notes |
|---|---|
| `Snakefile` | Never finished, and **does not parse**: several `shell:` directives contain unquoted paths, and one `rule` line is missing its trailing colon. Three rules are stubs marked `#TODO` with empty `input:`/`shell:` blocks. Its only reference to the rest of the tree was `make_record_ThermalGAN.py`, archived alongside it. |

## Duplicate SLURM wrapper

| File | Notes |
|---|---|
| `compute_MLM_features_quantized_copy.sh` | A copy of `../compute_MLM_features_quantized.sh` made only to run a second SLURM array over a different dataset concurrently. Not a separate pipeline step. |

The surviving wrapper covers Uniprot enzymes. The settings this copy used for
the IMG thermophiles run, should they need reproducing:

```sh
#SBATCH --array=50-99
FILE_FASTA=${BASE}/data/THOR_HUGE/fasta/IMG_Thermophiles/split_files/IMG_Thermophiles_NoCOG_${SLURM_ARRAY_TASK_ID}.fasta
DIR_FEATURE=${BASE}/data/THOR_HUGE/records/IMG_Thermophiles
--name IMG_Thermophiles_NOCOG_${SLURM_ARRAY_TASK_ID}
```

Everything else was identical, including the call to
`compute_MLM_features_quantized_tfrecord.py`.
