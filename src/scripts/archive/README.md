# Archived scripts

Superseded versions of the THOR (formerly ThermalGAN) training and generation
entry points, kept for provenance. **None of these are used.** The current
scripts are:

| Purpose | Wrapper | Python |
|---|---|---|
| Training | `../train_thor.sh` | `../train_thor.py` |
| Generation, from FASTA | `../generate_thor.sh` | `../generate_thor.py` |
| Generation, from TFRecords | `../generate_thor_batch.sh` | `../generate_thor_batch.py` |
| ET regressor, training | `../train_et_regressor.sh` | `../train_et_regressor.py` |
| ET regressor, inference | `../predict_et.sh` | `../predict_et.py` |

## Training lineage

All five Python files are variants of the same CycleGAN training loop. In the
order they were developed:

| File | What it added | Superseded by |
|---|---|---|
| `train_cyclegan.py` | Base version of the training loop. | all of the below |
| `train_cyclegan-restart.py` | Unconditional `model.load_weights(.../cycle_gan_model_final)` to resume a run. | the conditional resume in `-diff_training.py` |
| `train_cyclegan-diff_training.py` | Fractional training (`config['CycleGan']['Fractional_training']` alternates `train_step_bert_generator` / `train_step_generator_bert_all`), `history.csv` dump, conditional resume from `config['CycleGan']['saved']`. | `train_thor.py` |
| `train_cyclegan-diff_training_3.py` | As `-diff_training.py`, but fractional alternation applied to the `train_step_bert_all` branch instead. Dead-end branch; no wrapper ever called it. | `train_thor.py` |
| `train_cyclegan-diff_training_2.py` | Argparse moved above the TensorFlow import so `CUDA_VISIBLE_DEVICES` is set before TF initialises; `mixed_bfloat16` policy, TF32, profiler hooks, multi-GPU. | — **renamed to `../train_thor.py`** |

## Training wrappers

| File | Called | Why retired |
|---|---|---|
| `train_thermalgan.sh` | `train_cyclegan.py` | Local (non-SLURM) runner, hardcoded 2022 config path. |
| `train_thermalgan_slurm-restart.sh` | `train_cyclegan-diff_training.py` | Expired allocation `NAISS2023-22-617`. |
| `train_thermalgan_temdir.sh` | `train_cyclegan-diff_training.py` | Variant that rsynced data to `$TMPDIR`; expired allocation. |

The current wrapper, `../train_thor.sh`, was `train_thermalgan_slurm.sh`.

## Generation

| File | Why retired |
|---|---|
| `ThermalGAN_generate_sequences.py` | Built on `utils.models_cyclegan`, the older generator architecture; current scripts use `utils.models_gan_atte` (directly, or via `utils.models_cyclegan_self_loss`). Also hardcodes `/vault/ThermalGAN` paths. |
| `generate_thermalgan_slurm_vera.sh` | Vera-cluster wrapper for what is now `../generate_thor_batch.py`; expired allocation `C3SE2023-1-18`. The Alvis equivalent lives on as `../generate_thor_batch.sh`. Its `python test_generate_ThermalGAN.py` line refers to the pre-rename filename. |

Renamed, not retired: `generate_variants_single.py` → `../generate_thor.py`,
`test_generate_ThermalGAN.py` → `../generate_thor_batch.py` (despite the
`test_` prefix, it was never a test — it is the dataset-scale generation path).

## ET regressor (temperature predictor)

The published temperature model is a one-hot CNN regressor, trained by
`../train_et_regressor.py` (formerly `train_classifier.py`). These variants are
not used:

| File | Why retired |
|---|---|
| `train_classifier_onehot.py` | Abandoned mid-edit and **cannot run**: calls `models_class.get_classifier` though the module is imported as `models`, and passes `x_train` / `x_val` to `model.fit` after the lines defining them were deleted. Its `models_weights` / `new_model_weights` lists are never used, so the fine-tuning it was written for never happens. Kept only as a record of the intent — fine-tuning the OGT ensemble on IMG data. |
| `train_classifier_embeddings.py` + `.sh` | PyTorch regressor over precomputed ESM embeddings (h5). An alternative architecture that was not the implementation used in the paper. |
| `evaluate_classifier_embeddings.py` + `.sh` | Evaluation counterpart to the above. The wrapper also passes `-c ${CONFIG_FILE}` without ever setting `CONFIG_FILE`. |

## Evaluation

| File | Why retired |
|---|---|
| `evaluate_cyclegan.py` | **Cannot run**: imports `utils.models_new`, which does not exist in `src/utils/`. Predates the current model modules. |
| `get_temperature_prediction.py` + `predict_temperatures.sh` | One-off evaluation of the ET ensemble on held-out IMG records, writing true-vs-predicted CSVs. Takes no arguments; every path is hardcoded in the body. **Also buggy — see below.** The reviewer-facing inference path is `../predict_et.py`. |

### Known bugs in `get_temperature_prediction.py`

Recorded because the CSVs it produced may be wrong. Both are in `main()`:

1. **The output files are mislabelled.** The loop at line 96 iterates
   `batches_y` (the *thermo* split) but writes
   `Temperature_OGT_IMG_Mesophiles.csv`; the loop at line 111 iterates
   `batches_x` (*meso*) and writes `..._Thermophiles.csv`. The two progress
   messages are swapped to match the wrong names.
2. **The accumulators are never reset.** `ids`, `true_temps` and `pred_temps`
   are initialised once (lines 91–93); the second loop appends to the same
   lists. So `..._Thermophiles.csv` holds the thermo rows *and* the meso rows,
   duplicating everything already in the first file.

A `+ 50.5` offset is applied to the true temperatures in both loops without
explanation.

## Data processing

| File | Why retired |
|---|---|
| `compute_MLM_features_quantized_copy.sh` | A duplicate of `../data_processing/compute_MLM_features_quantized.sh`, made only to run a second SLURM array over a different dataset at the same time. Not a separate pipeline step. |

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

## `scratch/`

Notebooks and stubs kept only so nothing is lost; none are referenced by any
script. The three notebooks were all named `Untitled.ipynb` and were renamed by
origin on the way in.

| File | Was | What it is |
|---|---|---|
| `Untitled_scripts.ipynb` | `src/scripts/` | Loads the generator and prints `summary()`. |
| `Untitled_utils.ipynb` | `src/utils/` | Builds a one-layer `Sequential` around `GumbelSoftmax` to check it. |
| `Untitled_data_processing.ipynb` | `src/scripts/data_processing/` | Two cells testing multi-line string printing. |
| `test_scrip.sh` | `src/scripts/` | SLURM hello-world (`echo` + `sleep 60`), used to verify queue access. |
| `sample_fasta.py` | `src/scripts/data_processing/` | Empty file (0 bytes), referenced nowhere. |
| `models_cyclegan_self_loss.py.save` | `src/utils/` | Editor backup of an **older** state of `models_cyclegan_self_loss.py` — lacks `PIDWrapper`, `MAX_LEN`, `USE_XLA` and two imports present in the live module. Superseded by the file it backs up. |

Naming note: "ET" (environmental temperature) is the term used in the paper.
The codebase still says `OGT` in weights paths and `Classifier` in config keys
and in `utils/models_classifyer.py`; those were deliberately left unchanged, as
they are referenced by saved configs and by the trained weights on disk.
