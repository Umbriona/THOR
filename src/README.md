# Scripts reference

What each script does and what it takes. See the [root README](../README.md)
for an overview and setup; see `scripts/archive/README.md` and
`scripts/data_processing/archive/README.md` for retired scripts.

Each `.sh` is a SLURM wrapper that runs the matching `.py` inside a Singularity
container. **The wrappers carry absolute cluster paths and SLURM account names
that you will need to edit.** The Python scripts take ordinary arguments and
can be run directly.

---

## Generation

### `scripts/generate_thor.py` — generate variants from a FASTA
The main entry point for using a trained model. Self-contained: it computes
ESM-1v per-position likelihoods inline, so no TFRecord preprocessing is needed.
Writes a FASTA of variants plus per-variant metrics (generator NLL, ESM NLL,
predicted temperature), and optionally a JSONL of softmax distributions.

| Argument | |
|---|---|
| `--run_dir` | **required.** Trained run directory, containing `config.yaml` and `weights/epoch_<n>/` |
| `--fasta` *or* `--sequence` | input sequences, or a single raw sequence |
| `--weights` | weights directory *relative to `--run_dir`* — `weights_85` for a downloaded model, or `weights/epoch_39` for one of your own runs. The generator is loaded from `<run_dir>/<weights>/generator_G.h5` |
| `--replicates` | variants per input sequence (default 1) |
| `--temperature` | sampling temperature (default 1.0) |
| `--esm_filter_threshold` | drop variants whose ESM NLL exceeds this |
| `--filter_opt_cycles` | rounds of ESM-guided argmax optimisation |
| `--store_softmax` | also write per-position probabilities as JSONL |
| `--skip_ogt` | skip temperature prediction |
| `--output_dir`, `--name`, `--gpu`, `--device`, `--batch_tokens` | |

Wrapper: `generate_thor.sh`. Notebook: `generate_thor_colab.ipynb`.

### `scripts/generate_thor_batch.py` — generate across a whole dataset
Dataset-scale generation from **pre-built TFRecords**, used for the large-scale
experiments. Loads the full CycleGAN (with PID controllers) rather than the
generator alone, and writes 50 replicates per sequence.

| Argument | |
|---|---|
| `-i/--input` | **required.** Run directory holding `config.yaml` and `weights/` |
| `--epoch` | epoch to load |
| `--store_softmax` | also dump softmax probabilities |
| `-g/--gpu`, `-v/--verbose` | |

> Its input TFRecord glob and the ET weight paths are hardcoded in the body of
> the script (around line 110) and must be edited to point at your data.

Wrapper: `generate_thor_batch.sh`, which sweeps epochs 3–39.

---

## Training

### `scripts/train_thor.py` — train the THOR generator
CycleGAN training with mixed `bfloat16`, TF32 and multi-GPU support.
Everything is driven by the config file; the run writes weights, `config.yaml`
and `history.csv` into a timestamped results directory.

| Argument | |
|---|---|
| `-c/--config` | YAML config (see `config/Cycle_gan/`) |
| `-g/--gpu` | comma-separated GPU ids |
| `-v/--verbose` | |

Start from **`config/Cycle_gan/Reference_config.yaml`**, the current reference —
it is the only config in that directory that matches the present schema; the
rest are archived under `config/Cycle_gan/archive/`.

Config sections: `CycleGan` (architecture, `Losses`, `Optimizers`,
`LR_Schedule`, PID settings, `Fractional_training`), `Classifier` (the ET
critic), `Data_meso` / `Data_thermo` (TFRecord locations, `records_type: ESM`),
`Log`, `Results`.

Wrapper: `train_thor.sh` — 4×A100, 3-day limit, picks a config by
`$SLURM_ARRAY_TASK_ID`.

#### Fine-tuning on a specific family

To continue from an existing model rather than training from scratch — the
family-specific fine-tuning step — point `saved` at a previous run's results
directory in the config:

```yaml
CycleGan:
    saved: "/data/pretrained"      # the downloaded Hugging Face repo
    saved_weights: "weights_85"    # published models: 70, 75, 80, 85, 90, 95
    lambda_self: 0.85              # must match: weights_NN <-> 0.NN
```

The six published models share one architecture and differ only in
`lambda_self`, the PID setpoint that holds generated variants near the target
sequence identity. Loading one while leaving `lambda_self` at another value
pulls the model away from what it was trained for.

`saved_weights` names a directory inside `saved` holding the four `.h5` files
for G, F and both discriminators, plus optional `PID_G.pkl` / `PID_F.pkl`.

For continuing one of your own runs, `saved_epoch: 39` is shorthand for
`saved_weights: "weights/epoch_39"`, matching the layout training writes. With
`saved` set but neither key, the end-of-run checkpoint
`<saved>/weights/cycle_gan_model_final` is used instead. Set `saved` to
`"none"` or leave it out to train from scratch.

Fine-tuning restores weights and PID controller state; optimizers start fresh,
so give the run a learning-rate warmup.

Point `Data_meso` / `Data_thermo` at the family's records, and everything else
proceeds as a normal training run.

### `scripts/train_et_regressor.py` — train the temperature regressor
Trains the environmental-temperature (ET) model used as THOR's critic and to
score variants. Despite the `Classifier` naming kept in the config schema, this
is a **regression** model: MSE loss, R² metric. Three independently trained
copies form the ensemble.

| Argument | |
|---|---|
| `-c/--config` | YAML config (see `config/Classifier/`) |
| `-g/--gpu`, `-o/--output`, `-v/--verbose` | |

Wrapper: `train_et_regressor.sh`.

### `scripts/finetune_ESM.py` — fine-tune ESM-1v (exploratory)
Masked-LM fine-tuning of ESM-1v on your own sequences, so the likelihoods
guiding generation reflect the target family. Multi-GPU via `torchrun` (DDP).

> **Not used for the published results.** This was tried and made no
> appreciable difference, so THOR uses stock `facebook/esm1v_t33_650M_UR90S_1`
> throughout. The script is kept because the experiment is part of the record,
> and because fine-tuning may help on families further from ESM's training
> distribution. Nothing else in the pipeline depends on it.

| Argument | |
|---|---|
| `--data_path` | **required.** FASTA, or a text file with one sequence per line |
| `--output_dir` | **required.** |
| `--model_name` | default `facebook/esm1v_t33_650M_UR90S_1` |
| `--val_path`, `--epochs`, `--per_device_batch`, `--accum`, `--lr`, `--warmup_steps`, `--max_length` | |
| `--bf16`, `--resume auto` | |

Wrapper: `finetune_ESM.sh` — 4×A100 via `torchrun`.

---

## Prediction

### `scripts/predict_et.py` — predict temperature for a FASTA
Runs the trained ET ensemble over a FASTA and writes a new FASTA with the
predicted temperature appended to each header.

```sh
python predict_et.py <in.fasta> <out.fasta>
```

`--config`, `--weights`, `--batch-size` and `--max-len` default to the
container layout and can be overridden. Wrapper: `predict_et.sh`.

### `scripts/convert_ogt_savedmodel_to_h5.py` — checkpoint conversion
Converts legacy ET SavedModel checkpoints (`variables/variables`) into `.h5`.
Takes `--config`, `--ogt_dir`, `--output_dir`, `--models`.

---

## Data processing — `scripts/data_processing/`

### `compute_MLM_features.py` — the one feature step
Runs ESM-1v over a FASTA, quantizes the per-position probability
distributions, and shards the result straight to disk. **This both computes
features and writes the model-ready records** — there is no separate
record-building stage.

| Argument | |
|---|---|
| `-i/--input` | **required.** FASTA (temperature may be the last header token) |
| `-o/--output` | **required.** output directory |
| `--format` | `tfrecord` (default, gzip) or `pickle` |
| `--model` | ESM model: a HF id (default `facebook/esm1v_t33_650M_UR90S_1`, as used for the published results) or a local directory |
| `--mode` | `single` or random-partition masking |
| `--quantize` | `none` / `uint8` |
| `--max_entries` | records per shard |
| `--batch_tokens`, `--max_length`, `--bf16`, `--name` | |
| `--min_quality`, `--quality_metric` | drop low-confidence sequences |

Wrappers: `compute_MLM_features_tfrecord.sh` (SLURM array, the production run)
and `compute_MLM_features_pickle.sh` (same script, `--format pickle`).

### Record handling
| Script | Purpose |
|---|---|
| `filter_records.py` | Drops records whose id hash appears in `--id_hashes_npy`; used to remove test-set overlap. Takes `--input_dir`, `--out_dir`, `--shard_size_mb`. |
| `filter_records_entropy.py` | Filters by per-position entropy (`--entropy_min` / `--entropy_max`), removing low-information sequences. Takes `--input_dir`, `--feature_name`, `--out_dir`. |
| `check_record_lengths.py` | Scans shards and reports the sequence-length distribution. Takes `-d/--directory`. |
| `combine_tfrecords.py` | Merges shards into one file. **Takes no arguments** — edit the `pattern` and `out_path` in the call at the bottom. Wrapper: `combine_tfrecords.sh`. |

### Other
| Script | Purpose |
|---|---|
| `make_embeddings.py` | Average ESM embeddings per sequence, written as a pickle. Takes `-i/--input`, `-o/--output`, `--batch_size`, `--max_entries`. Wrapper: `make_embeddings.sh`. |
| `cluster.sh` | mmseqs `easy-cluster` at 70/60/50/40% identity, extracting the top 100 clusters at each threshold. This is how redundancy between train and test was controlled. Edit `FASTA_FILE` at the top. |

### `data_processing/dataset_construction/`
How the training data was built from raw IMG downloads — kept as a provenance
trail rather than a pipeline you need to re-run. See the README there for the
step order.

---

## `utils/`

Model and data definitions imported by the scripts above.

| Module | Contents |
|---|---|
| `models_cyclegan_self_loss.py` | The CycleGAN used by `train_thor.py` and `generate_thor_batch.py`: training steps, PID-controlled loss weighting, checkpointing |
| `models_gan_atte.py` | The generator architecture. `generate_thor.py` builds this directly to load G alone |
| `models_generator.py`, `models_discriminator.py` | Generator and discriminator components |
| `models_classifyer.py` | The ET regressor (`get_classifier`) — note the spelling and the classifier-era naming |
| `layers_new.py`, `layers_residual.py` | Self-attention, Gumbel-softmax, spectral normalisation, residual blocks |
| `preprocessing.py` | TFRecord parsing and dataset assembly |
| `loaders.py`, `losses.py`, `metrics.py`, `callbacks.py` | Losses, optimizers and metrics resolved from config |
