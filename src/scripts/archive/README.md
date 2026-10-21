# Archived training scripts

Superseded versions of the THOR (formerly ThermalGAN) training entry point,
kept for provenance. **None of these are used.** The current training scripts
are `../train_thor.sh` (SLURM wrapper) and `../train_thor.py`.

## Lineage

All five Python files are variants of the same CycleGAN training loop. In the
order they were developed:

| File | What it added | Superseded by |
|---|---|---|
| `train_cyclegan.py` | Base version of the training loop. | all of the below |
| `train_cyclegan-restart.py` | Unconditional `model.load_weights(.../cycle_gan_model_final)` to resume a run. | the conditional resume in `-diff_training.py` |
| `train_cyclegan-diff_training.py` | Fractional training (`config['CycleGan']['Fractional_training']` alternates `train_step_bert_generator` / `train_step_generator_bert_all`), `history.csv` dump, conditional resume from `config['CycleGan']['saved']`. | `train_thor.py` |
| `train_cyclegan-diff_training_3.py` | As `-diff_training.py`, but fractional alternation applied to the `train_step_bert_all` branch instead. Dead-end branch; no wrapper ever called it. | `train_thor.py` |
| `train_cyclegan-diff_training_2.py` | Argparse moved above the TensorFlow import so `CUDA_VISIBLE_DEVICES` is set before TF initialises; `mixed_bfloat16` policy, TF32, profiler hooks, multi-GPU. | — **renamed to `../train_thor.py`** |

## Wrappers

| File | Called | Why retired |
|---|---|---|
| `train_thermalgan.sh` | `train_cyclegan.py` | Local (non-SLURM) runner, hardcoded 2022 config path. |
| `train_thermalgan_slurm-restart.sh` | `train_cyclegan-diff_training.py` | Expired allocation `NAISS2023-22-617`. |
| `train_thermalgan_temdir.sh` | `train_cyclegan-diff_training.py` | Variant that rsynced data to `$TMPDIR`; expired allocation. |

The current wrapper, `../train_thor.sh`, was `train_thermalgan_slurm.sh`.
