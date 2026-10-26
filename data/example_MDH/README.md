# Example: fine-tuning THOR on malate dehydrogenase

A small, self-contained dataset for trying THOR's fine-tuning path end to end —
from FASTA, through ESM-1v feature computation, to a training run that starts
from the published weights.

**This demonstrates the mechanics, not the published result.** See
[Caveats](#caveats).

## Files

| File | Sequences | |
|---|---|---|
| `MDH_finetuning_set.fasta` | 3553 | the set as supplied, mesophiles and thermophiles together |
| `meso_train.fasta` | 3137 | 4–44 °C |
| `meso_val.fasta` | 349 | 4–44 °C |
| `thermo_train.fasta` | 60 | 60–98 °C |
| `thermo_val.fasta` | 7 | 60–85 °C |

Headers are `>ID Label Temperature`, e.g. `>Q8ZVB2 Thermophile 98.0`. The
pipeline reads the temperature from the **last** whitespace-separated token, so
this format works unchanged.

The splits are derived from the combined file by the label in the header, then
divided into train/validation by sorting on temperature and taking every tenth
sequence for validation. That is deterministic (no random seed) and keeps the
validation split spanning the full temperature range — which matters for the
67 thermophiles, where a random 10% could easily miss one end.

Validation is not optional here: the PID controllers that balance the
self-consistency loss are driven by validation accuracy, so a run without
validation records will not behave correctly.

## Running it

**1. Compute ESM features and write TFRecords** (needs a GPU):

```sh
bash data/example_MDH/prepare_example.sh
# or: sbatch data/example_MDH/prepare_example.sh
```

This runs `compute_MLM_features.py` over all four splits in `--mode random`,
the same mode used to build the published training records. Each sequence is
masked 16 times at 15% and the per-position probabilities are averaged and
quantized to uint8. Expect a few minutes on an A100, longer on a smaller card.

> `--mode single` also exists and is more thorough — it is leave-one-out, one
> forward pass per position. On this set that is roughly 1.1 million passes
> instead of 57 thousand, so it is not the right choice for a quick try.

ESM-1v is downloaded from Hugging Face on first use. Set `ESM_MODEL` to a local
directory to avoid re-downloading it on a cluster without outbound network.

**2. Fetch the pretrained weights** from
[Hugging Face](https://huggingface.co/Umbriona/THOR) into `data/pretrained/`,
which the container sees as `/data/pretrained`:

```
pretrained/
  weights_70/   config.yaml
                generator_G.h5      generator_F.h5
                discriminator_x.h5  discriminator_y.h5
                PID_G.pkl  PID_F.pkl  optimizer_*.pkl
  weights_75/   ...   weights_80/  ...   weights_85/  ...
  weights_90/   ...   weights_95/  ...
```

The models are named by the **target sequence identity** they were trained for:
`weights_85` holds generated variants around 85% identical to the input. All
six share one architecture and differ only in `lambda_self`, the PID setpoint
that enforces that identity.

Pick one with `CycleGan.saved_weights` in
`config/Cycle_gan/MDH_finetune.yaml` (default `weights_85`) — **and set
`lambda_self` to match**: `weights_70` goes with `lambda_self: 0.70`. The
config says so at both keys.

**3. Fine-tune:**

```sh
singularity exec --nv thor.sif python3 \
    /ThermalGAN/src/scripts/train_thor.py \
    -c /ThermalGAN/config/Cycle_gan/MDH_finetune.yaml
```

2000 steps at batch 32. Weights, `config.yaml` and `history.csv` are written to
a timestamped directory under `Results.base_dir`.

**4. Generate variants** from the fine-tuned model:

```sh
singularity exec --nv thor.sif python3 \
    /ThermalGAN/src/scripts/generate_thor.py \
    --run_dir <the timestamped results directory> \
    --weights weights/epoch_19 \
    --fasta /data/example_MDH/thermo_val.fasta \
    --replicates 5
```

To generate from a downloaded model instead of your fine-tuned one, point
`--run_dir` straight at it — each holds its own `config.yaml`, so no
`--weights` is needed:

```sh
    --run_dir /data/pretrained/weights_85
```

## What fine-tuning restores

Warm start, not an exact resume:

- **Restored:** both generators and both discriminators (`generator_G.h5`,
  `generator_F.h5`, `discriminator_x.h5`, `discriminator_y.h5`), and — when
  `PID_G.pkl` / `PID_F.pkl` are present — the PID controllers' accumulated
  state, so the self-loss weighting continues rather than re-converging from
  zero. A missing PID file is not an error; the controller simply starts fresh.
- **Fresh:** the optimizer state. Adam moments from the pretraining corpus do
  not necessarily help on a new distribution, and starting them clean is the
  safer default. This is why the learning-rate warmup in the config matters —
  a cold start at full rate can undo pretrained weights in the first few steps.
- **From your config, not the checkpoint:** the PID gains and setpoint, so
  changing `PID_P` or `lambda_self` here still takes effect.

## Caveats

- **The class balance is extreme: 3486 mesophiles to 67 thermophiles**, about
  52:1. THOR learns between unpaired populations so this is not fatal, but it is
  far from the regime the published models were trained in.
- **2000 steps is a demonstration length.** It is enough to see the losses move
  and to produce variants; it is not enough to reproduce the melting-temperature
  gains reported in the paper.
