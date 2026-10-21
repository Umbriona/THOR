# THOR — THermally Optimised Representations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1noVdf3ifoMZVk_RWrobONUiMg3d2DpYm)
[![Weights on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20weights-Umbriona%2FTHOR-yellow)](https://huggingface.co/Umbriona/THOR)

**Try it in your browser:** paste a protein sequence into the
[Colab notebook](https://colab.research.google.com/drive/1noVdf3ifoMZVk_RWrobONUiMg3d2DpYm)
and get thermostable variants back. No install, no weights to download by hand.

THOR generates thermostable variants of a protein while preserving the identity
of the starting sequence. It learns transformations between **unpaired**
mesophilic and thermophilic sequence populations, so it can raise the
thermostability of enzyme families it never saw thermophilic examples of during
training.

Architecturally it is a CycleGAN over protein sequences: a generator maps a
mesophilic sequence to a thermophilic counterpart, constrained by per-position
amino-acid likelihoods from ESM-1v and by a temperature regressor that scores
how thermophilic the result looks.

<img src="Covver_image_small.png" alt="THOR" width="420">

<details>
<summary><b>Abstract</b></summary>

Thermal adaptation has evolved repeatedly across the protein universe, yet
whether its underlying sequence principles can be separated from evolutionary
ancestry remains unknown. Here, we show that thermal adaptation follows
transferable sequence principles that extend beyond the evolutionary histories
of individual enzyme families. We developed THOR (THermally Optimised
Representations), a generative framework that learns transformations between
unpaired mesophilic and thermophilic sequence populations while preserving the
identity of the starting protein. Across evolutionarily divergent enzyme
families, THOR transferred thermostability without exposure to thermophilic
homologues during training. In malate dehydrogenase, excluded from cross-family
training, zero-shot designs increased apparent melting temperature by up to
15 °C, with gains reaching 36 °C following limited family-specific
fine-tuning. Majority of designed variants retained activity following
prolonged thermal challenge, while selected variants exhibited nearly threefold
higher catalytic turnover than wild type. Molecular dynamics simulations
revealed that comparable thermal phenotypes emerge through distinct,
background-dependent changes in protein flexibility and stabilising
interactions. Together, these findings demonstrate that thermal adaptation
encodes transferable information beyond lineage-specific evolutionary
solutions, revealing evolution as a source of reusable adaptive principles for
engineering phenotypes in proteins whose evolutionary lineages have never
acquired them.

</details>

## The three models

| Model | What it is | Trained by |
|---|---|---|
| **THOR generator** | CycleGAN generator + discriminators over sequence space | `src/scripts/train_thor.sh` |
| **ET regressor** | CNN predicting environmental temperature from sequence; used as the GAN's temperature critic and to score variants | `src/scripts/train_et_regressor.sh` |
| **ESM-1v (fine-tuned)** | Masked-LM providing per-position substitution probabilities | `src/scripts/finetune_ESM.sh` |

## Quick start

The [Colab notebook](https://colab.research.google.com/drive/1noVdf3ifoMZVk_RWrobONUiMg3d2DpYm)
is the fastest route: it clones this repository, pulls the pretrained weights
from [Hugging Face](https://huggingface.co/Umbriona/THOR) and runs a single
sequence end to end. It is version-controlled here as
[`src/scripts/generate_thor_colab.ipynb`](src/scripts/generate_thor_colab.ipynb).

To run locally, generating variants from a FASTA needs only a trained run
directory — no TFRecord preprocessing:

```sh
cd src/scripts
python generate_thor.py \
    --run_dir    <results/TIMESTAMPconfig_NN.yaml> \  # trained run: config.yaml + weights/
    --weights    weights/epoch_39 \                   # relative to --run_dir
    --fasta      <input.fasta> \
    --replicates 10 \
    --output_dir <out/>
```

This computes ESM likelihoods inline, runs the generator, and writes a FASTA of
variants plus per-variant metrics (generator and ESM negative log-likelihood,
predicted temperature). `src/scripts/generate_thor_colab.ipynb` wraps the same
script for a single pasted sequence.

Training the generator from scratch instead:

```sh
cd src/scripts
sbatch --array=0 train_thor.sh      # -> train_thor.py -c <config>.yaml
```

To **fine-tune on a specific family** rather than train from scratch, set
`saved` in the config to a previous run's results directory and point
`Data_meso` / `Data_thermo` at that family's records:

```yaml
CycleGan:
    saved: "/path/to/results/20251218-225521config_67.yaml"
    # saved_epoch: 39     # optional: a specific epoch, else the final checkpoint
```

Full descriptions of every script and its arguments: **[`src/README.md`](src/README.md)**.

## Repository layout

```
config/
  Cycle_gan/      THOR configs - start from Reference_config.yaml
  Classifier/     ET regressor configs
env/              Conda environment files and Singularity recipes
src/
  scripts/        Entry points - training, generation, prediction
    data_processing/    Feature computation and TFRecord pipeline
      dataset_construction/   How the training data was built from IMG
      archive/              Superseded scripts, with provenance notes
    archive/        Superseded scripts, with provenance notes
  utils/          Model definitions, layers, losses, data loaders
```

Each `archive/` directory has a README explaining what each retired script was
and what replaced it.

## Dependencies

Everything runs inside Singularity/Apptainer containers. The recipes are in
`env/singularity/`:

| Image | Recipe | Used by |
|---|---|---|
| `thermalgan-image1.sif` | `thermalgan-image1.recipie` | THOR training and generation (TensorFlow 2.13, CUDA) |
| `esm-finetune.sif` | `esm-finetune.recipe` | ESM-1v fine-tuning and feature computation (PyTorch 2.3.1, CUDA 12.1, HF Transformers) |

Build with:

```sh
singularity build thermalgan-image1.sif env/singularity/thermalgan-image1.recipie
```

Core requirements, if you prefer to assemble an environment by hand:
TensorFlow 2.13 (+ `tensorflow-addons`), PyTorch 2.3, `transformers`,
`biopython`, `pyyaml`, `pandas`, `numpy`, `simple_pid`. Conda specifications
for individual steps are in `env/`.

Hardware: THOR training was run on 4×A100 (80 GB). Generation from a FASTA fits
comfortably on a single A40 or T4.

## Running elsewhere

> **The SLURM wrappers (`*.sh`) carry absolute paths from the cluster THOR was
> developed on**, under `/mimer/NOBACKUP/groups/snic2022-6-127/...`, together
> with SLURM account names that will not exist for you. Before running any
> wrapper, edit the `SRC_DIR`, `DATA_DIR` and `IMAGE` variables near the top and
> replace `#SBATCH --account=...`.

The Python scripts themselves take ordinary command-line arguments and can be
run directly, which is usually the simpler route — see
[`src/README.md`](src/README.md) for each one's interface.

## Citation

<!-- TODO: add once the preprint is posted -->
