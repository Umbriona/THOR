# THOR — THermally Optimised Representations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1noVdf3ifoMZVk_RWrobONUiMg3d2DpYm)
[![Weights on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20weights-Umbriona%2FTHOR-yellow)](https://huggingface.co/Umbriona/THOR)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

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

Everything runs inside a single Singularity/Apptainer container, defined by
[`env/singularity/thor.recipe`](env/singularity/thor.recipe):

```sh
# ~30 GB of scratch is needed during the build; /tmp is often too small
export SINGULARITY_TMPDIR=$HOME/singularity_tmp && mkdir -p "$SINGULARITY_TMPDIR"
singularity build --fakeroot thor.sif env/singularity/thor.recipe

# check the image can actually run this repository (needs a GPU node)
bash env/singularity/verify_image.sh thor.sif
```

> **Build on a login node, verify on a compute node.** The build installs
> packages from PyPI, and compute nodes on most clusters have no outbound
> internet — there, pip fails with `Could not find a version that satisfies the
> requirement ... (from versions: none)`, which means the index was
> unreachable, not that the version is missing. Verification goes the other
> way: it wants the GPU that login nodes usually lack.
>
> If your site only permits builds on compute nodes, either set `http_proxy` /
> `https_proxy` to the site proxy, or stage the wheels from a networked node
> first:
>
> ```sh
> pip download -d wheelhouse -r env/singularity/requirements.txt   # networked node
> # then add to %post:  pip install --no-index --find-links=/wheelhouse ...
> ```

`verify_image.sh` goes past "do the packages import": it loads THOR's own
modules inside the container and builds the generator and discriminator from
`config/Cycle_gan/Reference_config.yaml`, which is where TensorFlow /
tensorflow-addons version mismatches actually surface. It needs no weights, no
data and no GPU beyond the device check.

`verify_image.sh` goes past "do the packages import": it loads THOR's own
modules inside the container and builds the generator and discriminator from
`config/Cycle_gan/Reference_config.yaml`, which is where TensorFlow /
tensorflow-addons version mismatches actually surface. It needs no weights, no
data and no GPU.

It bootstraps from the public NGC TensorFlow 23.09 image (TensorFlow 2.13,
CUDA 11.8, cuDNN 8.9) and adds PyTorch 2.1.2 with HuggingFace Transformers for
ESM-1v, plus `tensorflow-addons`, `simple_pid`, `biopython` and the rest. The
build ends with an import check, so a missing dependency fails the build rather
than a queued job.

One image covers training, generation, ET training and prediction, ESM-1v
fine-tuning and feature computation:

```sh
singularity exec --nv thor.sif python train_thor.py -c <config>.yaml
```

Not included: `mmseqs2`, used by `data_processing/cluster.sh`, which runs on
the host; `ray` and `seaborn`, needed only by two peripheral scripts
(`dataset_construction/remove_virus.py` and `analysis/temperature_analysis.py`).

Earlier per-step recipes are kept in `env/singularity/` for reference, and the
conda specifications in `env/` describe individual steps.

Hardware: THOR training was run on 4×A100 (80 GB). Generation from a FASTA fits
comfortably on a single A40 or T4.

## Running elsewhere

The SLURM wrappers (`*.sh`) work from a plain `git clone` without editing. Each
resolves the repository root from the directory you submit from, and falls back
to its own location when run directly. Defaults:

| Variable | Default |
|---|---|
| `THOR_ROOT` | the repository root |
| `THOR_IMAGE` | `$THOR_ROOT/thor.sif` |
| `THOR_DATA` | `$THOR_ROOT/data` |
| `THOR_RESULTS` | `$THOR_DATA/results` |
| `THOR_LOG` | `$THOR_ROOT/log` |

Override any of them from the environment, without touching the scripts:

```sh
cd src/scripts
THOR_DATA=/scratch/$USER/thor_data sbatch train_thor.sh
```

Two things remain site-specific, and the scripts say so where they appear:
`#SBATCH --account=CHANGE_ME` must be set to your allocation (or deleted if
your site doesn't use accounts), and the partition line is commented out for
you to fill in. GPU type and walltime in the `#SBATCH` headers reflect the
machine THOR was developed on and will likely need adjusting.

No SLURM? The Python scripts take ordinary command-line arguments and run
directly under `singularity exec` — often the simpler route. See
[`src/README.md`](src/README.md) for each one's interface.

## License

THOR is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free to use,
share and adapt **for non-commercial purposes**, with attribution. See
[`LICENSE`](LICENSE). For commercial use, please get in touch.

Note that this is not an OSI-approved open-source license: it restricts
commercial use.

Parts of this repository derive from third-party works under their own terms,
and the pretrained models and base images pulled at runtime carry their own
licenses — see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

<!-- TODO: add once the preprint is posted -->
