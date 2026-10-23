# THOR - full pipeline container.
#
# This is a translation of ../singularity/thor.recipe, kept for publishing a
# prebuilt image to a container registry (GitHub Actions builds this file for
# linux/amd64 and linux/arm64 and pushes to ghcr.io). Reviewers can then pull
# instead of building:
#
#     singularity pull thor.sif docker://ghcr.io/umbriona/thor:latest
#
# KEEP IN SYNC with ../singularity/thor.recipe and ../singularity/requirements.txt.
# The CI workflow has a `check-pins` job that fails if the pinned versions in
# the three files disagree.

FROM nvcr.io/nvidia/tensorflow:23.09-tf2-py3

LABEL org.opencontainers.image.title="THOR" \
      org.opencontainers.image.description="THermally Optimised Representations - full pipeline" \
      org.opencontainers.image.source="https://github.com/Umbriona/THOR" \
      org.opencontainers.image.licenses="CC-BY-NC-4.0"

ENV TF_CPP_MIN_LOG_LEVEL=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

# TensorFlow 2.13.0+nv23.9 requires numpy<=1.24.3 and the base image's numba
# requires numpy<1.25, but several packages below will pull numpy 1.26 unless
# held. The constraints file applies to every pip call that follows.
RUN python3 -m pip install --upgrade pip \
 && printf 'numpy<1.25\n' > /opt/thor-constraints.txt
ENV PIP_CONSTRAINT=/opt/thor-constraints.txt

# NOT installed: tensorflow-addons. THOR never constructs tfa layers, it has no
# aarch64 wheels, and it is end-of-life. See src/utils/tfa_compat.py.
RUN pip install tensorboard-plugin-profile==2.13.0

# PyTorch is the only architecture-dependent piece. CUDA-enabled aarch64 wheels
# exist only from torch 2.4 and only on the cu124/cu126 indexes; PyPI's aarch64
# wheels are CPU-only at every version. Note --index-url (not --extra-index-url)
# for aarch64: PyPI also has torch 2.5.1 for aarch64 as a CPU build, and pip may
# prefer it when the version matches.
RUN ARCH="$(uname -m)" \
 && echo "[info] building for ${ARCH}" \
 && if [ "$ARCH" = "aarch64" ]; then \
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cu124 ; \
    else \
        pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
            --extra-index-url https://download.pytorch.org/whl/cu118 ; \
    fi

# One pip call: resolving these separately lets huggingface-hub drift past the
# <1.0 ceiling transformers 4.41.2 requires.
RUN pip install \
        transformers==4.41.2 \
        tokenizers==0.19.1 \
        huggingface-hub==0.23.2 \
        datasets==2.20.0 \
        accelerate==0.33.0

# The base ships NVIDIA's transformer_engine, compiled against the PyTorch that
# came with it. Replacing PyTorch above leaves its compiled extension missing,
# and accelerate imports the package purely by name - which breaks
# `import accelerate`. Nothing in THOR uses it.
RUN pip uninstall -y transformer-engine transformer_engine 2>/dev/null || true ; \
    python3 - <<'PY'
import importlib.util, pathlib, shutil
spec = importlib.util.find_spec("transformer_engine")
if spec is not None and spec.submodule_search_locations:
    for loc in list(spec.submodule_search_locations):
        p = pathlib.Path(loc)
        print(f"[cleanup] removing leftover {p}")
        shutil.rmtree(p, ignore_errors=True)
else:
    print("[cleanup] transformer_engine not present")
PY

RUN pip install \
        biopython \
        pandas \
        PyYAML \
        matplotlib \
        scikit-learn \
        simple_pid \
        tqdm \
        typeguard==2.12.1

# cuDNN. The base already ships one matching its TensorFlow; this pip package is
# belt-and-braces and is published for x86_64 only, so do not fail without it.
RUN pip install nvidia-cudnn-cu11==8.9.5.30 \
 || echo "[info] nvidia-cudnn-cu11 unavailable for this platform; using the base image's cuDNN"

# A path that does not exist is harmless in LD_LIBRARY_PATH, so this covers both
# the case where the pip cuDNN was installed and where it was not.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# Fail the build here rather than at job submission.
RUN echo "--- pip check (informational) ---" \
 && { pip check || echo "(conflicts reported above; check whether they involve THOR's packages)" ; } \
 && echo "---------------------------------" \
 && python3 - <<'PY'
import platform
import tensorflow as tf
import torch, transformers, datasets, accelerate
import yaml, pandas, numpy, sklearn, Bio, simple_pid, typeguard, tqdm
import h5py
print("Platform          :", platform.machine())
print("TensorFlow        :", tf.__version__)
print("NumPy             :", numpy.__version__, "(TF 2.13 requires <=1.24.3)")
assert tuple(int(x) for x in numpy.__version__.split(".")[:2]) < (1, 25), \
    f"numpy {numpy.__version__} is too new for TensorFlow 2.13"
print("PyTorch           :", torch.__version__, "| CUDA:", torch.version.cuda or "NONE (CPU-only)")
print("Transformers      :", transformers.__version__)
print("Datasets          :", datasets.__version__)
print("Accelerate        :", accelerate.__version__)
# No GPU is present on a CI builder, so device enumeration is not checked here;
# env/singularity/verify_image.sh does that on a real GPU node.
if torch.version.cuda is None:
    raise SystemExit(
        "PyTorch has no CUDA support on this platform. CUDA wheels for the "
        "pinned version are x86_64-only; aarch64 needs torch>=2.4 from the "
        "cu124 index. Refusing to publish a CPU-only image."
    )
PY

CMD ["/bin/bash"]
