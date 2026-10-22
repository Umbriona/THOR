#!/bin/bash
# Verify that a built THOR image can actually run this repository.
#
#     bash env/singularity/verify_image.sh [path/to/thor.sif]
#
# Goes beyond "do the packages import": it loads THOR's own modules inside the
# container and builds a generator and a discriminator, which is where version
# mismatches between TensorFlow, tensorflow-addons and Keras actually surface.
# Needs no GPU, no weights and no data; add --nv if you also want the GPU check
# to be meaningful.

set -uo pipefail

if [ -z "${THOR_ROOT:-}" ]; then
    _here=$(cd "$(dirname "$0")" && pwd)
    THOR_ROOT=$(cd "$_here" 2>/dev/null && git rev-parse --show-toplevel 2>/dev/null) || THOR_ROOT=""
    [ -n "$THOR_ROOT" ] || THOR_ROOT=$(cd "$_here/../.." && pwd)
fi
THOR_IMAGE=${1:-${THOR_IMAGE:-$THOR_ROOT/thor.sif}}

if [ ! -f "$THOR_IMAGE" ]; then
    echo "No image at $THOR_IMAGE"
    echo "Build one with:"
    echo "    singularity build --fakeroot thor.sif $THOR_ROOT/env/singularity/thor.recipe"
    exit 1
fi

echo "image : $THOR_IMAGE"
echo "repo  : $THOR_ROOT"
echo

singularity exec --nv \
    --bind "$THOR_ROOT":/ThermalGAN \
    "$THOR_IMAGE" python3 - <<'PY'
import sys, traceback
sys.path.insert(0, "/ThermalGAN/src")

failures = []

def step(label, fn):
    try:
        result = fn()
        print(f"  ok    {label}" + (f"  [{result}]" if result else ""))
    except Exception as exc:
        failures.append((label, traceback.format_exc()))
        print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")

print("third-party packages")

def _tf():
    import tensorflow as tf
    import platform
    return f"{tf.__version__} on {platform.machine()}"
def _torch():
    import torch
    return f"{torch.__version__}, cuda build {torch.version.cuda}"
def _tr():
    import transformers
    return transformers.__version__
def _ds():
    import datasets
    return datasets.__version__
def _misc():
    import yaml, pandas, numpy, sklearn, Bio, simple_pid, typeguard, tqdm, h5py
    return "yaml pandas numpy sklearn Bio simple_pid typeguard tqdm h5py"

step("tensorflow",        _tf)
step("torch",             _torch)
step("transformers",      _tr)
step("datasets",          _ds)
step("support packages",  _misc)

print("\ngpu")
def _gpu_tf():
    import tensorflow as tf
    return f"{len(tf.config.list_physical_devices('GPU'))} device(s) visible to TF"
def _gpu_torch():
    import torch
    return f"torch.cuda.is_available() = {torch.cuda.is_available()}"
step("tensorflow GPUs", _gpu_tf)
step("torch CUDA",      _gpu_torch)

print("\nTHOR modules")
def _layers():
    from utils import layers_new, layers_residual
    return "layers_new, layers_residual"
def _models():
    from utils import models_generator, models_discriminator, models_classifyer
    return "generator, discriminator, classifier"
def _cyclegan():
    from utils import models_cyclegan_self_loss
    return "models_cyclegan_self_loss (used by train_thor.py)"
def _atte():
    from utils import models_gan_atte
    return "models_gan_atte (used by generate_thor.py)"
def _pre():
    from utils import preprocessing, loaders, losses, metrics, callbacks
    return "preprocessing, loaders, losses, metrics, callbacks"
step("layers",            _layers)
step("model modules",     _models)
step("CycleGAN module",   _cyclegan)
step("generator module",  _atte)
step("data/loss modules", _pre)

print("\nlayer construction")
def _tfa_optional():
    # tensorflow-addons is deliberately absent (no aarch64 wheels, EOL since
    # May 2024). THOR must import cleanly without it; utils/tfa_compat.py
    # supplies a placeholder that only raises if tfa is genuinely used.
    from utils.tfa_compat import tfa
    try:
        import tensorflow_addons  # noqa: F401
        return f"tfa installed ({tensorflow_addons.__version__}); not required"
    except ImportError:
        return "tfa absent, as expected; shim active"
def _specnorm():
    import tensorflow as tf
    from utils.layers_new import SpectralNormalization
    layer = SpectralNormalization(tf.keras.layers.Conv1D(4, 3, padding="same"))
    layer(tf.zeros((1, 16, 8)))
    return "SpectralNormalization (vendored)"
def _gumbel():
    import tensorflow as tf
    from utils.layers_new import GumbelSoftmax
    GumbelSoftmax()(tf.zeros((1, 16, 21)))
    return "GumbelSoftmax"
step("tfa optional-import",   _tfa_optional)
step("SpectralNormalization", _specnorm)
step("GumbelSoftmax",         _gumbel)

print("\nbuild the models from the reference config")
# Built exactly as utils/models_cyclegan_self_loss.py does (lines 63-66).
def _cfg():
    import yaml
    with open("/ThermalGAN/config/Cycle_gan/Reference_config.yaml") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)
def _gen():
    from utils import models_gan_atte as gan
    g = gan.Generator(config=_cfg()["CycleGan"]["Generator"])
    return f"{g.count_params():,} parameters"
def _disc():
    from utils import models_gan_atte as gan
    d = gan.Discriminator(config=_cfg()["CycleGan"]["Discriminator"])
    return f"{d.count_params():,} parameters"
step("Generator",     _gen)
step("Discriminator", _disc)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED\n")
    for label, tb in failures:
        print(f"----- {label} -----")
        print(tb)
    sys.exit(1)
print("All checks passed - the image can run this repository.")
PY

status=$?
echo
if [ $status -ne 0 ]; then
    echo "Verification FAILED (exit $status)."
else
    echo "Verification passed."
fi
exit $status
