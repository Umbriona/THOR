#!/usr/bin/env python
"""
Convert legacy OGT SavedModel checkpoints (variables/variables) into .h5 files.
"""
import argparse
import os
import sys
from pathlib import Path

import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THERMALGAN_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils import models_classifyer as models_class  # noqa: E402


def load_classifier_config(config_path: Path) -> dict:
    with config_path.open("r") as fh:
        cfg = yaml.load(fh, Loader=yaml.FullLoader)
    return cfg["Classifier"] if "Classifier" in cfg else cfg


def validate_checkpoint_prefix(prefix: Path) -> None:
    index_path = prefix.with_suffix(".index")
    data_files = list(prefix.parent.glob(prefix.name + ".data-*"))
    if not index_path.exists() or not data_files:
        raise FileNotFoundError(f"Missing checkpoint files for prefix: {prefix}")


def build_model(classifier_cfg: dict) -> tf.keras.Model:
    vocab = classifier_cfg.get("vocab_size", 21)
    return models_class.get_classifier(classifier_cfg, vocab)


def convert_one(model_name: str, weights_prefix: Path, out_dir: Path,
                classifier_cfg: dict, weights_only: bool) -> Path:
    model = build_model(classifier_cfg)
    model.load_weights(str(weights_prefix)).expect_partial()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("model_weights.h5" if weights_only else "model.h5")
    if weights_only:
        model.save_weights(out_path)
    else:
        model.save(out_path, include_optimizer=False)
    tf.keras.backend.clear_session()
    return out_path


def parse_args() -> argparse.Namespace:
    default_config = THERMALGAN_ROOT / "config" / "Classifier" / "config_classifier1.yaml"
    default_ogt = THERMALGAN_ROOT / "weights" / "OGT"
    parser = argparse.ArgumentParser(
        description="Convert OGT SavedModel checkpoints to .h5 files"
    )
    parser.add_argument("--config", default=str(default_config),
                        help="Path to classifier config YAML")
    parser.add_argument("--ogt_dir", default=str(default_ogt),
                        help="Directory containing Model1/Model2/Model3")
    parser.add_argument("--output_dir", default=None,
                        help="Optional output root (defaults to each ModelX dir)")
    parser.add_argument("--models", nargs="*", default=["Model1", "Model2", "Model3"],
                        help="Model subdirectories to convert")
    parser.add_argument("--weights_only", action="store_true",
                        help="Save weights-only .h5 instead of full model")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    classifier_cfg = load_classifier_config(config_path)

    ogt_dir = Path(args.ogt_dir)
    if not ogt_dir.exists():
        raise FileNotFoundError(f"OGT directory not found: {ogt_dir}")

    output_base = Path(args.output_dir) if args.output_dir else None
    for model_name in args.models:
        model_dir = ogt_dir / model_name
        weights_prefix = model_dir / "variables" / "variables"
        validate_checkpoint_prefix(weights_prefix)
        out_dir = (output_base / model_name) if output_base else model_dir
        out_path = convert_one(model_name, weights_prefix, out_dir,
                               classifier_cfg, args.weights_only)
        print(f"Saved {model_name} -> {out_path}")


if __name__ == "__main__":
    main()
