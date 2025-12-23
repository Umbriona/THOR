#!/usr/bin/env python
"""
Generate ThermalGAN variants for a small FASTA by computing ESM per-position
likelihoods inline and running only the trained generator G.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import tensorflow as tf
# Allow TensorFlow to grow GPU memory so Torch can also use the device.
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as exc:  # pragma: no cover
    print(f"[warn] Could not set TF memory growth: {exc}")
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # points to src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
scripts_root = PROJECT_ROOT / "scripts"
if str(scripts_root) not in sys.path:
    sys.path.append(str(scripts_root))

# Local imports
from utils import preprocessing as pre
from utils import models_gan_atte as gan
from data_processing.compute_MLM_features_quantized_tfrecord import (  # noqa: E501
    AA20,
    compute_single_mode_fast,
    load_fasta_with_meta,
)


MAX_LEN = 512
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def parse_args():
    parser = argparse.ArgumentParser("Generate ThermalGAN variants from FASTA (ESM inline)")
    parser.add_argument("--run_dir", required=True, help="Run directory holding config.yaml and weights/")
    parser.add_argument("--fasta", required=True, help="Input FASTA with optional temp in header")
    parser.add_argument("--epoch", type=int, default=1999, help="Epoch to load from run_dir/weights/epoch_{n}")
    parser.add_argument("--output_dir", default=None, help="Where to write FASTA/JSONL outputs (default: run_dir)")
    parser.add_argument("--replicates", type=int, default=1, help="Number of variants per input sequence")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generator logits")
    parser.add_argument("--store_softmax", action="store_true", help="Also emit per-position softmax to JSONL")
    parser.add_argument("--hf_model", default="facebook/esm1v_t33_650M_UR90S_1", help="HuggingFace ESM model id/path")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Torch device for ESM scoring")
    parser.add_argument("--batch_tokens", type=int, default=40000, help="Token budget for ESM single-mask mode")
    parser.add_argument("--max_length", type=int, default=1022, help="Max length passed to tokenizer (ESM side)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES for TensorFlow/GAN")
    parser.add_argument("--esm_bf16", action="store_true", help="Use bfloat16 autocast for ESM (GPU only)")
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "r") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def quantize_and_pad_probs(probs: np.ndarray) -> np.ndarray:
    """Pad to 512x21 and quantize to match training inputs (uint8 scaled 0-255)."""
    L = min(probs.shape[0], MAX_LEN)
    padded = np.zeros((MAX_LEN, 21), dtype=np.float32)
    scaled = np.rint(np.clip(probs[:L], 0.0, 1.0) * 255.0).astype(np.float32)
    padded[:L, :20] = scaled
    return padded


def encode_sequences(ids: List[str], seqs: List[str], probs: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, List[str], List[float]]:
    seq_id_tensors = []
    prob_tensors = []
    temps = []
    for rid, seq in zip(ids, seqs):
        clean_seq = seq.replace(" ", "")
        ids_tensor = pre.seq_bytes_to_ids(tf.constant(clean_seq.encode("utf-8")), max_length=MAX_LEN, records_type="ESM")
        seq_id_tensors.append(ids_tensor.numpy().astype(np.int32))
        prob_tensors.append(quantize_and_pad_probs(probs[rid]["probs"]))
        temp_val = probs[rid].get("temp", None)
        temps.append(float(temp_val) if temp_val is not None else float("nan"))
    return np.stack(seq_id_tensors, axis=0), np.stack(prob_tensors, axis=0), ids, temps


def build_generator(config: Dict) -> Tuple[tf.keras.Model, bool]:
    concat_modalities = config["CycleGan"].get(
        "concat_modalities_generator", config["CycleGan"].get("concat_modalities", False)
    )
    gen_cfg = dict(config["CycleGan"]["Generator"])
    gen_cfg["input_dim"] = 42 if concat_modalities else 21
    generator = gan.Generator(config=gen_cfg)
    return generator, concat_modalities


def prepare_generator_inputs(seq_ids: np.ndarray, prob_q: np.ndarray, concat_modalities: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    mask = tf.cast(seq_ids >= 0, tf.float32)[..., tf.newaxis]
    real_x = tf.one_hot(tf.where(seq_ids < 0, 0, seq_ids), depth=21, dtype=tf.float32)
    prob = tf.convert_to_tensor(prob_q, dtype=tf.float32) / 255.0
    if concat_modalities:
        gen_inp = tf.concat([real_x, prob], axis=-1)
    else:
        gen_inp = tf.math.add(real_x, prob)
    return gen_inp, mask


def sample_variants(generator: tf.keras.Model, gen_inp: tf.Tensor, mask: tf.Tensor, replicates: int, temperature: float):
    # One forward pass is enough; draw multiple categorical samples from the same logits.
    samples = []
    _, softmax_trimmed, logits = generator([gen_inp, mask], training=False)
    logits_adj = logits[..., :20] / max(temperature, 1e-4)
    log_probs = tf.nn.log_softmax(logits_adj, axis=-1)
    flat = tf.reshape(log_probs, [-1, log_probs.shape[-1]])
    #softmax_trimmed = tf.nn.softmax(logits_adj, axis=-1)

    for _ in range(replicates):
        drawn = tf.random.categorical(flat, num_samples=1)
        sampled = tf.reshape(drawn, tf.shape(logits_adj)[:-1])
        sampled = tf.where(tf.squeeze(mask, axis=-1) > 0, sampled, tf.constant(-1, dtype=sampled.dtype))
        samples.append((sampled.numpy(), softmax_trimmed.numpy()))
    return samples


def sequences_from_ids(ids_np: np.ndarray, mask_np: np.ndarray) -> List[str]:
    seqs = []
    for seq_ids, w in zip(list(ids_np), list(mask_np)):
        seqs.append(pre.convert_table(seq_ids, np.reshape(w, (MAX_LEN,))))
    return seqs


def write_outputs(out_dir: Path, epoch: int, fasta_ids: List[str], temps: List[float], wt_seqs: List[str],
                  mask_np: np.ndarray, variant_samples, store_softmax: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / f"variants_epoch_{epoch}.fasta"
    with open(fasta_path, "w") as fh:
        for rep_idx, (variants, _) in enumerate(variant_samples):
            for rid, temp, wt, var in zip(fasta_ids, temps, wt_seqs, sequences_from_ids(variants, mask_np)):
                fh.write(f">{rid}_wt_rep{rep_idx} {temp}\n{wt}\n")
                fh.write(f">{rid}_variant_rep{rep_idx}\n{var}\n")

    if store_softmax:
        probs_path = out_dir / f"variants_epoch_{epoch}_softmax.jsonl"
        with open(probs_path, "w") as pf:
            #for rep_idx, (_, probs) in enumerate(variant_samples):
            probs = variant_samples[0][1]
            rep_idx = 0
            for rid, prob, mask in zip(fasta_ids, probs, mask_np):
                length = int(np.sum(mask))
                trimmed = prob[:length].tolist()
                pf.write(json.dumps({"id": f"{rid}_rep{rep_idx}", "probs": trimmed}) + "\n")

    print(f"Wrote variants to {fasta_path}")
    if store_softmax:
        print(f"Wrote softmax to {probs_path}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    # Allow TensorFlow to grow GPU memory so Torch can also use the device.
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Could not set TF memory growth: {exc}")

    run_dir = Path(args.run_dir).resolve()
    print(f"Running script fom {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    print(f"Output dir set to: {output_dir}")
    config = load_config(run_dir)
    print(f"Loading config from : {run_dir}")
    # ESM scoring
    # Ensure compatibility with load_fasta_with_meta (expects args.model).
    args.model = args.hf_model
    print("[info] Loading ESM tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(args.hf_model)
    if hasattr(model, "esm") and hasattr(model.esm, "contact_head"):
        model.esm.contact_head = None
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = model.to(device)
    print(f"[info] ESM model on device: {device}")

    fasta_ids, seqs, temps = load_fasta_with_meta(args.fasta, default_temp=None, args=args)
    print("[info] Computing per-position ESM probabilities...")
    results = compute_single_mode_fast(
        model=model,
        tokenizer=tokenizer,
        ids=fasta_ids,
        seqs=seqs,
        temps=temps,
        max_length=args.max_length,
        device=model.device,
        aa_token_ids=tokenizer.convert_tokens_to_ids(list(AA20)),
        batch_tokens=args.batch_tokens,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        use_amp_bf16=args.esm_bf16,
    )
    print("[info] ESM probabilities computed.")

    seq_ids_np, prob_q_np, fasta_ids, temps = encode_sequences(fasta_ids, seqs, results)

    # Build generator and load weights
    generator, concat_modalities = build_generator(config)
    weight_path = run_dir / "weights" / f"epoch_{args.epoch}" / "generator_G.h5"
    print(f"[info] Loading generator weights from: {weight_path}")
    generator.load_weights(weight_path)

    gen_inp, mask = prepare_generator_inputs(seq_ids_np, prob_q_np, concat_modalities)
    print("[info] Sampling variants...")

    variants = sample_variants(generator, gen_inp, mask, args.replicates, args.temperature)

    mask_np = np.squeeze(mask.numpy(), axis=-1)
    wt_seqs = sequences_from_ids(seq_ids_np, mask_np)

    write_outputs(output_dir, args.epoch, fasta_ids, temps, wt_seqs, mask_np, variants, args.store_softmax)


if __name__ == "__main__":
    main()
