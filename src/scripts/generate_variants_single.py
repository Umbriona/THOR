#!/usr/bin/env python
"""
Generate ThermalGAN variants for a small FASTA or single sequence by computing
ESM per-position likelihoods inline and running the trained generator G. Emits
generator/ESM NLL metrics, predicted OGT, and supports optional ESM-based
filtering + argmax optimization.
"""
import argparse
import json
import os
import sys
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
from utils import models_classifyer as models_class
from data_processing.compute_MLM_features_quantized_tfrecord import (  # noqa: E501
    AA20,
    clean_seq,
    compute_single_mode_fast,
    compute_random_partition_mode,
    load_fasta_with_meta,
    to_protbert,
)


MAX_LEN = 512
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
LOG_EPS = 1e-9
OGT_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
OGT_AA_TO_IDX = {aa: i for i, aa in enumerate(OGT_ALPHABET)}


def parse_args():
    parser = argparse.ArgumentParser("Generate ThermalGAN variants from FASTA or single sequence (ESM inline)")
    parser.add_argument("--run_dir", required=True, help="Run directory holding config.yaml and weights/")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fasta", help="Input FASTA with optional temp in header")
    input_group.add_argument("--sequence", help="Input single protein sequence (string)")
    parser.add_argument("--seq_id", default="query", help="Identifier for --sequence input")
    parser.add_argument("--seq_temp", type=float, default=None, help="Optional temperature metadata for --sequence input")
    parser.add_argument("--epoch", type=int, default=1999, help="Epoch to load from run_dir/weights/epoch_{n}")
    parser.add_argument("--output_dir", default=None, help="Where to write FASTA/JSONL outputs (default: run_dir)")
    parser.add_argument("--name", default=None, help="Optional prefix for output files")
    parser.add_argument("--replicates", type=int, default=1, help="Number of variants per input sequence")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generator logits")
    parser.add_argument("--store_softmax", action="store_true", help="Also emit per-position softmax to JSONL")
    parser.add_argument("--hf_model", default="facebook/esm1v_t33_650M_UR90S_1", help="HuggingFace ESM model id/path")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Torch device for ESM scoring")
    parser.add_argument("--batch_tokens", type=int, default=40000, help="Token budget for ESM single-mask mode")
    parser.add_argument("--max_length", type=int, default=1022, help="Max length passed to tokenizer (ESM side)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES for TensorFlow/GAN")
    parser.add_argument("--esm_bf16", action="store_true", help="Use bfloat16 autocast for ESM (GPU only)")
    parser.add_argument("--esm_init_mode", choices=["single", "random"], default="single",
                        help="ESM mode for initial generator features (single-mask or random-partition)")
    parser.add_argument("--esm_random_mask_prob", type=float, default=0.15,
                        help="Mask probability for random-partition ESM init")
    parser.add_argument("--esm_random_chunk", type=int, default=8,
                        help="Number of masked chunks per batch for random-partition ESM init")
    parser.add_argument("--esm_random_seed", type=int, default=42, help="Seed for random-partition ESM init")
    parser.add_argument("--esm_filter_threshold", type=float, default=None,
                        help="ESM per-position NLL threshold; substitutions above this are reverted to WT")
    parser.add_argument("--skip_optimize", action="store_true",
                        help="Skip generator argmax optimization after filtering")
    parser.add_argument("--ogt_config", default="/ThermalGAN/config/Classifier/config_classifier1.yaml",
                        help="Path to OGT ensemble config")
    parser.add_argument("--ogt_weights", nargs=3,
                        default=[
                            "/ThermalGAN/weights/OGT/Model1/variables/variables",
                            "/ThermalGAN/weights/OGT/Model2/variables/variables",
                            "/ThermalGAN/weights/OGT/Model3/variables/variables",
                        ],
                        help="Three weight files for the OGT ensemble")
    parser.add_argument("--ogt_batch_size", type=int, default=128, help="Batch size for OGT prediction")
    parser.add_argument("--ogt_max_len", type=int, default=512, help="Max length for OGT predictor padding")
    parser.add_argument("--skip_ogt", action="store_true", help="Disable OGT prediction step")
    parser.add_argument("--filter_opt_cycles", type=int, default=1,
                        help="Number of filter+optimize cycles to run (>=0). Only the last cycle's optimized outputs are kept.")
    parser.add_argument("--sampling_cycles", type=int, default=0,
                        help="Number of additional filter+sample cycles before optimization; only the first sampling cycle draws multiple replicates.")
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "r") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def load_input_sequences(args) -> Tuple[List[str], List[str], List[Optional[float]]]:
    if args.sequence:
        seq = clean_seq(args.sequence)
        if args.hf_model.split("/")[-1] in ["prot_bert", "prot_bert_bfd"]:
            seq = to_protbert(seq)
        if not seq:
            raise ValueError("Provided --sequence is empty after cleaning.")
        return [args.seq_id], [seq], [args.seq_temp]
    return load_fasta_with_meta(args.fasta, default_temp=None, args=args)


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


def run_generator(generator: tf.keras.Model, gen_inp: tf.Tensor, mask: tf.Tensor, temperature: float):
    """Forward pass through the generator; return log-probs over AAs (trimmed to 20) and softmax."""
    _, softmax_full, logits = generator([gen_inp, mask], training=False)
    logits_adj = logits[..., :20] / max(temperature, 1e-4)
    log_probs = tf.nn.log_softmax(logits_adj, axis=-1)
    return log_probs, softmax_full


def sample_from_log_probs(log_probs: tf.Tensor, mask: tf.Tensor, replicates: int) -> List[np.ndarray]:
    """Draw categorical samples from log-probs; returns list of seq id arrays with pads set to -1."""
    samples = []
    flat = tf.reshape(log_probs, [-1, log_probs.shape[-1]])
    draws = tf.random.categorical(flat, num_samples=replicates)
    draws = tf.reshape(draws, [log_probs.shape[0], log_probs.shape[1], replicates])
    valid_mask = tf.cast(tf.squeeze(mask, axis=-1) > 0, draws.dtype)
    for rep in range(replicates):
        sampled = draws[..., rep]
        sampled = tf.where(valid_mask > 0, sampled, tf.constant(-1, dtype=sampled.dtype))
        samples.append(sampled.numpy().astype(np.int32))
    return samples


def argmax_from_log_probs(log_probs: tf.Tensor, mask: tf.Tensor) -> np.ndarray:
    """Deterministic argmax decode with padding preserved."""
    argmax_ids = tf.argmax(log_probs, axis=-1, output_type=tf.int32)
    valid_mask = tf.cast(tf.squeeze(mask, axis=-1) > 0, tf.int32)
    argmax_ids = tf.where(valid_mask > 0, argmax_ids, tf.constant(-1, dtype=tf.int32))
    return argmax_ids.numpy()


def sequences_from_ids(ids_np: np.ndarray, mask_np: np.ndarray) -> List[str]:
    seqs = []
    for seq_ids, w in zip(list(ids_np), list(mask_np)):
        seqs.append(pre.convert_table(seq_ids, np.reshape(w, (MAX_LEN,))))
    return seqs


def compute_generator_nll(seq_ids: np.ndarray, log_probs: np.ndarray, mask_np: np.ndarray,
                          wt_ids: Optional[np.ndarray] = None) -> Tuple[float, Optional[float]]:
    """Return (global_nll, sub_nll) for one sequence; sub_nll is None/NA if no substitutions."""
    valid_pos = (mask_np > 0) & (seq_ids >= 0)
    if not np.any(valid_pos):
        return float("nan"), None
    rows = np.nonzero(valid_pos)[0]
    cols = seq_ids[rows]
    logp = log_probs[rows, cols]
    global_nll = float(-np.mean(logp))

    sub_nll = None
    if wt_ids is not None:
        sub_mask = valid_pos & (seq_ids != wt_ids)
        if np.any(sub_mask):
            sub_rows = np.nonzero(sub_mask)[0]
            sub_cols = seq_ids[sub_rows]
            sub_logp = log_probs[sub_rows, sub_cols]
            sub_nll = float(-np.mean(sub_logp))
    return global_nll, sub_nll


def compute_esm_nll(seq: str, probs: np.ndarray, wt_seq: Optional[str] = None) -> Tuple[float, Optional[float]]:
    """Return (global_nll, sub_nll) from ESM probs for the provided sequence."""
    L = min(len(seq), probs.shape[0])
    logp_all = []
    logp_sub = []
    for i in range(L):
        aa = seq[i]
        idx = AA_TO_IDX.get(aa)
        if idx is None:
            continue
        p = float(probs[i, idx])
        logp = np.log(max(p, LOG_EPS))
        logp_all.append(logp)
        if wt_seq is not None and i < len(wt_seq) and wt_seq[i] != aa:
            logp_sub.append(logp)
    global_nll = float(-np.mean(logp_all)) if logp_all else float("nan")
    sub_nll = float(-np.mean(logp_sub)) if logp_sub else None
    return global_nll, sub_nll


def filter_bad_mutations(seq: str, wt_seq: str, probs: np.ndarray, threshold: float) -> Tuple[str, List[str]]:
    """Revert substitutions whose ESM NLL exceeds the threshold. Returns new seq and list like ['A20F']."""
    L = min(len(seq), probs.shape[0], len(wt_seq))
    seq_list = list(seq)
    reverted: List[str] = []
    for i in range(L):
        if seq_list[i] == wt_seq[i]:
            continue
        idx = AA_TO_IDX.get(seq_list[i])
        if idx is None:
            continue
        nll = -np.log(max(float(probs[i, idx]), LOG_EPS))
        if nll > threshold:
            wt_aa = wt_seq[i]
            mut_aa = seq_list[i]
            seq_list[i] = wt_seq[i]
            reverted.append(f"{wt_aa}{i+1}{mut_aa}")
    return "".join(seq_list), reverted


def format_metric(val: Optional[float]) -> str:
    if val is None:
        return "NA"
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return "NA"
    return f"{val:.4f}"


def compute_identity(seq: str, wt_seq: str) -> Optional[float]:
    if wt_seq is None:
        return None
    L = min(len(seq), len(wt_seq))
    if L == 0:
        return None
    matches = sum(1 for i in range(L) if seq[i] == wt_seq[i])
    return 100.0 * matches / len(wt_seq)


def write_outputs(out_dir: Path, epoch: int, records: List[Dict], store_softmax: bool,
                  base_softmax: Optional[np.ndarray], mask_np: Optional[np.ndarray], fasta_ids: List[str],
                  name_prefix: Optional[str] = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{name_prefix}_" if name_prefix else ""
    fasta_path = out_dir / f"{prefix}variants_epoch_{epoch}.fasta"
    kind_order = {"wt": 0, "sampled": 1, "filtered": 2, "optimized": 3}
    def rep_sort_value(rep):
        if isinstance(rep, int):
            return (0, rep)
        try:
            return (0, int(rep))
        except Exception:
            return (1, str(rep))
    records_sorted = sorted(
        records,
        key=lambda r: (
            r.get("base_id", ""),
            kind_order.get(r.get("kind", ""), 99),
            rep_sort_value(r.get("rep", "")),
            r.get("label", ""),
        ),
    )

    with open(fasta_path, "w") as fh:
        for rec in records_sorted:
            header_parts = [
                rec["label"],
                f"type={rec.get('kind', 'NA')}",
                f"predicted_OGT={format_metric(rec.get('ogt'))}",
                f"G_global_nll={format_metric(rec.get('g_global_nll'))}",
                f"G_sub_nll={format_metric(rec.get('g_sub_nll'))}",
                f"ESM_global_nll={format_metric(rec.get('esm_global_nll'))}",
                f"ESM_sub_nll={format_metric(rec.get('esm_sub_nll'))}",
                f"Identity_to_WT(%)={format_metric(rec.get('identity_to_wt'))}",
            ]
            if rec.get("temp") is not None:
                header_parts.append(f"temp={format_metric(rec['temp'])}")
            if rec.get("filter_threshold") is not None:
                header_parts.append(f"filter_threshold={format_metric(rec.get('filter_threshold'))}")
            if rec.get("reverted_positions"):
                header_parts.append(f"reverted_pos={','.join(str(p) for p in rec['reverted_positions'])}")
            fh.write(">" + " | ".join(header_parts) + "\n")
            fh.write(rec["seq"] + "\n")

    probs_path = None
    if store_softmax and base_softmax is not None and mask_np is not None:
        probs_path = out_dir / f"{prefix}variants_epoch_{epoch}_softmax.jsonl"
        with open(probs_path, "w") as pf:
            for rid, prob, mask_row in zip(fasta_ids, base_softmax, mask_np):
                length = int(np.sum(mask_row))
                trimmed = prob[:length].tolist()
                pf.write(json.dumps({"id": f"{rid}_rep0", "probs": trimmed}) + "\n")

    print(f"Wrote sequences to {fasta_path}")
    if probs_path:
        print(f"Wrote generator softmax to {probs_path}")


def run_esm(tokenizer, model, ids, seqs, temps, args):
    return compute_single_mode_fast(
        model=model,
        tokenizer=tokenizer,
        ids=ids,
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


def run_esm_random(tokenizer, model, ids, seqs, temps, args):
    return compute_random_partition_mode(
        model=model,
        tokenizer=tokenizer,
        ids=ids,
        seqs=seqs,
        temps=temps,
        max_length=args.max_length,
        device=model.device,
        aa_token_ids=tokenizer.convert_tokens_to_ids(list(AA20)),
        mask_prob=args.esm_random_mask_prob,
        chunk_batch=args.esm_random_chunk,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        seed=args.esm_random_seed,
    )


def one_hot_encode_for_ogt(seqs: List[str], max_len: int = 512) -> np.ndarray:
    x = np.zeros((len(seqs), max_len, len(OGT_ALPHABET)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = seq.upper()
        L = min(len(seq), max_len)
        for j in range(L):
            idx = OGT_AA_TO_IDX.get(seq[j], OGT_AA_TO_IDX["X"])
            x[i, j, idx] = 1.0
    return x


def build_ogt_ensemble(config_path: str, weights_paths: List[str]) -> tf.keras.Model:
    with open(config_path, "r") as fd:
        config_class = yaml.load(fd, Loader=yaml.FullLoader)

    model_input = tf.keras.layers.Input(shape=(512, len(OGT_ALPHABET)))
    model1 = models_class.get_classifier(config_class["Classifier"], len(OGT_ALPHABET))
    model2 = models_class.get_classifier(config_class["Classifier"], len(OGT_ALPHABET))
    model3 = models_class.get_classifier(config_class["Classifier"], len(OGT_ALPHABET))

    output1 = model1(model_input)
    output2 = model2(model_input)
    output3 = model3(model_input)

    print("Built OGT regression")

    model1.load_weights(weights_paths[0], skip_mismatch=False)
    model2.load_weights(weights_paths[1], skip_mismatch=False)
    model3.load_weights(weights_paths[2], skip_mismatch=False)

    print("Loaded OGT regression weights")

    ensemble_output = tf.keras.layers.Average()([output1, output2, output3])
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)
    return ensemble_model


def predict_ogt_for_records(records: List[Dict], config_path: str, weights_paths: List[str],
                            batch_size: int, max_len: int):
    if not records:
        return
    #for p in weights_paths:
    #    if not os.path.exists(p):
    #        raise FileNotFoundError(f"OGT weights not found: {p}")
    #if not os.path.exists(config_path):
    #    raise FileNotFoundError(f"OGT config not found: {config_path}")
    model = build_ogt_ensemble(config_path, weights_paths)
    seqs = [rec["seq"] for rec in records]
    x = one_hot_encode_for_ogt(seqs, max_len=max_len)
    preds = model.predict(x, batch_size=batch_size, verbose=0).reshape((-1,))
    for rec, val in zip(records, preds):
        rec["ogt"] = float(val)


def run_esm_for_records(records: List[Dict], tokenizer, model, args):
    ids = [rec["label"] for rec in records]
    seqs = [rec["seq"] for rec in records]
    temps = [rec.get("temp") for rec in records]
    return run_esm(tokenizer, model, ids, seqs, temps, args)


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

    fasta_ids, seqs, temps = load_input_sequences(args)
    wt_seq_map = {rid: seq for rid, seq in zip(fasta_ids, seqs)}
    # ESM for generator features (single-mask or random-partition) and for metrics
    if args.esm_init_mode == "random":
        print(f"[info] Computing ESM random-partition features for generator (mask_prob={args.esm_random_mask_prob})...")
        feature_results = run_esm_random(tokenizer, model, fasta_ids, seqs, temps, args)
        print("[info] Computing per-position single-mask ESM probabilities for metrics...")
        wt_results = run_esm(tokenizer, model, fasta_ids, seqs, temps, args)
    else:
        print("[info] Computing per-position ESM probabilities for wildtypes (single-mask)...")
        wt_results = run_esm(tokenizer, model, fasta_ids, seqs, temps, args)
        feature_results = wt_results
    print("[info] ESM probabilities computed for wildtypes.")

    seq_ids_np, prob_q_np, fasta_ids, temps = encode_sequences(fasta_ids, seqs, feature_results)

    # Build generator and load weights
    generator, concat_modalities = build_generator(config)
    weight_path = run_dir / "weights" / f"epoch_{args.epoch}" / "generator_G.h5"
    print(f"[info] Loading generator weights from: {weight_path}")
    generator.load_weights(weight_path)

    gen_inp, mask = prepare_generator_inputs(seq_ids_np, prob_q_np, concat_modalities)
    print("[info] Running generator forward pass...")
    log_probs_tf, softmax_tf = run_generator(generator, gen_inp, mask, args.temperature)

    mask_np = np.squeeze(mask.numpy(), axis=-1)
    log_probs_np = log_probs_tf.numpy()
    base_softmax = softmax_tf.numpy()
    wt_seqs = sequences_from_ids(seq_ids_np, mask_np)

    records: List[Dict] = []

    # Wildtype records (per replicate to mirror existing behavior)
    for rep_idx in range(args.replicates):
        for i, rid in enumerate(fasta_ids):
            g_global, g_sub = compute_generator_nll(seq_ids_np[i], log_probs_np[i], mask_np[i])
            esm_global, esm_sub = compute_esm_nll(wt_seqs[i], wt_results[rid]["probs"])
            records.append({
                "label": f"{rid}_wt_rep{rep_idx}",
                "base_id": rid,
                "rep": rep_idx,
                "kind": "wt",
                "seq": wt_seqs[i],
                "temp": temps[i],
                "g_global_nll": g_global,
                "g_sub_nll": g_sub,
                "esm_global_nll": esm_global,
                "esm_sub_nll": esm_sub,
                "identity_to_wt": None,
            })

    # Sampled variants
    print("[info] Sampling variants...")
    variant_samples = sample_from_log_probs(log_probs_tf, mask, args.replicates)
    # Also keep a maximum-likelihood variant (argmax) per sequence for the initial generator pass
    variant_argmax_ids = argmax_from_log_probs(log_probs_tf, mask)
    variant_argmax_seqs = sequences_from_ids(variant_argmax_ids, mask_np)
    variant_records: List[Dict] = []
    for rep_idx, sample_ids in enumerate(variant_samples):
        sample_seqs = sequences_from_ids(sample_ids, mask_np)
        for i, rid in enumerate(fasta_ids):
            g_global, g_sub = compute_generator_nll(sample_ids[i], log_probs_np[i], mask_np[i], wt_ids=seq_ids_np[i])
            variant_records.append({
                "label": f"{rid}_variant_rep{rep_idx}",
                "base_id": rid,
                "rep": rep_idx,
                "kind": "sampled",
                "seq": sample_seqs[i],
                "temp": temps[i],
                "g_global_nll": g_global,
                "g_sub_nll": g_sub,
                "identity_to_wt": compute_identity(sample_seqs[i], wt_seq_map[rid]),
            })
    # Max-likelihood variants (rep_max)
    for i, rid in enumerate(fasta_ids):
        g_global, g_sub = compute_generator_nll(variant_argmax_ids[i], log_probs_np[i], mask_np[i], wt_ids=seq_ids_np[i])
        variant_records.append({
            "label": f"{rid}_variant_rep_max",
            "base_id": rid,
            "rep": "max",
            "kind": "sampled",
            "seq": variant_argmax_seqs[i],
            "temp": temps[i],
            "g_global_nll": g_global,
            "g_sub_nll": g_sub,
            "identity_to_wt": compute_identity(variant_argmax_seqs[i], wt_seq_map[rid]),
        })

    # ESM metrics for sampled variants
    print("[info] Computing ESM probabilities for sampled variants...")
    variant_results = run_esm_for_records(variant_records, tokenizer, model, args)
    for rec in variant_records:
        res = variant_results[rec["label"]]
        esm_global, esm_sub = compute_esm_nll(rec["seq"], res["probs"], wt_seq_map[rec["base_id"]])
        rec["esm_global_nll"] = esm_global
        rec["esm_sub_nll"] = esm_sub
        rec["identity_to_wt"] = compute_identity(rec["seq"], wt_seq_map[rec["base_id"]])

    records.extend(variant_records)

    current_records = variant_records
    current_results = variant_results
    sampling_final: List[Dict] = []
    wt_idx_map = {rid: idx for idx, rid in enumerate(fasta_ids)} if args.esm_filter_threshold is not None else {}

    # Optional filtering + sampling cycles before optimization
    if args.esm_filter_threshold is not None and args.sampling_cycles > 0:
        sampling_cycles = max(1, args.sampling_cycles)
        for cycle_idx in range(sampling_cycles):
            print(f"[info] Sampling+filter cycle {cycle_idx+1}/{sampling_cycles}...")
            filtered_records: List[Dict] = []
            filtered_ids = []
            filtered_seqs = []
            filtered_temps = []

            for rec in current_records:
                probs = current_results[rec["label"]]["probs"]
                wt_seq = wt_seq_map[rec["base_id"]]
                filt_seq, reverted = filter_bad_mutations(rec["seq"], wt_seq, probs, args.esm_filter_threshold)
                filt_label = f"{rec['label']}_filtered"
                filtered_records.append({
                    "label": filt_label,
                    "base_id": rec["base_id"],
                    "rep": rec["rep"],
                    "kind": "filtered",
                    "seq": filt_seq,
                    "temp": rec["temp"],
                    "filter_threshold": args.esm_filter_threshold,
                    "reverted_positions": reverted,
                    "identity_to_wt": compute_identity(filt_seq, wt_seq),
                })
                filtered_ids.append(filt_label)
                filtered_seqs.append(filt_seq)
                filtered_temps.append(rec["temp"])

            if not filtered_records:
                print("[warn] No records after sampling filter; stopping sampling cycles.")
                break

            filtered_results = run_esm_for_records(filtered_records, tokenizer, model, args)
            for rec in filtered_records:
                res = filtered_results[rec["label"]]
                esm_global, esm_sub = compute_esm_nll(rec["seq"], res["probs"], wt_seq_map[rec["base_id"]])
                rec["esm_global_nll"] = esm_global
                rec["esm_sub_nll"] = esm_sub

            filt_seq_ids_np, filt_prob_q_np, _, _ = encode_sequences(filtered_ids, filtered_seqs, filtered_results)
            f_gen_inp, f_mask = prepare_generator_inputs(filt_seq_ids_np, filt_prob_q_np, concat_modalities)
            f_log_probs_tf, _ = run_generator(generator, f_gen_inp, f_mask, args.temperature)
            f_log_probs_np = f_log_probs_tf.numpy()
            f_mask_np = np.squeeze(f_mask.numpy(), axis=-1)

            for idx, rec in enumerate(filtered_records):
                wt_ids = seq_ids_np[wt_idx_map[rec["base_id"]]]
                g_global, g_sub = compute_generator_nll(filt_seq_ids_np[idx], f_log_probs_np[idx], f_mask_np[idx], wt_ids=wt_ids)
                rec["g_global_nll"] = g_global
                rec["g_sub_nll"] = g_sub

            sample_reps = args.replicates if cycle_idx == 0 else 1
            sampled_records: List[Dict] = []
            sampled_results = {}
            sampled_ids_all = sample_from_log_probs(f_log_probs_tf, f_mask, sample_reps)
            for rep_idx, sample_ids in enumerate(sampled_ids_all):
                sample_seqs = sequences_from_ids(sample_ids, f_mask_np)
                for i, filt_rec in enumerate(filtered_records):
                    wt_ids = seq_ids_np[wt_idx_map[filt_rec["base_id"]]]
                    g_global, g_sub = compute_generator_nll(sample_ids[i], f_log_probs_np[i], f_mask_np[i], wt_ids=wt_ids)
                    sample_label = f"{filt_rec['label']}_samp{cycle_idx+1}_rep{rep_idx}"
                    sample_rep = f"{filt_rec['rep']}_s{cycle_idx+1}_rep{rep_idx}"
                    sampled_records.append({
                        "label": sample_label,
                        "base_id": filt_rec["base_id"],
                        "rep": sample_rep,
                        "kind": "sampled",
                        "seq": sample_seqs[i],
                        "temp": filt_rec["temp"],
                        "filter_threshold": args.esm_filter_threshold,
                        "g_global_nll": g_global,
                        "g_sub_nll": g_sub,
                        "identity_to_wt": compute_identity(sample_seqs[i], wt_seq_map[filt_rec["base_id"]]),
                    })

            if sampled_records:
                sampled_results = run_esm_for_records(sampled_records, tokenizer, model, args)
                for rec in sampled_records:
                    res = sampled_results[rec["label"]]
                    esm_global, esm_sub = compute_esm_nll(rec["seq"], res["probs"], wt_seq_map[rec["base_id"]])
                    rec["esm_global_nll"] = esm_global
                    rec["esm_sub_nll"] = esm_sub

            records.extend(filtered_records)
            records.extend(sampled_records)
            current_records = sampled_records
            current_results = sampled_results
            sampling_final = sampled_records

            if not current_records:
                print("[warn] No sampled records after cycle; stopping sampling cycles.")
                break

    # Optional filtering + optimization cycles
    filtered_final: List[Dict] = []
    opt_final: List[Dict] = []
    if args.esm_filter_threshold is not None and current_records:
        cycles = max(0, args.filter_opt_cycles)
        for cycle_idx in range(cycles):
            print(f"[info] Filter/optimize cycle {cycle_idx+1}/{cycles}...")
            filtered_records: List[Dict] = []
            filtered_ids = []
            filtered_seqs = []
            filtered_temps = []

            for rec in current_records:
                probs = current_results[rec["label"]]["probs"]
                wt_seq = wt_seq_map[rec["base_id"]]
                filt_seq, reverted = filter_bad_mutations(rec["seq"], wt_seq, probs, args.esm_filter_threshold)
                filt_label = f"{rec['label']}_filtered"
                filtered_records.append({
                    "label": filt_label,
                    "base_id": rec["base_id"],
                    "rep": rec["rep"],
                    "kind": "filtered",
                    "seq": filt_seq,
                    "temp": rec["temp"],
                    "filter_threshold": args.esm_filter_threshold,
                    "reverted_positions": reverted,
                    "identity_to_wt": compute_identity(filt_seq, wt_seq),
                })
                filtered_ids.append(filt_label)
                filtered_seqs.append(filt_seq)
                filtered_temps.append(rec["temp"])

            if not filtered_records:
                print("[warn] No records after filtering; stopping cycles.")
                break

            filtered_results = run_esm_for_records(filtered_records, tokenizer, model, args)
            for rec in filtered_records:
                res = filtered_results[rec["label"]]
                esm_global, esm_sub = compute_esm_nll(rec["seq"], res["probs"], wt_seq_map[rec["base_id"]])
                rec["esm_global_nll"] = esm_global
                rec["esm_sub_nll"] = esm_sub

            filt_seq_ids_np, filt_prob_q_np, _, _ = encode_sequences(filtered_ids, filtered_seqs, filtered_results)
            f_gen_inp, f_mask = prepare_generator_inputs(filt_seq_ids_np, filt_prob_q_np, concat_modalities)
            f_log_probs_tf, _ = run_generator(generator, f_gen_inp, f_mask, args.temperature)
            f_log_probs_np = f_log_probs_tf.numpy()
            f_mask_np = np.squeeze(f_mask.numpy(), axis=-1)
            opt_records: List[Dict] = []

            for idx, rec in enumerate(filtered_records):
                wt_ids = seq_ids_np[wt_idx_map[rec["base_id"]]]
                g_global, g_sub = compute_generator_nll(filt_seq_ids_np[idx], f_log_probs_np[idx], f_mask_np[idx], wt_ids=wt_ids)
                rec["g_global_nll"] = g_global
                rec["g_sub_nll"] = g_sub

            if not args.skip_optimize:
                opt_ids_np = argmax_from_log_probs(f_log_probs_tf, f_mask)
                opt_seqs = sequences_from_ids(opt_ids_np, f_mask_np)
                opt_ids = []
                opt_temps = []
                for idx, rec in enumerate(filtered_records):
                    wt_ids = seq_ids_np[wt_idx_map[rec["base_id"]]]
                    g_global, g_sub = compute_generator_nll(opt_ids_np[idx], f_log_probs_np[idx], f_mask_np[idx], wt_ids=wt_ids)
                    opt_label = f"{rec['label']}_opt"
                    opt_records.append({
                        "label": opt_label,
                        "base_id": rec["base_id"],
                        "rep": rec["rep"],
                        "kind": "optimized",
                        "seq": opt_seqs[idx],
                        "temp": rec["temp"],
                        "filter_threshold": args.esm_filter_threshold,
                        "g_global_nll": g_global,
                        "g_sub_nll": g_sub,
                        "identity_to_wt": compute_identity(opt_seqs[idx], wt_seq_map[rec["base_id"]]),
                    })
                    opt_ids.append(opt_label)
                    opt_temps.append(rec["temp"])

                opt_results = run_esm(tokenizer, model, opt_ids, [r["seq"] for r in opt_records], opt_temps, args)
                for rec in opt_records:
                    res = opt_results[rec["label"]]
                    esm_global, esm_sub = compute_esm_nll(rec["seq"], res["probs"], wt_seq_map[rec["base_id"]])
                    rec["esm_global_nll"] = esm_global
                    rec["esm_sub_nll"] = esm_sub
            else:
                print("[info] Skipping optimization step (--skip_optimize set).")
                opt_results = {}

            # Prepare for next cycle
            current_records = opt_records if opt_records else filtered_records
            current_results = opt_results if opt_records else filtered_results

            # Keep only the last cycle's outputs
            if cycle_idx == cycles - 1:
                filtered_final = filtered_records
                opt_final = opt_records

    records.extend(filtered_final)
    records.extend(opt_final)

    # Final safety filter after last optimization cycle
    if args.esm_filter_threshold is not None:
        final_targets = opt_final if opt_final else filtered_final
        if not final_targets:
            final_targets = sampling_final if sampling_final else current_records
        if final_targets:
            print("[info] Applying final post-optimization filtering...")
            wt_idx_map = {rid: idx for idx, rid in enumerate(fasta_ids)}
            # ESM on final targets
            final_results = run_esm_for_records(final_targets, tokenizer, model, args)
            # Apply filtering
            final_filtered_records: List[Dict] = []
            for rec in final_targets:
                res = final_results[rec["label"]]
                wt_seq = wt_seq_map[rec["base_id"]]
                filt_seq, reverted = filter_bad_mutations(rec["seq"], wt_seq, res["probs"], args.esm_filter_threshold)
                new_rec = dict(rec)
                new_rec["seq"] = filt_seq
                new_rec["reverted_positions"] = reverted
                new_rec["identity_to_wt"] = compute_identity(filt_seq, wt_seq)
                final_filtered_records.append(new_rec)

            # Recompute ESM on filtered seqs
            ff_results = run_esm_for_records(final_filtered_records, tokenizer, model, args)

            # Generator metrics on filtered seqs
            ff_ids = [r["label"] for r in final_filtered_records]
            ff_seqs = [r["seq"] for r in final_filtered_records]
            ff_seq_ids_np, ff_prob_q_np, _, _ = encode_sequences(ff_ids, ff_seqs, ff_results)
            ff_gen_inp, ff_mask = prepare_generator_inputs(ff_seq_ids_np, ff_prob_q_np, concat_modalities)
            ff_log_probs_tf, _ = run_generator(generator, ff_gen_inp, ff_mask, args.temperature)
            ff_log_probs_np = ff_log_probs_tf.numpy()
            ff_mask_np = np.squeeze(ff_mask.numpy(), axis=-1)

            for idx, rec in enumerate(final_filtered_records):
                wt_ids = seq_ids_np[wt_idx_map[rec["base_id"]]]
                g_global, g_sub = compute_generator_nll(ff_seq_ids_np[idx], ff_log_probs_np[idx], ff_mask_np[idx], wt_ids=wt_ids)
                rec["g_global_nll"] = g_global
                rec["g_sub_nll"] = g_sub
                res = ff_results[rec["label"]]
                esm_global, esm_sub = compute_esm_nll(rec["seq"], res["probs"], wt_seq_map[rec["base_id"]])
                rec["esm_global_nll"] = esm_global
                rec["esm_sub_nll"] = esm_sub

            # Replace final outputs with filtered versions
            if opt_final:
                opt_final = final_filtered_records
            else:
                filtered_final = final_filtered_records

    # Refresh records with final filtered outputs
    records = [rec for rec in records if rec.get("kind") not in {"filtered", "optimized"}]  # drop prior filtered/opt
    records.extend(filtered_final)
    records.extend(opt_final)

    tf.keras.backend.clear_session()
    del generator
    gc.collect()

    # OGT prediction
    if args.skip_ogt:
        print("[info] Skipping OGT prediction as requested.")
    else:
        print("[info] Predicting OGT for all sequences...")
        predict_ogt_for_records(records, args.ogt_config, args.ogt_weights, args.ogt_batch_size, args.ogt_max_len)


    # Write outputs (final)
    write_outputs(output_dir, args.epoch, records, args.store_softmax, base_softmax, mask_np, fasta_ids, args.name)


if __name__ == "__main__":
    main()
