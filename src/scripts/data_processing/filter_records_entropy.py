# filter_tfrecords_by_entropy_from_length.py
import os, glob, argparse
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
MAX_LEN = 512
EPS = 1e-8
LOG2 = tf.math.log(tf.constant(2.0, tf.float32))

def _get_feature(ex: tf.train.Example, name: str):
    feats = ex.features.feature
    return feats[name] if name in feats else None


def extract_scalar_int_from_example(raw: bytes, feature_name: str, *, fmt: str) -> Optional[int]:
    """
    fmt:
      - "int64_list": int64_list.value[0]
      - "bytes_tensor": bytes_list[0] is a serialized scalar tf.Tensor
    """
    ex = tf.train.Example()
    ex.ParseFromString(raw)
    f = _get_feature(ex, feature_name)
    v = f.int64_list.value
    return int(v[0])



def probs_q_to_512x21_tf(serialized: tf.Tensor) -> tf.Tensor:
    """
    serialized: scalar tf.string containing tf.io.serialize_tensor output of uint8 [L,20]
    returns: float32 [512,21]
    """
    raw = tf.io.parse_tensor(serialized, out_type=tf.uint8)  # [L,20]
    raw = tf.cast(raw, tf.float32) / 255.0
    raw = raw[:MAX_LEN, :]  # clip if longer
    raw = tf.pad(raw, [[0, tf.maximum(0, MAX_LEN - tf.shape(raw)[0])], [0, 0]])
    probs = tf.concat([raw, tf.zeros([MAX_LEN, 1], tf.float32)], axis=1)  # [512,21]
    probs = tf.ensure_shape(probs, [MAX_LEN, 21])
    return probs

@tf.function
def batch_keep_and_entropy(raw_batch, ent_min, ent_max, normalize_probs=False):
    """
    raw_batch: [B] tf.string (serialized tf.train.Example)
    returns:
      keep:    [B] tf.bool
      entropy: [B] tf.float32
    """
    feats = tf.io.parse_example(
        raw_batch,
        {
            "length": tf.io.FixedLenFeature([], tf.int64),
            "probs_q": tf.io.FixedLenFeature([], tf.string),
        },
    )
    lengths = tf.cast(tf.clip_by_value(feats["length"], 0, MAX_LEN), tf.int32)  # [B]
    probs_q = feats["probs_q"]  # [B] tf.string

    # Decode probs_q per element inside TF (still much cheaper than Python loop)
    probs = tf.map_fn(
        probs_q_to_512x21_tf,
        probs_q,
        fn_output_signature=tf.TensorSpec([MAX_LEN, 21], tf.float32),
        parallel_iterations=64,
    )  # [B,512,21]

    p = probs
    if normalize_probs:
        p = tf.maximum(p, 0.0)
        p = p / (tf.reduce_sum(p, axis=-1, keepdims=True) + EPS)

    ent_nats = -tf.reduce_sum(p * tf.math.log(p + EPS), axis=-1)   # [B,512]
    ent_bits = ent_nats / LOG2                                     # [B,512]

    mask = tf.sequence_mask(lengths, MAX_LEN, dtype=tf.float32)     # [B,512]
    denom = tf.reduce_sum(mask, axis=1) + EPS                       # [B]
    ent_per = tf.reduce_sum(ent_bits * mask, axis=1) / denom        # [B]

    # If length == 0, treat as "not keep" (you can change this)
    valid = lengths > 0
    keep = valid & (ent_per >= ent_min) & (ent_per <= ent_max)
    return keep, ent_per

AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(files, compression_type, batch_size, num_parallel_reads=AUTOTUNE):
    # Interleave reads across files for better throughput
    ds_files = tf.data.Dataset.from_tensor_slices(files)
    ds = ds_files.interleave(
        lambda fp: tf.data.TFRecordDataset(fp, compression_type=compression_type),
        cycle_length=num_parallel_reads if num_parallel_reads != AUTOTUNE else 16,
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def extract_array_from_example(
    raw: bytes,
    feature_name: str,
    *,
    fmt: str,
    dtype: str,
    shape: Tuple[int, ...],
) -> Optional[np.ndarray]:
    """
    fmt:
      - "float_list": TFExample float_list (flattened)
      - "int64_list": TFExample int64_list (flattened)
      - "bytes_tensor": TFExample bytes_list[0] is a serialized tf.Tensor (tf.io.parse_tensor)
      - "bytes_raw": TFExample bytes_list[0] is raw bytes for a NumPy buffer
    """
    ex = tf.train.Example()
    ex.ParseFromString(raw)
    f = _get_feature(ex, 'probs_q')
    serialized = f.bytes_list.value[0]
    f = probs_q_to_512x21(serialized)
    if f is None:
        return None

    arr = f.numpy()

    return arr


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-8)


def masked_avg_entropy_bits(
    feature: np.ndarray,
    length: int,
    *,
    max_len: int,
    from_logits: bool = False,
    eps: float = 1e-8,
    normalize_probs: bool = True,
) -> float:
    """
    feature: (max_len, C) probabilities or logits
    length: number of valid residues (>=0)
    returns: scalar average entropy over valid positions, in bits
    """
    if feature.ndim != 2:
        raise ValueError(f"feature must be rank-2 (L,C); got shape={feature.shape}")
    if feature.shape[0] != max_len:
        raise ValueError(f"feature first dim must be max_len={max_len}; got {feature.shape[0]}")

    # clamp length into [0, max_len]
    L = int(max(0, min(int(length), int(max_len))))

    if from_logits:
        p = softmax_np(feature.astype(np.float32), axis=-1)
    else:
        p = feature.astype(np.float32)
        if normalize_probs:
            p = np.maximum(p, 0.0)
            p = p / (np.sum(p, axis=-1, keepdims=True) + eps)

    ent_nats = -np.sum(p * np.log(p + eps), axis=-1)   # (max_len,)
    ent_bits = ent_nats / np.log(2.0)                  # (max_len,)

    if L == 0:
        # no valid positions; define as 0.0 or treat as missing elsewhere
        return 0.0

    return float(np.mean(ent_bits[:L]))


class ShardedWriter:
    def __init__(self, out_dir: str, prefix: str, shard_size_mb: int, compression: str):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_bytes = shard_size_mb * 1024 * 1024
        self.compression = compression
        os.makedirs(out_dir, exist_ok=True)
        self.idx = 0
        self.cur_bytes = 0
        self.w = None
        self._open()

    def _open(self):
        if self.w:
            self.w.close()
        suffix = ".tfrecord"
        if self.compression == "GZIP":
            suffix = ".tfrecord.gz"
        elif self.compression == "ZLIB":
            suffix = ".tfrecord.zlib"
        path = os.path.join(self.out_dir, f"{self.prefix}-{self.idx:05d}{suffix}")
        options = None if self.compression == "NONE" else tf.io.TFRecordOptions(compression_type=self.compression)
        self.w = tf.io.TFRecordWriter(path, options=options)
        self.idx += 1
        self.cur_bytes = 0

    def write(self, rec: bytes):
        if self.cur_bytes >= self.shard_bytes:
            self._open()
        self.w.write(rec)
        self.cur_bytes += len(rec)

    def close(self):
        if self.w:
            self.w.close()
            self.w = None


def should_keep(ent: float, ent_min: Optional[float], ent_max: Optional[float]) -> bool:
    if ent_min is None and ent_max is None:
        return True
    if ent_min is not None and ent < ent_min:
        return False
    if ent_max is not None and ent > ent_max:
        return False
    return True

def extract_id_bytes_from_example(raw: bytes, feature_name: str) -> bytes | None:
    ex = tf.train.Example()
    ex.ParseFromString(raw)
    feats = ex.features.feature
    if feature_name not in feats:
        return None
    v = feats[feature_name].bytes_list.value
    if not v:
        return None
    return v[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pattern", default="*tfrecord*", help="Glob inside input_dir")
    ap.add_argument("--compression", choices=["NONE", "GZIP", "ZLIB"], default="NONE")

    # Feature tensor (flattened L*C or serialized tensor), where L=max_len
    ap.add_argument("--feature_name", required=True)
    ap.add_argument("--feature_format", choices=["float_list", "int64_list", "bytes_tensor", "bytes_raw"], default="float_list")
    ap.add_argument("--feature_dtype", default="float32")

    # Length scalar used to construct mask
    ap.add_argument("--length_name", default="length")
    ap.add_argument("--length_format", choices=["int64_list", "bytes_tensor"], default="bint64_list")

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--channels", type=int, default=20)

    # Entropy params
    ap.add_argument("--from_logits", action="store_true")
    ap.add_argument("--normalize_probs", action="store_true")

    # Filtering rule
    ap.add_argument("--entropy_min", type=float, default=0.0, help="Keep only if entropy >= min")
    ap.add_argument("--entropy_max", type=float, default=3.5, help="Keep only if entropy <= max")

    # Output
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size_mb", type=int, default=512)

    # Missing/bad handling
    ap.add_argument("--missing_to", choices=["keep", "filter", "skip"], default="keep",
                    help="What to do if feature/length missing")
    ap.add_argument("--bad_to", choices=["skip", "filter"], default="skip",
                    help="What to do if parse/shape errors occur")

    # What to do if length==0 (no valid residues)
    ap.add_argument("--len0_to", choices=["keep", "filter", "skip"], default="filter")

    ap.add_argument("--log_every", type=int, default=1000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise FileNotFoundError("No TFRecord files matched.")

    out_filtered = ShardedWriter(args.out_dir, "filtered_out", args.shard_size_mb, args.compression)
    out_remaining = ShardedWriter(args.out_dir, "remaining", args.shard_size_mb, args.compression)

    comp = None if args.compression == "NONE" else args.compression

    total = kept = filt = missing = bad = 0
    ent_sum = 0.0
    ent_count = 0

    feat_shape = (args.max_len, args.channels)

    try:
        batch_size = 4096  # good starting point; try 1024/2048/4096/8192
        ent_min = tf.constant(args.entropy_min, tf.float32)
        ent_max = tf.constant(args.entropy_max, tf.float32)

        ds = make_dataset(files, comp, batch_size)

        total = kept = filt = 0
        ent_sum = 0.0
        ent_count = 0

        for raw_batch in ds:
            keep_tf, ent_tf = batch_keep_and_entropy(
                raw_batch,
                ent_min=ent_min,
                ent_max=ent_max,
                normalize_probs=args.normalize_probs,
            )

            raw_np = raw_batch.numpy()     # array of bytes, length B
            keep_np = keep_tf.numpy()      # bool array, length B
            ent_np  = ent_tf.numpy()       # float array, length B

            # stats
            bsz = raw_np.shape[0]
            total += bsz
            ent_sum += float(ent_np.sum())
            ent_count += bsz

            # write (still per-record, but now only I/O in Python)
            for rec, k in zip(raw_np, keep_np):
                if k:
                    out_remaining.write(rec); kept += 1
                else:
                    out_filtered.write(rec); filt += 1

            if args.log_every and (total % args.log_every == 0):
                print(f"seen={total:,} kept={kept:,} filtered={filt:,} mean_entropy(bits)={(ent_sum/max(ent_count,1)):.4f}")


    finally:
        out_filtered.close()
        out_remaining.close()

    mean_ent = ent_sum / max(ent_count, 1)
    print(
        f"\nDone. total={total:,} kept={kept:,} filtered={filt:,} "
        f"missing={missing:,} bad={bad:,} mean_entropy(bits)={mean_ent:.4f}"
    )


if __name__ == "__main__":
    main()
