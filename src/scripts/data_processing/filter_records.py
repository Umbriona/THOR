# filter_tfrecords_by_hashed_id_batched.py
import os, glob, argparse, hashlib
import numpy as np
import tensorflow as tf


def hash64_bytes_batch(ids: np.ndarray) -> np.ndarray:
    """
    ids: np.ndarray of dtype=object or bytes-like (shape [B])
    returns uint64 array [B] with blake2b-64
    """
    out = np.empty((len(ids),), dtype=np.uint64)
    for i, b in enumerate(ids):
        # b can be numpy bytes_ or python bytes
        if isinstance(b, np.ndarray):
            b = b.tobytes()
        elif isinstance(b, np.bytes_):
            b = bytes(b)
        h = hashlib.blake2b(b, digest_size=8).digest()
        out[i] = np.frombuffer(h, dtype=np.uint64)[0]
    return out


def contains_sorted_uint64_batch(sorted_arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    sorted_arr: sorted uint64 array (can be memmap)
    x: uint64 array [B]
    returns bool array [B]
    """
    idx = np.searchsorted(sorted_arr, x)
    ok = idx < sorted_arr.size
    # avoid indexing out of bounds
    idx_safe = np.where(ok, idx, 0)
    hit = ok & (sorted_arr[idx_safe] == x)
    return hit


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


def make_dataset(files, compression_type, batch_size, cycle_length=16):
    AUTOTUNE = tf.data.AUTOTUNE
    ds_files = tf.data.Dataset.from_tensor_slices(files)
    ds = ds_files.interleave(
        lambda fp: tf.data.TFRecordDataset(fp, compression_type=compression_type),
        cycle_length=cycle_length,
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pattern", default="*.tfrecord*", help="Glob inside input_dir")
    ap.add_argument("--id_hashes_npy", required=True, help="Sorted uint64 hashes (.npy)")
    ap.add_argument("--id_feature", default="id", help="TFExample bytes feature holding the ID")
    ap.add_argument("--compression", choices=["NONE", "GZIP", "ZLIB"], default="NONE")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size_mb", type=int, default=512)
    ap.add_argument("--missing_to", choices=["keep", "filter", "skip"], default="keep")

    # batching knobs
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--cycle_length", type=int, default=16, help="parallel file interleave")
    ap.add_argument("--log_every", type=int, default=100000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise FileNotFoundError("No TFRecord files matched.")

    hashes = np.load(args.id_hashes_npy, mmap_mode="r")
    if hashes.dtype != np.uint64:
        hashes = hashes.astype(np.uint64, copy=False)

    out_filtered = ShardedWriter(args.out_dir, "filtered_out", args.shard_size_mb, args.compression)
    out_remaining = ShardedWriter(args.out_dir, "remaining", args.shard_size_mb, args.compression)

    comp = None if args.compression == "NONE" else args.compression

    ds = make_dataset(files, comp, args.batch_size, cycle_length=args.cycle_length)

    # batch parse spec
    spec = {args.id_feature: tf.io.FixedLenFeature([], tf.string, default_value=b"")}

    total = kept = filt = missing = bad = 0

    try:
        for raw_batch in ds:
            # raw_batch: [B] tf.string (each element is serialized Example bytes)
            try:
                parsed = tf.io.parse_example(raw_batch, spec)
            except Exception:
                # If parsing a whole batch fails, fall back to per-record skip
                bad += int(raw_batch.shape[0])
                continue

            ids_tf = parsed[args.id_feature]  # [B] tf.string, default b""
            ids_np = ids_tf.numpy()           # [B] bytes
            raw_np = raw_batch.numpy()        # [B] bytes (the serialized Example)

            # missing ids handling
            is_missing = np.fromiter((len(x) == 0 for x in ids_np), count=len(ids_np), dtype=np.bool_)

            # compute hashes only for non-missing
            keep_mask = np.ones((len(ids_np),), dtype=np.bool_)

            non_missing_idx = np.where(~is_missing)[0]
            if non_missing_idx.size > 0:
                h = hash64_bytes_batch(ids_np[non_missing_idx])
                in_set = contains_sorted_uint64_batch(hashes, h)
                # in_set==True => filtered out
                keep_mask[non_missing_idx] = ~in_set

            # decide what to do with missing
            if is_missing.any():
                missing += int(is_missing.sum())
                if args.missing_to == "skip":
                    keep_mask[is_missing] = False  # and also don't write to filtered
                    # We'll handle skip below
                elif args.missing_to == "filter":
                    keep_mask[is_missing] = False
                else:
                    keep_mask[is_missing] = True

            # write
            for rec, km, miss in zip(raw_np, keep_mask, is_missing):
                total += 1
                if miss and args.missing_to == "skip":
                    continue
                if km:
                    out_remaining.write(rec); kept += 1
                else:
                    out_filtered.write(rec); filt += 1

            if args.log_every and (total % args.log_every == 0):
                print(f"seen={total:,} filtered={filt:,} kept={kept:,} missing={missing:,} bad={bad:,}")

    finally:
        out_filtered.close()
        out_remaining.close()

    print(f"\nDone. total={total:,} filtered={filt:,} kept={kept:,} missing={missing:,} bad={bad:,}")


if __name__ == "__main__":
    main()
