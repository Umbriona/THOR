import os
import glob
import tensorflow as tf

def merge_tfrecords(
    pattern,                      # e.g. "data/train-*.tfrecord.gz"
    out_path,                     # e.g. "data/train-merged.tfrecord.gz"
    report_every=100000           # print a progress line every N records
):
    # Collect & sort input files for deterministic ordering
    inputs = sorted(glob.glob(pattern))
    if not inputs:
        raise FileNotFoundError(f"No files match pattern: {pattern}")

    # Make sure we don't accidentally include the output as an input
    inputs = [p for p in inputs if os.path.abspath(p) != os.path.abspath(out_path)]
    print(f"Found {len(inputs)} input files.")

    # Create a dataset that streams across all files in parallel
    ds = tf.data.TFRecordDataset(
        inputs,
        compression_type="GZIP",
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    # (Optional) Improve host I/O throughput a bit
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Write out as GZIP-compressed TFRecord
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    count = 0
    with tf.io.TFRecordWriter(out_path, options=options) as w:
        for raw in ds:
            w.write(raw.numpy())
            count += 1
            if report_every and count % report_every == 0:
                print(f"... wrote {count:,} records")

    print(f"✅ Done. Wrote {count:,} records to {out_path}")

# Example
merge_tfrecords("/data/records/train/Target_*.tfrecord.gz", "/data/records/train_Target.tfrecord.gz")