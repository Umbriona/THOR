#!/usr/bin/env python
import argparse, os, math, pickle, sys, re
from pathlib import Path
from time import time
from typing import Iterable, List, Dict, Tuple, Optional

import torch
import numpy as np
import tensorflow as tf

import json
from tqdm import tqdm

try:
    from Bio import SeqIO
except ImportError:
    print("Biopython is required (pip install biopython).", file=sys.stderr)
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForMaskedLM


AA20 = "ACDEFGHIKLMNPQRSTVWY"   # canonical 20 amino acids (order stable)
AA_SET = set(AA20 + "XBZJUO*")  # allow uncommon letters; map to X


def clean_seq(seq: str) -> str:
    s = re.sub(r"\s+", "", seq.upper())
    # replace any non AA with X
    return "".join(ch if ch in AA_SET else "X" for ch in s)

# Prepare sequences for ProtBert
def to_protbert(seq: str) -> str:
    return " ".join(list(seq.replace(" ", "").upper()))

def parse_id_and_temp(rec_id: str, default_temp: Optional[float]) -> Tuple[str, Optional[float]]:
    """
    Expect headers like:
      >SEQ123 37.5
    I.e., the temperature appears after the first space. We'll parse the second token as float if present.
    """
    # strip '>' if present (SeqIO .id is already sans '>')
    parts = rec_id.split(" ")
    rid = parts[0]
    temp = default_temp
    if len(parts) > 1:
        try:
            temp = float(parts[-1])
        except: 
            print(f"No temperature info, using default: {temp}")
    return rid, temp


def load_fasta_with_meta(fpath: str, default_temp: Optional[float], args=None) -> Tuple[List[str], List[str], List[Optional[float]]]:
    records = []
    print(f"model is: {args.model.split('/')[-1]}")
    for rec in SeqIO.parse(fpath, "fasta"):
        #rid, temp = parse_id_and_temp(rec.description, default_temp)
        seq = clean_seq(str(rec.seq))
        if args.model.split("/")[-1] in ["prot_bert", "prot_bert_bfd"]:
            seq = to_protbert(seq)
        try:
            temp=float(rec.description.split()[-1])
        except:
            temp=float(rec.description.split()[-2])
        records.append((rec.id, seq, temp))
    # sort by length for nicer batching
    records.sort(key=lambda x: len(x[1]), reverse=True)
    ids = [r[0] for r in records]
    seqs = [r[1] for r in records]
    temps = [r[2] for r in records]
    return ids, seqs, temps


def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def tokenize_batch(tokenizer, seqs: List[str], max_length: int, device):
    return tokenizer(
        seqs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)


def mask_inputs(inputs, mask_positions_per_example, mask_token_id, pad_token_id):
    masked = {k: v.clone() for k, v in inputs.items()}
    input_ids = masked["input_ids"]
    for i, pos_list in enumerate(mask_positions_per_example):
        for pos in pos_list:
            if 0 <= pos < input_ids.shape[1] and input_ids[i, pos] != pad_token_id:
                input_ids[i, pos] = mask_token_id
    return masked


def positions_for_random_mask_partition(tokenizer, seqs: List[str],
                                        mask_prob: float,
                                        rng: np.random.Generator):
    """
    For each sequence:
      - collect token positions that correspond to residues (exclude specials/pad)
      - randomly permute them
      - split into consecutive chunks of size k ≈ round(mask_prob * L), last chunk shorter
    Returns: per-seq list of chunks (each chunk = sorted list of token positions to mask).
    """
    all_masks = []
    specials = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                tokenizer.eos_token_id, tokenizer.bos_token_id,
                tokenizer.pad_token_id}
    for s in seqs:
        enc = tokenizer(s, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        seq_positions = [i for i, tid in enumerate(ids)
                         if tid not in specials and tid != tokenizer.mask_token_id]
        L = len(seq_positions)
        # chunk size k in [1, L]
        k = max(1, min(L, int(round(mask_prob * L))))
        # random permutation, then chunk
        perm = rng.permutation(seq_positions)
        chunks = [sorted(perm[i:i+k].tolist()) for i in range(0, L, k)]
        all_masks.append(chunks)
    return all_masks


def gather_probs_at_positions(logits: torch.Tensor,
                              positions_per_example: List[List[int]],
                              aa_token_ids: List[int]) -> List[np.ndarray]:
    probs = torch.softmax(logits, dim=-1)
    out = []
    for i, pos_list in enumerate(positions_per_example):
        if len(pos_list) == 0:
            out.append(np.zeros((0, 20), dtype=np.float32))
            continue
        p = probs[i, torch.tensor(pos_list, device=probs.device), :]
        p20 = p[:, aa_token_ids]
        out.append(p20.detach().to(torch.float32).cpu().numpy())
    return out


def compute_single_mode(model, tokenizer, ids, seqs, temps, max_length, device,
                        aa_token_ids, batch_tokens, pad_token_id, mask_token_id):
    model.eval()
    results: Dict[str, Dict] = {}

    enc_all = tokenizer(seqs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids_all = enc_all["input_ids"]
    attention_mask_all = enc_all["attention_mask"]
    specials = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                tokenizer.eos_token_id, tokenizer.bos_token_id,
                tokenizer.pad_token_id}

    seq_token_positions = []
    for i in range(len(seqs)):
        ids_row = input_ids_all[i]
        pos = [j for j, tid in enumerate(ids_row.tolist())
               if tid not in specials and tid != tokenizer.mask_token_id and attention_mask_all[i, j] == 1]
        seq_token_positions.append(pos)
        results[ids[i]] = {
            "seq": seqs[i],
            "temp": temps[i],
            "length": len(pos),
            # "probs" will be filled later
        }

    work = []
    for si, positions in enumerate(seq_token_positions):
        for local_idx, tok_pos in enumerate(positions):
            work.append((si, tok_pos, local_idx))

    mean_tokens = int(attention_mask_all.sum(dim=1).float().mean().item())
    variants_per_batch = max(1, batch_tokens // max(16, mean_tokens))

    # allocate storage lazily (we don’t know L per seq until above)
    for rid in results:
        L = results[rid]["length"]
        results[rid]["probs"] = np.zeros((L, 20), dtype=np.float32)

    with torch.no_grad():
        for chunk in chunked(work, variants_per_batch):
            seq_batch = [seqs[si] for (si, _, _) in chunk]
            inputs = tokenize_batch(tokenizer, seq_batch, max_length, device)
            pos_lists = [[tok_pos] for (_, tok_pos, _) in chunk]
            masked_inputs = mask_inputs(inputs, pos_lists, mask_token_id, pad_token_id)

            logits = model(**masked_inputs).logits
            probs_list = gather_probs_at_positions(logits, [pl for pl in pos_lists], aa_token_ids)

            for (si, _, local_idx), p in zip(chunk, probs_list):
                results[ids[si]]["probs"][local_idx, :] = p[0]

    return results

def chunked_iterable(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

WorkItem = Tuple[int, int, int]
def chunk_by_token_budget(
    work: List[WorkItem],
    seq_true_len: List[int],
    budget: int,
    quadratic: bool = False,
    round_len_to: int = 8,     # align to Tensor Core–friendly sizes
    hard_cap_items: int = None # optional: cap items per chunk for memory safety
) -> Iterable[List[WorkItem]]:
    """
    Yield chunks whose estimated cost stays within 'budget'.
    Cost is B * Lmax (linear) or B * Lmax^2 (quadratic), where Lmax is the
    max true length among sequences in the chunk, optionally rounded up
    to 'round_len_to' to reflect padding-to-multiple-of behavior.
    """
    # Optionally sort by length so batches are homogeneous (highly recommended)
    work_sorted = sorted(work, key=lambda w: seq_true_len[w[0]])

    chunk: List[WorkItem] = []
    Lmax = 0

    def est_cost(B: int, L: int) -> int:
        if quadratic:
            return B * (L * L)
        return B * L

    for item in work_sorted:
        si, _, _ = item
        L = seq_true_len[si]
        # round up L to pad-to-multiple-of (keeps estimate closer to reality)
        if round_len_to and round_len_to > 1:
            L = ((L + round_len_to - 1) // round_len_to) * round_len_to

        new_Lmax = max(Lmax, L)
        new_B = len(chunk) + 1
        new_cost = est_cost(new_B, new_Lmax)

        over_items = (hard_cap_items is not None and new_B > hard_cap_items)

        if chunk and (new_cost > budget or over_items):
            # flush current chunk
            yield chunk
            chunk = [item]
            Lmax = L
        else:
            chunk.append(item)
            Lmax = new_Lmax

    if chunk:
        yield chunk

def compute_single_mode_fast(
    model,
    tokenizer,
    ids: List[str],
    seqs: List[str],
    temps: List[float],
    max_length: int,
    device: torch.device,
    aa_token_ids: List[int],
    batch_tokens: int,
    pad_token_id: int,
    mask_token_id: int,
    use_amp_bf16: bool = True,  # A100: great speedup
) -> Dict[str, Dict]:
    model.eval()

    # 1) Tokenize once on CPU; pin for faster H2D
    enc_all = tokenizer(
        seqs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    ##print(f"AAs in Seqs is: {sum([len(seq) for seq in seqs])}")
    ##print(f"Shape inputs_ids is:{enc_all['input_ids'].size()} \n  ")


    input_ids_all = enc_all["input_ids"] .contiguous().pin_memory()
    attention_mask_all = enc_all["attention_mask"].contiguous().pin_memory()
    true_len = attention_mask_all.sum(dim=1).to(torch.int).tolist()
    # 2) Figure out positions to mask per sequence (excluding specials)
    specials = {
        tokenizer.cls_token_id, tokenizer.sep_token_id,
        tokenizer.eos_token_id, tokenizer.bos_token_id,
        tokenizer.pad_token_id
    }

    seq_token_positions: List[List[int]] = []
    for i in range(len(seqs)):
        row = input_ids_all[i]
        am = attention_mask_all[i]
        pos = [j for j, tid in enumerate(row.tolist())
               if (tid not in specials) and (tid != tokenizer.mask_token_id) and (am[j].item() == 1)]
        seq_token_positions.append(pos)
    ##print(f" seq_tok_positions is: {seq_token_positions}\n")
    # Results scaffold
    results: Dict[str, Dict] = {}
    for i, rid in enumerate(ids):
        L = len(seq_token_positions[i])
        results[rid] = {
            "seq": seqs[i],
            "temp": temps[i],
            "length": L,
            "probs": np.zeros((L, 20), dtype=np.float32),
        }

    # 3) Build flat work list: (seq_index, token_pos, local_idx_in_seq)
    work: List[Tuple[int, int, int]] = []
    for si, positions in enumerate(seq_token_positions):
        for local_idx, tok_pos in enumerate(positions):
            work.append((si, tok_pos, local_idx))
    ##print(f"len(Work) is: {len(work)}")
    # 4) Choose variants_per_batch by token budget
    #    Use average effective length (sum of attention mask) to estimate
    mean_tokens = int(attention_mask_all.sum(dim=1).float().mean().item())
    variants_per_batch = max(1, batch_tokens // max(16, mean_tokens))
    
    ##print(f"Mean tokens is: {mean_tokens}")
    ##print(f"Variants per batch is: {variants_per_batch}")

    # Prebuild AA ids on device
    aa_ids_dev = torch.as_tensor(aa_token_ids, device=device, dtype=torch.long)

    # Single CUDA stream is fine here; for even more overlap, you can use streams.
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (use_amp_bf16 and device.type == "cuda") else torch.cuda.amp.autocast(enabled=False)

    with torch.inference_mode(), autocast_ctx:
        #for chunk in chunked_iterable(work, variants_per_batch):
        work_done = 0
        iteration = 0
        for chunk in chunk_by_token_budget(
        work,
        seq_true_len=true_len,
        budget=20_000,          # your target
        quadratic=False,        # set True to approximate attention FLOPs
        round_len_to=8,
        hard_cap_items=None):
            B = len(chunk)
           ## print(f"Length cunk (B) is: {B}")
            if B == 0:
                continue
          ##  #print(f"Chunk: {chunk}")
            # Indices and positions for this batch
            seq_idx_cpu = torch.tensor([si for (si, _, _) in chunk], dtype=torch.long)  # CPU
            pos_dev = torch.tensor([p for (_, p, _) in chunk], device=device, dtype=torch.long)

            ##print(f"seq_idx_cpu: {seq_idx_cpu}")
            ##print(f"pos_dev: {pos_dev}")

            # 5) Slice and send tokenized inputs to device (non_blocking from pinned)
            input_ids = input_ids_all.index_select(0, seq_idx_cpu).to(device, non_blocking=True)
            attn_mask = attention_mask_all.index_select(0, seq_idx_cpu).to(device, non_blocking=True)

            L_batch = int(attn_mask.sum(dim=1).max().item())   # max non-pad length in batch
            input_ids = input_ids[:, :L_batch].clone()
            attn_mask = attn_mask[:, :L_batch]

            # 6) Mask the one position per variant (in-place on a clone)
            input_ids = input_ids.clone()
            input_ids[torch.arange(B, device=device), pos_dev] = mask_token_id

            # 7) Forward
            ##print(f"Size of input to model: {input_ids.size()}")
            ##print(f"Input to model: {input_ids}")

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits  # [B, L, V]

            # 8) Select only the masked positions first, then normalize over vocab
            #    selected_logits: [B, V]
            selected_logits = logits[torch.arange(B, device=logits.device), pos_dev, :]

            # Gather AA logits then normalize with logsumexp for stability
            aa_logits = selected_logits.index_select(-1, aa_ids_dev)         # [B, 20]
            logZ = torch.logsumexp(selected_logits, dim=-1, keepdim=True)    # [B, 1]
            aa_probs = (aa_logits - logZ).exp()                               # [B, 20]

            # 9) Single D2H copy for the whole batch
            aa_probs_cpu = aa_probs.to(torch.float32).cpu().numpy()          # (B, 20)

            # 10) Write back to per-seq slots
            for (si, _, local_idx), row in zip(chunk, aa_probs_cpu):
                results[ids[si]]["probs"][local_idx, :] = row
            work_done += len(chunk)
            print(f"Iteration: {iteration} done; {1-(work_done/len(work))} left to do")
            iteration += 1
            
    return results


def compute_random_partition_mode(model, tokenizer, ids, seqs, temps, max_length, device,
                                  aa_token_ids, mask_prob, chunk_batch,
                                  pad_token_id, mask_token_id, seed):
    """
    Random-partition masking: each residue masked exactly once.
    For each sequence, create shuffled chunks (~mask_prob fraction each),
    run inference per chunk, and write per-position probabilities.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    results: Dict[str, Dict] = {}

    # plan masks (per-seq list of chunks)
    all_masks = positions_for_random_mask_partition(tokenizer, seqs, mask_prob, rng)

    with torch.no_grad():
        for si, s in enumerate(tqdm(seqs)):
            # set up result record
            # also define the ordered list of "sequence token positions" and a map to local indices
            enc = tokenizer(s, return_tensors="pt")
            ids_row = enc["input_ids"][0]
            specials = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                        tokenizer.eos_token_id, tokenizer.bos_token_id,
                        tokenizer.pad_token_id}
            positions = [j for j, tid in enumerate(ids_row.tolist())
                         if tid not in specials and tid != tokenizer.mask_token_id]
            L = len(positions)
            tokpos2local = {p: i for i, p in enumerate(positions)}

            results[ids[si]] = {
                "seq": s,
                "temp": temps[si],
                "length": L,
                "probs": np.zeros((L, 20), dtype=np.float32),
            }

            # process this sequence's chunks in mini-batches
            for rep_chunk in chunked(all_masks[si], max(1, chunk_batch)):
                # Build a batch of masked variants of *the same* sequence
                seq_batch = [s] * len(rep_chunk)
                inputs = tokenize_batch(tokenizer, seq_batch, max_length, device)
                masked_inputs = mask_inputs(inputs, rep_chunk, mask_token_id, pad_token_id)
                logits = model(**masked_inputs).logits  # [B, T, V]

                # For each example in the batch, write the probs for its masked positions
                for bi, pos_list in enumerate(rep_chunk):
                    if not pos_list:
                        continue
                    p = torch.softmax(logits[bi], dim=-1)[:, aa_token_ids]  # [T,20]
                    # gather and assign
                    sel = p[torch.tensor(pos_list, device=p.device), :].detach().to(torch.float32).cpu().numpy()
                    for tokpos, row in zip(pos_list, sel):
                        local_i = tokpos2local.get(tokpos, None)
                        if local_i is not None:
                            results[ids[si]]["probs"][local_i, :] = row.astype(np.float32)
            #print(f"Sequence: {si} done")

    return results


def quantize_probs(probs: np.ndarray, qdtype: Optional[str]) -> Tuple[Dict, Optional[np.ndarray]]:
    """
    probs: float32 array [L,20] in [0,1].
    qdtype: None | 'uint8' | 'uint16'
    Returns: (qinfo, qarray or None)
    """
    if qdtype is None:
        return {}, None
    if qdtype not in {"uint8", "uint16"}:
        raise ValueError("qdtype must be one of: uint8, uint16, or None")

    if qdtype == "uint8":
        scale = 255.0
        arr = np.rint(np.clip(probs, 0.0, 1.0) * scale).astype(np.uint8)
    else:
        scale = 65535.0
        arr = np.rint(np.clip(probs, 0.0, 1.0) * scale).astype(np.uint16)

    qinfo = {
        "dtype": qdtype,
        "scale": scale,          # linear scale in [0,1], dequantize via arr/scale
        "aa_order": AA20,
        "encoding": "linear01"
    }
    return qinfo, arr


def save_sharded(results_dict: Dict[str, Dict], out_dir: str, base_name: str,
                 max_entries: int, quantize: Optional[str], keep_float: bool):
    """
    Each shard is a dict:
      id -> {
        "seq": str,
        "temp": float or None,
        "length": int,
        "probs": float32 [L,20] (optional, if keep_float=True),
        "probs_q": uint8/uint16 [L,20] (if quantize set),
        "q_info": {...} (if quantize set)
      }
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(results_dict.items())
    shard_idx = 0
    for i in range(0, len(items), max_entries):
        shard = {}
        for rid, entry in items[i:i+max_entries]:
            rec = {
                "seq": entry["seq"],
                "temp": float(entry["temp"]) if entry.get("temp") is not None else None,
                "length": int(entry["length"]),
            }
            probs = entry["probs"]
            if quantize is not None:
                q_info, arr_q = quantize_probs(probs, quantize)
                rec["probs_q"] = arr_q
                rec["q_info"] = q_info
                if keep_float:
                    rec["probs"] = probs.astype(np.float32)
            else:
                rec["probs"] = probs.astype(np.float32)
            shard[rid] = rec

        fpath = out_dir / f"{base_name}_{shard_idx}.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[write] {fpath} ({len(shard)} records)")
        shard_idx += 1


########### TFRecord functionality ############

def _bytes_feature(b: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))

def _int64_feature(x: int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x)]))

def _float_feature(x: float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x)]))

def _str_feature(s: str):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[s.encode("utf-8")]))

def _serialize_ndarray(arr: np.ndarray) -> bytes:
    # use TF to serialize as a Tensor (includes dtype + shape)
    t = tf.convert_to_tensor(arr)
    return tf.io.serialize_tensor(t).numpy()

def to_tfexample(rec_id: str, entry: Dict) -> tf.train.Example:
    feats = {
        "id": _str_feature(rec_id),
        "seq": _str_feature(entry["seq"]),
        "length": _int64_feature(int(entry["length"])),
    }
    # temperature may be None
    if entry.get("temp") is None:
        feats["temp_is_null"] = _int64_feature(1)
    else:
        feats["temp"] = _float_feature(float(entry["temp"]))
        feats["temp_is_null"] = _int64_feature(0)

    # probabilities: float or quantized
    if "probs" in entry and entry["probs"] is not None:
        feats["probs"] = _bytes_feature(_serialize_ndarray(entry["probs"].astype(np.float32)))
    if "probs_q" in entry and entry["probs_q" ] is not None:
        feats["probs_q"] = _bytes_feature(_serialize_ndarray(entry["probs_q"]))
    if "q_info" in entry and entry["q_info"] is not None:
        feats["q_info_json"] = _str_feature(json.dumps(entry["q_info"]))

    if "quality" in entry and entry["quality"] is not None:
        # Store as float; for mean_logprob it's <=0, for others in (0,1]
        feats["quality"] = _float_feature(float(entry["quality"]))

    return tf.train.Example(features=tf.train.Features(feature=feats))

def save_tfrecord_sharded(results_dict: Dict[str, Dict],
                          out_dir: str,
                          base_name: str,
                          max_entries: int,
                          compress: str = "gzip"):  # "none" or "gzip"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = list(results_dict.items())
    shard_idx = 0
    for i in range(0, len(items), max_entries):
        shard_items = items[i:i+max_entries]
        tfrec_path = out / f"{base_name}_{shard_idx:05d}.tfrecord"
        options = None
        if compress and compress.lower() == "gzip":
            options = tf.io.TFRecordOptions(compression_type="GZIP")
            tfrec_path = tfrec_path.with_suffix(tfrec_path.suffix + ".gz")

        with tf.io.TFRecordWriter(str(tfrec_path), options=options) as w:
            for rid, entry in shard_items:
                ex = to_tfexample(rid, entry)
                w.write(ex.SerializeToString())
        print(f"[write] {tfrec_path} ({len(shard_items)} records)")
        shard_idx += 1


###############################################


#################### Computing liklihoods ##############

def _seq_letters(s: str) -> str:
    # works for both ProtBert (space-separated) and plain
    return s.replace(" ", "")

_AA2IDX = {aa: i for i, aa in enumerate(AA20)}

def compute_quality_for_record(entry: Dict, metric: str = "geomean_prob") -> Optional[float]:
    """
    entry: {"seq": str, "probs": np.ndarray [L,20], "length": int}
    metric: one of "mean_prob", "geomean_prob", "mean_logprob"
    Returns float in (0,1], or log-space for mean_logprob, or None if not computable.
    """
    seq = _seq_letters(entry["seq"])
    probs = entry.get("probs", None)
    if probs is None or probs.size == 0:
        return None
    # Build index of true residues for positions we scored (length==probs.shape[0])
    idxs = []
    for ch in seq[:probs.shape[0]]:  # guard if any mismatch
        j = _AA2IDX.get(ch, None)
        idxs.append(j)
    idxs = np.array(idxs, dtype=object)
    mask = np.array([j is not None for j in idxs], dtype=bool)
    if mask.sum() == 0:
        return None
    row_idx = np.nonzero(mask)[0]
    col_idx = np.array([idxs[k] for k in row_idx], dtype=int)
    p = probs[row_idx, col_idx].astype(np.float64)
    p = np.clip(p, 1e-12, 1.0)

    if metric == "mean_prob":
        return float(p.mean())
    elif metric == "geomean_prob":
        return float(np.exp(np.log(p).mean()))
    elif metric == "mean_logprob":
        return float(np.log(p).mean())  # note: <= 0
    else:
        return float(np.exp(np.log(p).mean()))  # default

########################################################

#################### Grouping #########################

def _parse_bins(s: Optional[str]) -> Optional[List[float]]:
    if not s: 
        return None
    edges = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(edges) < 2:
        raise ValueError("--quality_bins must have at least two comma-separated edges")
    if any(not (0.0 <= e <= 1.0) for e in edges):
        raise ValueError("quality bin edges must be in [0,1]")
    if sorted(edges) != edges:
        raise ValueError("quality bin edges must be sorted ascending")
    return edges

def _iter_shards(sorted_items, max_entries):
    # sorted_items: list of (rid, entry) already sorted by quality desc
    for i in range(0, len(sorted_items), max_entries):
        yield sorted_items[i:i+max_entries]

def _format_q(q: float, places: int = 4) -> str:
    return f"{q:.{places}f}"

def build_shards_by_quality(results: Dict[str, Dict],
                            quality_metric: str,
                            min_quality: Optional[float],
                            drop_nan_quality: bool,
                            bins: Optional[List[float]],
                            max_entries: int):
    """
    Yields tuples: (label_suffix, shard_items)
      - label_suffix becomes part of filename, includes qmin for this shard
      - shard_items is list[(rid, prepared_entry_dict)]
    The entry dict gets an added 'quality' key.
    """
    # 1) compute quality
    items = []
    for rid, entry in results.items():
        q = compute_quality_for_record(entry, metric=quality_metric)
        if q is None and drop_nan_quality:
            continue
        if (min_quality is not None) and (q is not None) and (q < min_quality):
            continue
        e = dict(entry)
        e["quality"] = q
        items.append((rid, e))

    if not items:
        return  # nothing to yield

    # Decide comparator direction:
    # For mean_prob/geomean_prob higher is better; for mean_logprob (<=0) less negative is better => still sort desc.
    items.sort(key=lambda x: (x[1]["quality"] if x[1]["quality"] is not None else -np.inf), reverse=True)

    if not bins:
        # No binning: shard the full sorted list
        for shard in _iter_shards(items, max_entries):
            qmin = min(e["quality"] for _, e in shard if e["quality"] is not None)
            yield (f"qmin{_format_q(qmin)}", shard)
        return

    # Binning
    edges = bins
    for bi in range(len(edges) - 1):
        lo, hi = edges[bi], edges[bi+1]
        in_bin = [(rid, e)
                  for (rid, e) in items
                  if (e["quality"] is not None) and (lo <= e["quality"] < hi or (bi == len(edges)-2 and e["quality"] == hi))]
        if not in_bin:
            continue
        # already globally sorted desc; keep that order
        for shard in _iter_shards(in_bin, max_entries):
            qmin = min(e["quality"] for _, e in shard if e["quality"] is not None)
            yield (f"bin{bi:02d}_lo{_format_q(lo)}_qmin{_format_q(qmin)}", shard)

########################################################

######################### Saving Bined/Grouped ######################

def save_grouped_pickle(grouped_iter, out_dir: str, base_name: str,
                        quantize: Optional[str], keep_float: bool):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shard_counter = 0
    for label, shard in grouped_iter:
        shard_dict = {}
        for rid, entry in shard:
            rec = {
                "seq": entry["seq"],
                "temp": float(entry["temp"]) if entry.get("temp") is not None else None,
                "length": int(entry["length"]),
                "quality": entry.get("quality", None),
            }
            probs = entry["probs"]
            if quantize is not None:
                q_info, arr_q = quantize_probs(probs, quantize)
                rec["probs_q"] = arr_q
                rec["q_info"] = q_info
                if keep_float:
                    rec["probs"] = probs.astype(np.float32)
            else:
                rec["probs"] = probs.astype(np.float32)
            shard_dict[rid] = rec

        fpath = out / f"{base_name}_{label}_{shard_counter:05d}.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(shard_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[write] {fpath} ({len(shard)} records)")
        shard_counter += 1

def save_grouped_tfrecord(grouped_iter, out_dir: str, base_name: str,
                          quantize: Optional[str], keep_float: bool,
                          compress: str = "gzip"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shard_counter = 0
    for label, shard in grouped_iter:
        # prepare per-record quantization mirroring pickle path
        prepped = {}
        for rid, entry in shard:
            rec = {
                "seq": entry["seq"],
                "temp": float(entry["temp"]) if entry.get("temp") is not None else None,
                "length": int(entry["length"]),
                "quality": entry.get("quality", None),
            }
            probs = entry["probs"]
            if quantize is not None:
                q_info, arr_q = quantize_probs(probs, quantize)
                rec["probs_q"] = arr_q
                rec["q_info"] = q_info
                if keep_float:
                    rec["probs"] = probs.astype(np.float32)
            else:
                rec["probs"] = probs.astype(np.float32)
            prepped[rid] = rec

        tfrec_path = out / f"{base_name}_{label}_{shard_counter:05d}.tfrecord"
        options = None
        if compress and compress.lower() == "gzip":
            options = tf.io.TFRecordOptions(compression_type="GZIP")
            tfrec_path = tfrec_path.with_suffix(tfrec_path.suffix + ".gz")

        with tf.io.TFRecordWriter(str(tfrec_path), options=options) as w:
            for rid, rec in prepped.items():
                ex = to_tfexample(rid, rec)
                w.write(ex.SerializeToString())
        print(f"[write] {tfrec_path} ({len(shard)} records)")
        shard_counter += 1


###########################################################################


def main():
    p = argparse.ArgumentParser("Compute per-position substitution probabilities with ESM-1v (with quantization & metadata)")
    p.add_argument("-i", "--input", required=True, help="Input FASTA (header: '>ID <temp>')")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--name", default="sequence_mlm_features", help="Base name for shard files")
    p.add_argument("--model", default="facebook/esm1v_t33_650M_UR90S_1", help="HF model id or local path")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 inference if available")
    p.add_argument("--max_length", type=int, default=1022)
    p.add_argument("--mode", choices=["single", "random"], default="single",
                   help="'single' = leave-one-out; 'random' = mask p%% and average")
    p.add_argument("--mask_prob", type=float, default=0.15, help="Masking proportion for random mode")
    p.add_argument("--batch_tokens", type=int, default=40000,
                   help="Approx token budget per forward (single mode)")
    p.add_argument("--random_batch", type=int, default=16, help="Batch size (repeats per seq) in random mode")
    p.add_argument("--max_entries", type=int, default=50000, help="Records per pickle shard")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default_temp", type=float, default=None, help="Default temperature if not present/parsable in FASTA header")
    p.add_argument("--quantize", choices=["none", "uint8", "uint16"], default="none",
                   help="Quantize probs to shrink size")
    p.add_argument("--keep_float", action="store_true", help="Keep float32 probs alongside quantized")
    p.add_argument("--format", choices=["pickle", "tfrecord"], default="tfrecord",
                   help="Output format for shards")
    p.add_argument("--tfrecord_compress", choices=["none", "gzip"], default="gzip",
                   help="Compression for TFRecords")

    p.add_argument("--quality_metric",
                     choices=["mean_prob", "geomean_prob", "mean_logprob"],
                     default="geomean_prob",
                     help="How to aggregate per-position probabilities for the true residues")
    p.add_argument("--min_quality", type=float, default=None,
                    help="Drop sequences with quality below this threshold")
    p.add_argument("--quality_bins", type=str, default=None,
                    help='Comma-separated bin edges in (0,1], e.g. "0.0,0.1,0.2,0.5,1.0". '
                        'If unset, no binning; we just shard the globally sorted list.')
    p.add_argument("--drop_nan_quality", action="store_true",
                    help="Drop sequences whose quality can’t be computed (e.g., all residues not in AA20)")


    args = p.parse_args()

    t0 = time()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    if hasattr(model, "esm") and hasattr(model.esm, "contact_head"):
        model.esm.contact_head = None
    model.eval().to(device)
    if args.bf16 and device.type == "cuda":
        model.to(dtype=torch.bfloat16)

    # Token ids for AA20
    aa_token_ids = tokenizer.convert_tokens_to_ids(list(AA20))
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id

    # Read fasta (+ metadata)
    ids, seqs, temps = load_fasta_with_meta(args.input, args.default_temp, args)
    print(f"[load] {len(seqs)} sequences from {args.input}")

    # Compute per-mode
    if args.mode == "single":
        results = compute_single_mode_fast(
            model, tokenizer, ids, seqs, temps, args.max_length, device,
            aa_token_ids, args.batch_tokens, pad_token_id, mask_token_id
        )
    else:
        results = compute_random_partition_mode(
            model, tokenizer, ids, seqs, temps, args.max_length, device,
            aa_token_ids, args.mask_prob, args.random_batch,
            pad_token_id, mask_token_id, args.seed
        )

    # Save shards with quantization
    qopt = None if args.quantize == "none" else args.quantize
    bins = _parse_bins(args.quality_bins)

    # Build grouped iterator: (label_suffix, shard_items)
    grouped = build_shards_by_quality(
        results,
        quality_metric=args.quality_metric,
        min_quality=args.min_quality,
        drop_nan_quality=args.drop_nan_quality,
        bins=bins,
        max_entries=args.max_entries,
    )

    # Guard: if nothing left after filtering
    grouped = list(grouped)
    if len(grouped) == 0:
        print("[warn] No records to write after quality filtering.", file=sys.stderr)
        sys.exit(0)

    # Write
    if args.format == "pickle":
        save_grouped_pickle(grouped, args.output, args.name, qopt, args.keep_float)
    else:
        save_grouped_tfrecord(grouped, args.output, args.name, qopt, args.keep_float, args.tfrecord_compress)

    print(f"[done] saved to {args.output} | elapsed {(time()-t0)/60:.1f} min")



if __name__ == "__main__":
    main()
