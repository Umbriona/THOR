#!/usr/bin/env python
import argparse, os, math, pickle, sys, re
from pathlib import Path
from time import time
from typing import Iterable, List, Dict, Tuple, Optional

import torch
import numpy as np

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
        for si, s in enumerate(seqs):
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
            print(f"Sequence: {si} done")

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
    save_sharded(results, args.output, args.name, args.max_entries, qopt, args.keep_float)

    print(f"[done] saved to {args.output} | elapsed {(time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
