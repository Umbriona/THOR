#!/usr/bin/env python

"""
Fine-tune ESM-1v on masked language modeling (MLM).

- Input: FASTA (.fa/.fasta) or plain text (.txt) with one amino-acid sequence per line.
- Objective: MLM with 15% masking (same as ESM pretraining / BERT-style).
- Dist: Use torchrun for multi-GPU (DDP). Works with 1 node, 4 GPUs.

Example:
  torchrun --nproc_per_node=4 train_esm_mlm.py \
    --model_name facebook/esm1v_t33_650M_UR90S_1 \
    --data_path /path/to/sequences.fasta \
    --output_dir /path/to/out \
    --max_length 1022 \
    --epochs 2 \
    --per_device_batch 4 \
    --accum 4 \
    --lr 1e-4

References:
- HF Transformers ESM docs (MLM, tokenization, Trainer). 
- ESM-1v model IDs (facebook/esm1v_t33_650M_UR90S_[1..5]).
"""

import argparse
import os
import re
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

AA_VALID = set(list("ACDEFGHIKLMNPQRSTVWYXBZJUO*"))  # tolerate uncommon letters; tokenizer will map/unk

def is_fasta(path: Path):
    return path.suffix.lower() in {".fa", ".fasta", ".faa"}

def fasta_to_txt(fasta_path: Path, out_txt_path: Path, min_len=1, max_len=10_000):
    """Convert FASTA to one-sequence-per-line .txt with basic cleaning."""
    n_in, n_out = 0, 0
    with open(fasta_path, "r") as fin, open(out_txt_path, "w") as fout:
        seq = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    s = "".join(seq).upper()
                    s = re.sub(r"\s+", "", s)
                    s = re.sub(r"[^A-Za-z\*\-]", "", s)
                    s = s.replace("-", "")  # drop gaps if any
                    if min_len <= len(s) <= max_len:
                        fout.write(s + "\n")
                        n_out += 1
                    seq = []
                n_in += 1
            else:
                seq.append(line)
        if seq:
            s = "".join(seq).upper()
            s = re.sub(r"\s+", "", s)
            s = re.sub(r"[^A-Za-z\*\-]", "", s)
            s = s.replace("-", "")
            if min_len <= len(s) <= max_len:
                fout.write(s + "\n")
                n_out += 1
    return n_in, n_out

def prepare_text_source(data_path: Path, max_len: int):
    """
    Returns path to a text file with one sequence per line, converting FASTA if needed.
    Supports a directory containing multiple files (mix of fasta/txt).
    """
    if data_path.is_file():
        if is_fasta(data_path):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            tmp.close()
            fasta_to_txt(data_path, Path(tmp.name), min_len=1, max_len=max_len)
            return Path(tmp.name)
        else:
            return data_path

    # directory: merge all into a temp txt
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.close()
    wrote = 0
    with open(tmp.name, "w") as fout:
        for p in sorted(data_path.glob("**/*")):
            if not p.is_file():
                continue
            if is_fasta(p):
                _, n_out = fasta_to_txt(p, Path(tmp.name), min_len=1, max_len=max_len)
                wrote += n_out
            elif p.suffix.lower() in {".txt"}:
                with open(p, "r") as fin:
                    for line in fin:
                        s = line.strip().upper()
                        if 0 < len(s) <= max_len:
                            fout.write(s + "\n")
                            wrote += 1
    return Path(tmp.name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/esm1v_t33_650M_UR90S_1")
    parser.add_argument("--data_path", type=str, required=True, help="FASTA or TXT file, or a directory of them")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1022)  # ESM-1v context window
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per_device_batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4, help="gradient_accumulation_steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision (recommended on A100)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint dir to resume from")
    parser.add_argument("--val_path", type=str, default=None, help="FASTA/TXT file for validation (optional)")
    parser.add_argument("--eval_steps", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    if hasattr(model, "esm") and hasattr(model.esm, "contact_head"):
        # Either remove it entirely:
        model.esm.contact_head = None
        # (Optional safety) or freeze if you prefer:
        # for p in model.esm.contact_head.parameters():
        #     p.requires_grad = False

        # Memory/perf helpers
    # Enable GC with non-reentrant engine (avoids “mark ready twice”)
    #model.gradient_checkpointing_enable(
    #gradient_checkpointing_kwargs={"use_reentrant": False})
    model.gradient_checkpointing_disable()
    #model.gradient_checkpointing_enable()

    torch.set_float32_matmul_precision("high")

    # Prepare dataset (Arrow on disk; efficient for 1.9M sequences)
    # Train
    train_src = prepare_text_source(Path(args.data_path), args.max_length)
    ds = {"train": str(train_src)}

    # Validation (optional)
    eval_dataset = None
    if args.val_path is not None:
        val_src = prepare_text_source(Path(args.val_path), args.max_length)
        ds["validation"] = str(val_src)

    raw = load_dataset("text", data_files=ds)

    def tok(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length, padding=False)

    tokd = raw.map(tok, batched=True, remove_columns=["text"])
    # Data collator applies 15% masking like ESM pretraining
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        bf16=args.bf16,
        fp16=False if args.bf16 else True,
        # eval + checkpointing
        evaluation_strategy="steps" if "validation" in tokd else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        load_best_model_at_end=True if "validation" in tokd else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        # DDP niceties
        ddp_backend="nccl",
        ddp_find_unused_parameters=True,
        # gradient_checkpointing stays OFF per your test
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokd["train"],
        eval_dataset=tokd.get("validation"),
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # --- resume handling (see section 2 below) ---
    resume_ckpt = args.resume
    if resume_ckpt == "auto":
        resume_ckpt = True  # HF auto-detects last checkpoint in output_dir

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Optional: report eval perplexity at the end
    import math
    if "validation" in tokd:
        metrics = trainer.evaluate()
        ppl = math.exp(metrics["eval_loss"])
        print(f"Final eval loss: {metrics['eval_loss']:.4f} | perplexity: {ppl:.3f}")

    # quick sanity: compute final perplexity on a small held-out slice if you like
    # (left out for brevity; MLM perplexity needs masking logic for eval)

if __name__ == "__main__":
    main()
