#!/usr/bin/env python3
"""Download and prepare SlimPajama dataset for DataRater training."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# Avoid torch import (not needed for tokenization)
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from datasets import load_dataset
from transformers import AutoTokenizer


def create_sequences(
    token_ids: list[int],
    seq_len: int,
    stride: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """
    Create input/target sequence pairs from tokenized text.

    Args:
        token_ids: List of token IDs
        seq_len: Target sequence length
        stride: How many tokens to advance between sequences (default: seq_len, no overlap)

    Returns:
        List of (input_ids, target_ids) tuples
    """
    if stride is None:
        stride = seq_len

    sequences = []
    for i in range(0, len(token_ids) - seq_len + 1, stride):
        input_seq = token_ids[i : i + seq_len]
        # Target is input shifted by 1 (next token prediction)
        target_seq = token_ids[i + 1 : i + seq_len + 1]
        # Pad target if needed (shouldn't happen, but safety check)
        if len(target_seq) < seq_len:
            target_seq = target_seq + [token_ids[-1]] * (seq_len - len(target_seq))
        sequences.append((input_seq, target_seq))
    return sequences


def process_dataset(
    dataset_name: str,
    output_dir: Path,
    seq_len_lm: int,
    seq_len_rater: int,
    train_docs: int,
    val_docs: int,
    seed: int = 42,
) -> None:
    """
    Download SlimPajama, tokenize, and write to NDJSON format.

    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Directory to write JSONL files
        seq_len_lm: Sequence length for language model
        seq_len_rater: Sequence length for rater (not used in output, but kept for reference)
        train_docs: Number of documents to use for training
        val_docs: Number of documents to use for validation
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    print(f"Loading tokenizer: gpt2")
    # Use fast tokenizer (pure Rust, no torch dependency)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_a"
    val_path = output_dir / "val"
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    train_file = train_path / "part_0000.jsonl"
    val_file = val_path / "part_0000.jsonl"

    train_count = 0
    val_count = 0
    doc_count = 0
    max_train_sequences = train_docs * 20  # Rough estimate: ~20 sequences per doc
    max_val_sequences = val_docs * 20

    print(f"Processing documents and creating sequences...")
    print(f"Target: ~{max_train_sequences} train sequences, ~{max_val_sequences} val sequences")
    with open(train_file, "w") as train_f, open(val_file, "w") as val_f:
        for example in dataset:
            if train_count >= max_train_sequences and val_count >= max_val_sequences:
                break

            text = example.get("text", "")
            if not text or len(text.strip()) < 100:  # Skip very short texts
                continue

            # Tokenize
            tokens = tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            if len(tokens) < seq_len_lm:
                continue  # Skip sequences shorter than required length

            # Create sequences (non-overlapping windows)
            sequences = create_sequences(tokens, seq_len_lm, stride=seq_len_lm)

            # Determine split (use first train_docs for train, rest for val)
            is_train = doc_count < train_docs

            # Write sequences
            for input_ids, target_ids in sequences:
                record = {
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                }
                json_line = json.dumps(record) + "\n"

                if is_train:
                    if train_count >= max_train_sequences:
                        continue
                    train_f.write(json_line)
                    train_count += 1
                else:
                    if val_count >= max_val_sequences:
                        continue
                    val_f.write(json_line)
                    val_count += 1

            doc_count += 1

            if doc_count % 1000 == 0:
                print(
                    f"Processed {doc_count} documents, "
                    f"{train_count} train sequences, "
                    f"{val_count} val sequences"
                )

    print(f"\nDone!")
    print(f"Train sequences: {train_count}")
    print(f"Val sequences: {val_count}")
    print(f"Train file: {train_file}")
    print(f"Val file: {val_file}")
    print(f"\nUpdate config.yaml with:")
    print(f"  data.train_a_path: '{train_path}/*.jsonl'")
    print(f"  data.val_path: '{val_path}/*.jsonl'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare SlimPajama dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cerebras/SlimPajama-627B",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--seq-len-lm",
        type=int,
        default=1024,
        help="Sequence length for language model",
    )
    parser.add_argument(
        "--seq-len-rater",
        type=int,
        default=256,
        help="Sequence length for rater (reference only)",
    )
    parser.add_argument(
        "--train-docs",
        type=int,
        default=10000,
        help="Number of documents for training",
    )
    parser.add_argument(
        "--val-docs",
        type=int,
        default=1000,
        help="Number of documents for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        seq_len_lm=args.seq_len_lm,
        seq_len_rater=args.seq_len_rater,
        train_docs=args.train_docs,
        val_docs=args.val_docs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

