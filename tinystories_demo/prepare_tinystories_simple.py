#!/usr/bin/env python3
"""Prepare a TinyStories dataset for ELF training.

This script downloads a small subset of the TinyStories corpus, tokenizes each
example with a Hugging Face tokenizer, and saves train/validation splits to
local disk. The output format is compatible with ELF's dataset loader via
`datasets.load_from_disk`.
"""

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TinyStories data for ELF")
    parser.add_argument("--output-dir", default="tinystories_demo/data")
    parser.add_argument("--tokenizer-name", default="t5-small")
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def build_dataset(texts, tokenizer, max_length):
    records = []
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        records.append({"input_ids": token_ids})
    return Dataset.from_list(records)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train")
    raw_dataset = raw_dataset.shuffle(seed=args.seed)

    total_size = min(args.train_size + args.val_size, len(raw_dataset))
    train_size = min(args.train_size, total_size)
    val_size = max(0, total_size - train_size)

    train_texts = [row["text"] for row in raw_dataset.select(range(train_size))]
    val_texts = [row["text"] for row in raw_dataset.select(range(train_size, train_size + val_size))]

    train_ds = build_dataset(train_texts, tokenizer, args.max_length)
    val_ds = build_dataset(val_texts, tokenizer, args.max_length)

    train_path = output_dir / "train"
    val_path = output_dir / "val"
    train_ds.save_to_disk(str(train_path))
    val_ds.save_to_disk(str(val_path))

    print(f"Saved train split to {train_path}")
    print(f"Saved validation split to {val_path}")


if __name__ == "__main__":
    main()
