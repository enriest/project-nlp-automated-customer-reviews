#!/usr/bin/env python3
"""
Cache a small summarization model locally for offline runs.
This downloads model weights/tokenizer into offline_models/summarizer/<model_name_slug>/

Usage:
  python scripts/cache_summarization_model.py --model sshleifer/distilbart-cnn-12-6
  python scripts/cache_summarization_model.py --model philschmid/bart-large-cnn-samsum

Then run summarization offline:
  python scripts/summarize_by_category_and_rating.py \
    --input archive/1429_1.csv \
    --top-n 5 \
    --model-name offline_models/summarizer/sshleifer-distilbart-cnn-12-6 \
    --cache-dir offline_models/summarizer/sshleifer-distilbart-cnn-12-6 \
    --offline
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def slugify_model_name(name: str) -> str:
    return name.strip().replace("/", "-")


def main():
    parser = argparse.ArgumentParser(description="Cache a summarization model locally")
    parser.add_argument("--model", type=str, default="sshleifer/distilbart-cnn-12-6", help="HF model id or local path")
    parser.add_argument("--out-dir", type=str, default="offline_models/summarizer", help="Where to store cached model")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception as e:
        print(f"Transformers not available: {e}")
        return 2

    out_dir = Path(args.out_dir) / slugify_model_name(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Caching model '{args.model}' to: {out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(out_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=str(out_dir))

    # Save a local copy for explicit path usage
    tokenizer.save_pretrained(str(out_dir))
    model.save_pretrained(str(out_dir))
    print("[done] Model cached.")


if __name__ == "__main__":
    raise SystemExit(main())
