#!/usr/bin/env python3
"""
Summarize reviews by product category and star rating (1-5).

Features
- Loads one or more CSV files with Datafiniti-like schema.
- Parses product categories from the `categories` column (comma-separated).
- Groups reviews by category and by `reviews.rating` (1..5).
- Generates a concise summary per rating using a Generative AI model (HF transformers)
  with a robust extractive TF‑IDF fallback when a local model isn't available.
- Supports limiting to top-N categories (by review count) to keep runs manageable.
- Writes JSON outputs under outputs/summaries/category_slug.json

Usage examples
  python scripts/summarize_by_category_and_rating.py \
    --input archive/1429_1.csv \
    --top-n 5 \
    --max-reviews-per-rating 200 \
    --model-name sshleifer/distilbart-cnn-12-6

  python scripts/summarize_by_category_and_rating.py --input archive --top-n 10 --use-extractive

Notes
- If the summarization model can't be loaded (no internet/cache), the script falls back to
  an extractive summarizer based on TF‑IDF sentence scoring. This ensures the script always runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_INPUT_GLOB = "archive/**/*.csv"
DEFAULT_OUTPUT_DIR = "outputs/summaries"
DEFAULT_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # small-ish summarization model


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "misc"


def ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_input_files(input_path: str | Path) -> List[Path]:
    p = Path(input_path)
    if p.is_dir():
        return sorted(p.glob("**/*.csv"))
    elif p.is_file() and p.suffix.lower() == ".csv":
        return [p]
    else:
        # allow glob patterns
        return [Path(x) for x in Path().glob(str(p))]


def read_reviews(files: List[Path], sample_rows: Optional[int] = None) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=sample_rows, low_memory=False)
        except Exception:
            # Fall back to default read
            df = pd.read_csv(f, low_memory=False)
        df["__source_file"] = str(f)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No CSV files found to load.")
    return pd.concat(frames, ignore_index=True)


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Expected columns (based on Datafiniti):
    # - categories: comma-separated string
    # - reviews.text: text
    # - reviews.rating: numeric 1..5
    # Some datasets may use alternative names; try to adapt.
    col_map_candidates = {
        "text": ["reviews.text", "reviewText", "text", "content"],
        "rating": ["reviews.rating", "rating", "stars", "overall"],
        "categories": ["categories", "category", "product_category"],
        "title": ["reviews.title", "summary", "title"],
    }

    def pick(colnames: List[str]) -> Optional[str]:
        for name in colnames:
            if name in df.columns:
                return name
        return None

    text_col = pick(col_map_candidates["text"]) or "reviews.text"
    rating_col = pick(col_map_candidates["rating"]) or "reviews.rating"
    categories_col = pick(col_map_candidates["categories"]) or "categories"
    title_col = pick(col_map_candidates["title"])  # optional

    # Filter to rows with text and rating
    if text_col not in df.columns or rating_col not in df.columns:
        missing = [c for c in [text_col, rating_col] if c not in df.columns]
        raise KeyError(f"Missing required columns: {missing}")

    work = df[[c for c in [text_col, rating_col, categories_col, title_col] if c in df.columns]].copy()
    work.rename(columns={
        text_col: "text",
        rating_col: "rating",
        categories_col: "categories" if categories_col in work.columns else "categories",
        **({title_col: "title"} if title_col in work.columns else {}),
    }, inplace=True)

    # Clean types
    work["text"] = work["text"].astype(str).str.strip()
    work = work[work["text"].str.len() > 0]

    # Ratings: coerce to integers 1..5
    work["rating"] = pd.to_numeric(work["rating"], errors="coerce").round().astype("Int64")
    work = work[work["rating"].between(1, 5, inclusive="both")]

    # Categories: split comma-separated, keep top-level tokens
    if "categories" in work.columns:
        work["categories"] = work["categories"].fillna("").astype(str)
    else:
        work["categories"] = "Unknown"

    return work.reset_index(drop=True)


def explode_categories(df: pd.DataFrame) -> pd.DataFrame:
    # Split categories by comma and trim whitespace
    cats = df["categories"].fillna("").astype(str).apply(lambda s: [c.strip() for c in s.split(",") if c.strip()])
    df = df.copy()
    df["__categories_list"] = cats
    df = df.explode("__categories_list")
    df.rename(columns={"__categories_list": "category"}, inplace=True)
    df["category"] = df["category"].fillna("Unknown").replace("", "Unknown")
    return df


def pick_top_categories(df: pd.DataFrame, top_n: int, min_reviews: int = 1) -> List[str]:
    counts = df.groupby("category").size().sort_values(ascending=False)
    counts = counts[counts >= min_reviews]
    return list(counts.head(top_n).index)


def chunk_text(sentences: List[str], max_chars: int) -> List[str]:
    chunks, cur = [], []
    total = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if total + len(s) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            cur, total = [s], len(s)
        else:
            cur.append(s)
            total += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def simple_sentence_split(text: str) -> List[str]:
    # Lightweight splitter to avoid heavy dependencies
    # Splits on period/question/exclamation while keeping abbreviations minimally intact
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extractive_summary_tfidf(texts: List[str], max_sentences: int = 6) -> str:
    """Select top sentences using TF‑IDF against the centroid."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Build sentences corpus
    all_sentences: List[str] = []
    for t in texts:
        all_sentences.extend(simple_sentence_split(t))
    # Deduplicate and keep reasonable length
    seen = set()
    sentences = []
    for s in all_sentences:
        s_norm = s.lower()
        if 30 <= len(s) <= 400 and s_norm not in seen:
            seen.add(s_norm)
            sentences.append(s)
    if not sentences:
        return ""

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    X = vectorizer.fit_transform(sentences)
    centroid = np.asarray(X.mean(axis=0))
    scores = cosine_similarity(X, centroid).ravel()
    top_idx = scores.argsort()[::-1][:max_sentences]
    # Keep original order for readability
    top_idx_sorted = sorted(top_idx)
    summary = " ".join([sentences[i] for i in top_idx_sorted])
    return summary


@dataclass
class Summarizer:
    model_name: str = DEFAULT_MODEL_NAME
    device: Optional[int] = None
    use_extractive_only: bool = False
    cache_dir: Optional[str] = None
    offline: bool = False

    def __post_init__(self):
        self._pipe = None
        if self.use_extractive_only:
            return
        try:
            # Lazy import to avoid heavy deps if not used
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

            if self.offline:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            tok_kwargs = {}
            mod_kwargs = {}
            if self.cache_dir:
                tok_kwargs["cache_dir"] = self.cache_dir
                mod_kwargs["cache_dir"] = self.cache_dir

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **mod_kwargs)

            pipe_kwargs = {"task": "summarization", "model": model, "tokenizer": tokenizer}
            if self.device is not None:
                pipe_kwargs["device"] = self.device
            self._pipe = pipeline(**pipe_kwargs)
        except Exception as e:
            sys.stderr.write(f"[warn] Failed to load model '{self.model_name}' ({e}). Falling back to extractive.\n")
            self._pipe = None

    def summarize(self, texts: List[str], target_chars: int = 512) -> str:
        # If no texts, return empty
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return ""

        # Try abstractive with chunking if available
        if self._pipe is not None:
            # Conservative chunking (HF models have token limits ~1024-2048)
            sentences = []
            for t in texts:
                sentences.extend(simple_sentence_split(t))
            chunks = chunk_text(sentences, max_chars=3500)
            partial_summaries = []
            for ch in chunks:
                try:
                    out = self._pipe(ch, max_length=220, min_length=60, do_sample=False)
                    if isinstance(out, list) and out and "summary_text" in out[0]:
                        partial_summaries.append(out[0]["summary_text"])
                    else:
                        # Unexpected output shape
                        partial_summaries.append(ch[:target_chars])
                except Exception:
                    partial_summaries.append(ch[:target_chars])
            if len(partial_summaries) == 1:
                return partial_summaries[0]
            # Final pass to unify
            try:
                combined = " \n".join(partial_summaries)
                out = self._pipe(combined, max_length=240, min_length=80, do_sample=False)
                if isinstance(out, list) and out and "summary_text" in out[0]:
                    return out[0]["summary_text"]
            except Exception:
                pass
            # Fallback to extractive on partials
            return extractive_summary_tfidf(partial_summaries, max_sentences=6)

        # Extractive fallback
        return extractive_summary_tfidf(texts, max_sentences=6)


def build_rating_summaries(
    df: pd.DataFrame,
    category: str,
    summarizer: Summarizer,
    max_reviews_per_rating: int = 500,
) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    dcat = df[df["category"] == category]
    for rating in range(1, 6):
        dr = dcat[dcat["rating"] == rating]
        count_total = int(dr.shape[0])
        if count_total == 0:
            result[str(rating)] = {
                "count": 0,
                "summary": "",
                "sampled": 0,
            }
            continue
        # Sample to cap runtime
        if max_reviews_per_rating and count_total > max_reviews_per_rating:
            dr = dr.sample(n=max_reviews_per_rating, random_state=42)
        texts = dr["text"].astype(str).tolist()
        summary = summarizer.summarize(texts)
        result[str(rating)] = {
            "count": count_total,
            "summary": summary,
            "sampled": int(dr.shape[0]),
        }
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize reviews by category and star rating.")
    parser.add_argument("--input", type=str, default="archive", help="Input CSV file, folder, or glob pattern")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Where to write JSON summaries")
    parser.add_argument("--top-n", type=int, default=10, help="Process at most top-N categories by review count")
    parser.add_argument("--min-reviews", type=int, default=20, help="Min reviews per category to include")
    parser.add_argument("--max-reviews-per-rating", type=int, default=400, help="Cap reviews per rating bucket")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="HF summarization model name")
    parser.add_argument("--device", type=int, default=None, help="Device index for HF pipeline (e.g., 0 for GPU)")
    parser.add_argument("--use-extractive", action="store_true", help="Force extractive TF‑IDF summarizer")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache directory or local model dir")
    parser.add_argument("--offline", action="store_true", help="Run transformers in offline mode (no network)")
    parser.add_argument("--sample-rows", type=int, default=None, help="Read only first N rows per file (debug)")
    parser.add_argument("--category", type=str, default=None, help="Summarize only this exact category (case-insensitive or slug)")
    parser.add_argument("--category-contains", type=str, default=None, help="Summarize categories whose name contains this substring (case-insensitive)")

    args = parser.parse_args(argv)

    out_dir = ensure_output_dir(args.output_dir)

    files = find_input_files(args.input)
    if not files:
        print(f"No input files found for: {args.input}", file=sys.stderr)
        return 2

    print(f"[info] Loading {len(files)} file(s)...")
    df_raw = read_reviews(files, sample_rows=args.sample_rows)
    print(f"[info] Loaded {len(df_raw):,} rows.")

    df = normalize_schema(df_raw)
    df = explode_categories(df)
    print(f"[info] Expanded to {len(df):,} rows across categories.")

    top_categories = pick_top_categories(df, top_n=args.top_n, min_reviews=args.min_reviews)
    # If category filters provided, narrow down
    if args.category:
        target = args.category.strip().lower()
        top_categories = [c for c in df["category"].unique().tolist() if (c.lower() == target or slugify(c) == slugify(target))]
    elif args.category_contains:
        sub = args.category_contains.strip().lower()
        candidates = [c for c in df["category"].unique().tolist() if sub in c.lower()]
        # preserve order by frequency
        freq_order = pick_top_categories(df[df["category"].isin(candidates)], top_n=len(candidates), min_reviews=args.min_reviews)
        top_categories = freq_order[: args.top_n]
    print(f"[info] Selected top {len(top_categories)} categories: {top_categories[:10]}{'...' if len(top_categories)>10 else ''}")

    summarizer = Summarizer(
        model_name=args.model_name,
        device=args.device,
        use_extractive_only=bool(args.use_extractive),
        cache_dir=args.cache_dir,
        offline=bool(args.offline),
    )

    overall_index: List[Dict[str, object]] = []

    for cat in top_categories:
        print(f"[info] Summarizing category: {cat}")
        data = build_rating_summaries(
            df=df,
            category=cat,
            summarizer=summarizer,
            max_reviews_per_rating=args.max_reviews_per_rating,
        )
        payload = {
            "category": cat,
            "category_slug": slugify(cat),
            "total_reviews": int(df[df["category"] == cat].shape[0]),
            "ratings": data,
        }
        fname = out_dir / f"{slugify(cat)}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        overall_index.append({
            "category": cat,
            "category_slug": slugify(cat),
            "file": str(fname),
            "total_reviews": payload["total_reviews"],
        })

    # Write an index file for convenience
    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"categories": overall_index}, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote {len(overall_index)} category summaries to: {out_dir}")
    print(f"[done] Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
