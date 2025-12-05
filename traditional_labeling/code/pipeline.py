from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .bm25 import compute_bm25_tops
from .jaccard import compute_jaccard_tops
from .tfidf import compute_tfidf_tops
from .utils import parse_semicolon_ids, read_corpus, to_semicolon_str, write_corpus


def build_overlap_labels(
    input_csv: str,
    output_csv: str,
    tfidf_threshold: float = 0.40,
    jaccard_threshold: float = 0.30,
    bm25_percentile: float = 90.0,
    max_top: int = 12,
) -> None:
    """
    Build 'traditional' neighbor labels based on TF-IDF cosine, Jaccard, and BM25.
    Writes:
      - tf_idf_tops
      - jaccard_tops
      - bm25_tops
      - overlap2
      - overlap3
    """
    df = read_corpus(input_csv, id_col="id", text_col="text")
    df = df.copy()
    ids = df["id"].to_numpy()
    texts = df["text"].astype(str).tolist()
    print(f"[Info] rows: {len(df):,}")

    # 1) TF-IDF
    print("[TF-IDF] computing...")
    df["tf_idf_tops"] = compute_tfidf_tops(texts, ids, threshold=tfidf_threshold, max_top=max_top)

    # 2) Jaccard
    print("[Jaccard] computing...")
    df["jaccard_tops"] = compute_jaccard_tops(texts, ids, threshold=jaccard_threshold, max_top=max_top)

    # 3) BM25
    print("[BM25] computing...")
    df["bm25_tops"] = compute_bm25_tops(texts, ids, percentile=bm25_percentile, max_top=max_top)

    # 4) Overlaps
    print("[Overlap] computing overlap2/overlap3...")
    tf_sets = [set(parse_semicolon_ids(s)) for s in df["tf_idf_tops"].tolist()]
    ja_sets = [set(parse_semicolon_ids(s)) for s in df["jaccard_tops"].tolist()]
    bm_sets = [set(parse_semicolon_ids(s)) for s in df["bm25_tops"].tolist()]
    overlap2_list: List[List[int]] = []
    overlap3_list: List[List[int]] = []
    for a, b, c in zip(tf_sets, ja_sets, bm_sets):
        in2 = (a & b) | (a & c) | (b & c)
        in3 = a & b & c
        overlap2_list.append(sorted(in2))
        overlap3_list.append(sorted(in3))
    df["overlap2"] = [to_semicolon_str(lst) for lst in overlap2_list]
    df["overlap3"] = [to_semicolon_str(lst) for lst in overlap3_list]

    write_corpus(df, output_csv)
    print(f"[Done] Saved: {output_csv}")


