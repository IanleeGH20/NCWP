from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .embeddings import HFMeanPoolingEncoder, NemotronEncoder
from .utils import (
    compute_neighbor_ids,
    parse_semicolon_ids,
    pick_device,
    read_corpus,
    write_corpus,
)


LLAMA_COL = "nvidia/llama-embed-nemotron-8b"
QWEN4_COL = "qwen3-4b"
QWEN8_COL = "qwen3-8b"


def _build_encoder(model_key: str, device: str, max_len: int) -> tuple[str, object]:
    """
    Returns (column_name, encoder_instance)
    """
    if model_key == "llama":
        return LLAMA_COL, NemotronEncoder("nvidia/llama-embed-nemotron-8b", device=device, max_len=max_len)
    if model_key == "qwen4b":
        return QWEN4_COL, HFMeanPoolingEncoder("Qwen/Qwen3-Embedding-4B", device=device, max_len=max_len)
    if model_key == "qwen8b":
        return QWEN8_COL, HFMeanPoolingEncoder("Qwen/Qwen3-Embedding-8B", device=device, max_len=max_len)
    raise ValueError(f"Unknown model key: {model_key}")


def _safe_encode_texts(encoder, texts: List[str]) -> np.ndarray:
    try:
        return encoder.embed(texts)
    except Exception as e:
        # As a last resort, return empty embeddings to avoid hard failure.
        print(f"[WARN] Embedding failed for {type(encoder).__name__}: {e}")
        return np.zeros((len(texts), 0), dtype=np.float32)


def _add_neighbors_columns(
    df: pd.DataFrame,
    ids: np.ndarray,
    texts: List[str],
    models: List[str],
    device: str,
    max_len: int,
    sim_threshold: float,
) -> pd.DataFrame:
    for key in models:
        col_name, encoder = _build_encoder(key, device=device, max_len=max_len)
        print(f"[Embedding] {key} -> column '{col_name}' (device={device})")
        vecs = _safe_encode_texts(encoder, texts)
        print(f"  embeddings shape: {vecs.shape}")
        print(f"[Neighbors] {col_name} (thr={sim_threshold})")
        df[col_name] = compute_neighbor_ids(vecs, ids, sim_threshold)
    return df


def _add_neighbors_union(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    def _row_union(row) -> str:
        s = set()
        for c in target_cols:
            if c in row:
                s.update(parse_semicolon_ids(row[c]))
        if not s:
            return ""
        return ";".join(str(x) for x in sorted(s))

    print(f"[Union] Building neighbors union from: {target_cols}")
    df["neighbors_union"] = df.apply(_row_union, axis=1)
    df["neighbors_union_count"] = df["neighbors_union"].apply(
        lambda x: 0 if not x else len([t for t in str(x).split(";") if t != ""])
    )
    return df


def run_pipeline(
    input_csv: str,
    output_csv: str,
    sim_threshold: float = 0.80,
    batch_size: int = 64,  # kept for future parity; current encoders use fixed batch size internally
    prefer_cuda: bool = False,
    max_len: int = 512,
    models: List[str] | None = None,
) -> None:
    """
    Main pipeline entry point.
    - Reads input CSV (expects 'id', 'text')
    - Runs embedding + neighbor generation for requested models
    - Builds neighbors_union and neighbors_union_count
    - Writes output CSV
    """
    if models is None:
        models = ["llama", "qwen4b", "qwen8b"]

    device = pick_device(prefer_cuda=prefer_cuda, min_free_gb=2.0)
    print(f"[Info] device: {device}")

    df = read_corpus(input_csv, id_col="id", text_col="text")
    ids = df["id"].to_numpy()
    texts = df["text"].astype(str).tolist()
    print(f"[Info] rows: {len(df):,}")

    # Step 1: add per-model neighbor columns
    df = _add_neighbors_columns(
        df=df,
        ids=ids,
        texts=texts,
        models=models,
        device=device,
        max_len=max_len,
        sim_threshold=sim_threshold,
    )

    # Step 2: build neighbors_union
    available_cols = [c for c in [LLAMA_COL, QWEN4_COL, QWEN8_COL] if c in df.columns]
    df = _add_neighbors_union(df, target_cols=available_cols)

    # Step 3: write output
    write_corpus(df, output_csv)
    print(f"[Done] Saved labeled corpus to: {output_csv}")


