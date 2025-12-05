from __future__ import annotations

import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def read_corpus(input_csv: str, id_col: str = "id", text_col: str = "text") -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"CSV must have columns '{id_col}' and '{text_col}'")
    return df


def write_corpus(df: pd.DataFrame, output_csv: str) -> None:
    ensure_parent_dir(output_csv)
    df.to_csv(output_csv, index=False)


def pick_device(prefer_cuda: bool = False, min_free_gb: float = 2.0) -> str:
    if prefer_cuda and torch.cuda.is_available():
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            if free_gb >= min_free_gb:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def normalize_rows_np(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return x / norms


def compute_neighbor_ids(embeddings: np.ndarray, ids: np.ndarray, threshold: float) -> List[str]:
    """
    Compute semicolon-separated neighbor ID strings for each row where cosine >= threshold.
    embeddings: 2D numpy array [N, D], assumed L2-normalized (will normalize if not).
    ids: 1D numpy array of length N.
    """
    embs = np.asarray(embeddings)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    if embs.ndim == 2 and embs.shape[0] != len(ids) and embs.shape[1] == len(ids):
        embs = embs.T
    if embs.ndim != 2 or embs.shape[0] != len(ids):
        raise ValueError(f"Embeddings shape mismatch. ids={len(ids)}, embeddings={embs.shape}")

    embs = normalize_rows_np(embs)
    sim = embs @ embs.T  # [N, N]
    np.fill_diagonal(sim, -np.inf)

    out: List[str] = []
    for i in range(sim.shape[0]):
        row = sim[i]
        mask = row >= threshold
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            out.append("")
            continue
        order = np.argsort(-row[idxs])
        sorted_idxs = idxs[order]
        sorted_ids = [str(ids[j]) for j in sorted_idxs]
        out.append(";".join(sorted_ids))
    return out


def parse_semicolon_ids(cell: str | float | int | None) -> List[int]:
    if cell is None:
        return []
    if isinstance(cell, float) and np.isnan(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    out: List[int] = []
    for t in s.split(";"):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(int(float(t)))
        except Exception:
            continue
    return out


