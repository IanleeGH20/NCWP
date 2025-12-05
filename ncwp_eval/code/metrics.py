from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def build_pos_sets_for_split(df: pd.DataFrame, split_ids: List[int], which: str) -> List[Set[int]]:
    assert "id" in df.columns and which in df.columns
    id_to_local = {int(rid): i for i, rid in enumerate([int(x) for x in split_ids])}

    def parse_ids(cell) -> List[int]:
        if pd.isna(cell):
            return []
        s = str(cell).replace(";", ",")
        out = []
        for x in s.split(","):
            x = x.strip()
            if x.isdigit():
                out.append(int(x))
        return out

    rid_to_pos = {}
    for _, row in df[["id", which]].iterrows():
        rid = int(row["id"])
        rid_to_pos[rid] = parse_ids(row[which])

    pos_sets: List[Set[int]] = []
    for rid in split_ids:
        pos_ids = rid_to_pos.get(int(rid), [])
        local_pos = set()
        for pr in pos_ids:
            if pr == rid:
                continue
            if pr in id_to_local:
                local_pos.add(int(id_to_local[pr]))
        pos_sets.append(local_pos)
    return pos_sets


@torch.no_grad()
def metrics_for_pos_sets(emb: np.ndarray, pos_sets: List[Set[int]], ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    """
    Compute retrieval metrics (mAP, nDCG@maxK) on positive sets.
    """
    X = torch.tensor(emb, dtype=torch.float32)
    X = F.normalize(X, dim=-1)
    S = (X @ X.t()).cpu().numpy()
    N = S.shape[0]
    ap_values: List[float] = []
    ndcg_values: List[float] = []
    # compute per-query ranking
    for i in range(N):
        row = S[i].copy()
        row[i] = -np.inf
        order = np.argsort(-row)
        pos = pos_sets[i]
        if not pos:
            continue
        # AP
        hits = 0
        total = 0
        ap_sum = 0.0
        for rank, j in enumerate(order, start=1):
            if j in pos:
                hits += 1
                ap_sum += hits / rank
            total += 1
        ap = ap_sum / max(1, len(pos))
        ap_values.append(ap)
        # nDCG@maxK
        K = max(ks)
        gains = [1.0 if order[r] in pos else 0.0 for r in range(min(K, len(order)))]
        dcg = sum([g / np.log2(r + 2) for r, g in enumerate(gains)])
        ideal_gains = [1.0] * min(K, len(pos))
        idcg = sum([g / np.log2(r + 2) for r, g in enumerate(ideal_gains)])
        ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)
    return {
        "mAP": float(np.mean(ap_values)) if ap_values else 0.0,
        "nDCG@maxK": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
    }


