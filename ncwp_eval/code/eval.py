from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from decoder_lm.code.data import SimpleVocab
from decoder_lm.code.model import GPT, GPTConfig
from .metrics import build_pos_sets_for_split, metrics_for_pos_sets
from .projector import (
    fit_pca_whitening,
    transform_pca_whitening,
    transform_pca_soft_whitening,
    transform_pca,
    l2_normalize_np,
    fit_new_proposal_projector,
    transform_new_proposal_projector,
    fit_lpp_projector,
    transform_linear_projector,
    fit_random_projector,
)
# external embedder methods removed for this experiment set


class TextDataset(Dataset):
    def __init__(self, texts: List[str], vocab: SimpleVocab, block_size: int):
        self.samples = []
        for s in texts:
            ids = vocab.encode(s)
            if not ids:
                ids = [vocab.pad_id]
            x = torch.full((block_size,), vocab.pad_id, dtype=torch.long)
            attn = torch.zeros((block_size,), dtype=torch.bool)
            L = min(len(ids), block_size)
            x[:L] = torch.tensor(ids[:L], dtype=torch.long)
            attn[:L] = True
            self.samples.append((x, attn))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _find_weights_dir_and_paths(weights_dir: str, base_dim: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to find a directory that contains both:
      - vocab.json
      - base_dim_{base_dim}.pt  (preferred) or base_dim_{base_dim}_last.pt (fallback)
    Search order:
      1) weights_dir itself
      2) recursively search subdirectories of weights_dir
    Returns (vocab_path, weight_path) or (None, None) if not found.
    """
    candidates = [f"base_dim_{base_dim}.pt", f"base_dim_{base_dim}_last.pt"]

    # Check given dir first
    vocab_path = os.path.join(weights_dir, "vocab.json")
    for wname in candidates:
        weight_path = os.path.join(weights_dir, wname)
        if os.path.exists(vocab_path) and os.path.exists(weight_path):
            return vocab_path, weight_path

    # Walk subdirs
    for root, dirs, files in os.walk(weights_dir):
        if "vocab.json" in files:
            for wname in candidates:
                if wname in files:
                    return os.path.join(root, "vocab.json"), os.path.join(root, wname)
    return None, None


def _derive_base_id(weight_path: str) -> str:
    """
    Build a concise identifier from a base weight filename/path.
    Examples:
      .../base_dim_512.pt           -> dim512
      .../base_dim_512_last.pt      -> dim512_last
      .../base_dim_512_step_200.pt  -> dim512_step200
    """
    name = os.path.basename(weight_path)
    m = re.match(r"base_dim_(\d+)(?:_(last|step_\d+))?\.pt$", name)
    if m:
        bd = m.group(1)
        tag = m.group(2)
        if tag is None:
            return f"dim{bd}"
        if tag == "last":
            return f"dim{bd}_last"
        if tag.startswith("step_"):
            step = tag.split("_", 1)[1]
            return f"dim{bd}_step{step}"
        return f"dim{bd}"
    # fallback
    return os.path.splitext(name)[0]


def _list_all_base_weights(weights_root: str) -> Dict[int, List[str]]:
    """
    Recursively list all base_dim_* weight files grouped by base_dim, only when vocab.json exists in same dir.
    Returns: { base_dim: [weight_paths...] }
    """
    found: Dict[int, List[str]] = {}
    for root, dirs, files in os.walk(weights_root):
        if "vocab.json" not in files:
            continue
        for fn in files:
            m = re.match(r"base_dim_(\d+)(?:_(?:last|step_\d+))?\.pt$", fn)
            if not m:
                continue
            bd = int(m.group(1))
            path = os.path.join(root, fn)
            found.setdefault(bd, []).append(path)
    # sort each list by preference: best -> last -> step ascending
    for bd, lst in found.items():
        def key(p: str):
            n = os.path.basename(p)
            if "_step_" in n:
                try:
                    step = int(n.split("_step_")[1].split(".")[0])
                except Exception:
                    step = 0
                return (2, step)
            if "_last" in n:
                return (1, 0)
            return (0, 0)  # best
        lst.sort(key=key)
    return found


def load_base_model(weights_dir: str, base_dim: int, block_size: int, device: str, weight_path: Optional[str] = None) -> Tuple[GPT, SimpleVocab, str, str]:
    if weight_path is None:
        vocab_path, weight_path = _find_weights_dir_and_paths(weights_dir, base_dim)
    else:
        vocab_path = os.path.join(os.path.dirname(weight_path), "vocab.json")
    if vocab_path is None or weight_path is None:
        raise FileNotFoundError(
            f"Missing vocab.json or base weights for base_dim={base_dim} under '{weights_dir}'. "
            f"Expected files like 'vocab.json' and 'base_dim_{base_dim}.pt' "
            f"(or 'base_dim_{base_dim}_last.pt'). "
            f"Make sure to train decoder_lm first and/or set the correct tokenizer subdir (e.g., weights/ws, weights/bpe, weights/wordpiece)."
        )
    vocab = SimpleVocab.load(vocab_path)
    cfg = GPTConfig(vocab_size=vocab.vocab_size, n_embd=base_dim, n_head=max(1, base_dim // 64), n_layer=4, block_size=block_size)
    model = GPT(cfg).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, vocab, vocab_path, weight_path


@torch.no_grad()
def encode_corpus(model: GPT, vocab: SimpleVocab, texts: List[str], block_size: int, device: str) -> np.ndarray:
    ds = TextDataset(texts, vocab, block_size=block_size)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    embs: List[np.ndarray] = []
    for xb, attn in dl:
        xb = xb.to(device)
        attn = attn.to(device)
        e = model.sentence_embeddings(xb, attn_mask=attn)  # [B, C]
        embs.append(e.cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def run_ncwp_evaluation(
    corpus_csv: str,
    weights_dir: str,
    base_dim: int = 512,
    dims: List[int] | None = None,
    device: str = "cpu",
    block_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
    weight_path: Optional[str] = None,
    save_projectors: bool = False,
    projector_root: Optional[str] = None,
    checkpoint_label: Optional[str] = None,
    progress: bool = False,
    progress_bar: bool = False,
) -> pd.DataFrame:
    """
    - Load base LM and encode corpus (base_dim)
    - Split into train/test (train: fit transforms/projectors, test: evaluate)
    - Evaluate PCA-Whitening, BERT-Whitening, SimCSE, BERT-flow, NCWP(new_proposal), Base
    """
    df = pd.read_csv(corpus_csv)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("corpus CSV must have at least 'id' and 'text'")
    ids_all = df["id"].astype(int).tolist()
    texts_all = df["text"].astype(str).tolist()

    # Split train/test deterministically
    N = len(texts_all)
    idx_all = list(range(N))
    import random as _rnd
    _rnd.seed(seed)
    _rnd.shuffle(idx_all)
    n_val = max(1, int(round(N * val_ratio)))
    test_idx = sorted(idx_all[:n_val])
    train_idx = sorted(idx_all[n_val:])
    ids_test = [ids_all[i] for i in test_idx]
    texts_train = [texts_all[i] for i in train_idx]
    texts_test = [texts_all[i] for i in test_idx]

    if progress and not progress_bar:
        print(f"[NCWP] Loading base model (base_dim={base_dim}) from {weights_dir} ...")
    model, vocab, vocab_path, weight_path_used = load_base_model(weights_dir, base_dim=base_dim, block_size=block_size, device=device, weight_path=weight_path)
    if progress and not progress_bar:
        print(f"[NCWP] Encoding corpus with base model (N={len(texts_all)}) ...")
    E_all = encode_corpus(model, vocab, texts_all, block_size=block_size, device=device)  # [N, base_dim]
    E_all = l2_normalize_np(E_all)
    E_tr = E_all[train_idx]
    E_te = E_all[test_idx]

    # choose GT columns available (include optional overlap/U2/U3 if present)
    candidate_cols = ["nvidia/llama-embed-nemotron-8b", "qwen3-4b", "qwen3-8b", "neighbors_union", "U2", "U3", "overlap2", "overlap3"]
    GT_COLS = [c for c in candidate_cols if c in df.columns]
    if not GT_COLS:
        raise ValueError("No GT columns found. Expected one of llama/qwen3-4b/qwen3-8b/neighbors_union/U2/U3/overlap2/overlap3.")

    # Build pos sets using TEST split only
    pos_sets: Dict[str, List[set[int]]] = {}
    for col in GT_COLS:
        pos_sets[col] = build_pos_sets_for_split(df, ids_test, col)

    if dims is None:
        dims = [2, 4, 8, 16, 32, 64, 128, 256]
    dims = [d for d in dims if d <= base_dim]

    # Fit PCA once on TRAIN (SVD). Reuse for PCA/BERT-Whitening/NCWP initial components.
    if progress and not progress_bar:
        print(f"[NCWP] Fitting PCA whitening on TRAIN (size={E_tr.shape[0]}) ...")
    mu, comps, scales = fit_pca_whitening(E_tr, n_components=min(base_dim, E_tr.shape[1]))

    rows = []
    base_id = _derive_base_id(weight_path_used)
    proj_root = projector_root
    if proj_root is None:
        proj_root = os.path.join(os.path.dirname(os.path.abspath(corpus_csv)), "projectors")
    def _save_npz(method: str, k: int, payload: Dict[str, np.ndarray], extra: Dict[str, object] | None = None):
        if not save_projectors:
            return
        folder_id = (checkpoint_label or base_id)
        out_dir = os.path.join(proj_root, folder_id, method)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"k_{k}.npz")
        meta = {
            "base_dim": int(base_dim),
            "weight_path": str(weight_path_used),
            "vocab_path": str(vocab_path),
            "checkpoint": checkpoint_label or base_id,
            "method": method,
            "k": int(k),
        }
        if extra:
            meta.update(extra)
        try:
            np.savez(out_path, **payload, __meta__=json.dumps(meta))
        except Exception:
            pass
    # Base (no projection) — single row with away_dim=base_dim
    base_metrics: Dict[str, Dict[str, float]] = {}
    for col in GT_COLS:
        base_metrics[col] = metrics_for_pos_sets(E_te, pos_sets[col], ks=(1, 5, 10))
    base_row = {
        "method": "Base",
        "base_dim": base_dim,
        "away_dim": base_dim,
    }
    base_row["base_weight"] = os.path.basename(weight_path_used)
    if checkpoint_label:
        base_row["checkpoint"] = checkpoint_label
    for col in GT_COLS:
        m = base_metrics[col]
        for mname, mval in m.items():
            base_row[f"{col}_{mname}"] = mval
    rows.append(base_row)

    # PCA-Whitening
    for k in dims:
        Z = transform_pca_whitening(E_te, mu, comps, scales, k)
        row = {"method": "PCA-Whitening", "base_dim": base_dim, "away_dim": k}
        row["base_weight"] = os.path.basename(weight_path_used)
        if checkpoint_label:
            row["checkpoint"] = checkpoint_label
        for col in GT_COLS:
            m = metrics_for_pos_sets(Z, pos_sets[col], ks=(1, 5, 10))
            for mname, mval in m.items():
                row[f"{col}_{mname}"] = mval
        rows.append(row)
        _save_npz("PCA-Whitening", k, {"mean": mu, "components": comps[:, :k], "scales": scales[:k]})

    # Soft-Whitening (shrinked whitening on PCA scales)
    for k in dims:
        Z = transform_pca_soft_whitening(E_te, mu, comps, scales, k, alpha=0.10)
        row = {"method": "Soft-Whitening", "base_dim": base_dim, "away_dim": k}
        row["base_weight"] = os.path.basename(weight_path_used)
        if checkpoint_label:
            row["checkpoint"] = checkpoint_label
        for col in GT_COLS:
            m = metrics_for_pos_sets(Z, pos_sets[col], ks=(1, 5, 10))
            for mname, mval in m.items():
                row[f"{col}_{mname}"] = mval
        rows.append(row)
        _save_npz("Soft-Whitening", k, {"mean": mu, "components": comps[:, :k], "scales": scales[:k]}, extra={"alpha": 0.10})

    # NCWP = 'new_proposal' 방식: TRAIN에서 학습, TEST로 적용
    for k in dims:
        if progress and not progress_bar:
            print(f"[NCWP] Training NCWP projector for k={k} ...")
        W_k, mu_in_k, mu_out_k, std_out_k = fit_new_proposal_projector(E_tr, mean=mu, components=comps, k=k, verbose=(progress and not progress_bar), use_tqdm=progress_bar)
        Z = transform_new_proposal_projector(E_te, W=W_k, mu_in=mu_in_k, mu_out=mu_out_k, std_out=std_out_k)
        row = {"method": "NCWP", "base_dim": base_dim, "away_dim": k}
        row["base_weight"] = os.path.basename(weight_path_used)
        if checkpoint_label:
            row["checkpoint"] = checkpoint_label
        for col in GT_COLS:
            m = metrics_for_pos_sets(Z, pos_sets[col], ks=(1, 5, 10))
            for mname, mval in m.items():
                row[f"{col}_{mname}"] = mval
        rows.append(row)
        _save_npz("NCWP", k, {"W": W_k, "mu_in": mu_in_k, "mu_out": mu_out_k, "std_out": std_out_k})

    # Locality Preserving Projections (LPP)
    for k in dims:
        W_lpp, mu_lpp = fit_lpp_projector(E_tr, k=k, n_neighbors=10, reg=1e-3, metric="cosine")
        Z = transform_linear_projector(E_te, W_lpp, mu_lpp)
        row = {"method": "LPP", "base_dim": base_dim, "away_dim": k}
        row["base_weight"] = os.path.basename(weight_path_used)
        if checkpoint_label:
            row["checkpoint"] = checkpoint_label
        for col in GT_COLS:
            m = metrics_for_pos_sets(Z, pos_sets[col], ks=(1, 5, 10))
            for mname, mval in m.items():
                row[f"{col}_{mname}"] = mval
        rows.append(row)
        _save_npz("LPP", k, {"W": W_lpp, "mu_in": mu_lpp})

    # Random Projection
    for k in dims:
        W_rp = fit_random_projector(dim_in=E_tr.shape[1], k=k, seed=seed + k)
        mu_rp = E_tr.mean(axis=0).astype(np.float32)
        Z = transform_linear_projector(E_te, W_rp, mu_rp)
        row = {"method": "Random Projection", "base_dim": base_dim, "away_dim": k}
        row["base_weight"] = os.path.basename(weight_path_used)
        if checkpoint_label:
            row["checkpoint"] = checkpoint_label
        for col in GT_COLS:
            m = metrics_for_pos_sets(Z, pos_sets[col], ks=(1, 5, 10))
            for mname, mval in m.items():
                row[f"{col}_{mname}"] = mval
        rows.append(row)
        _save_npz("Random Projection", k, {"W": W_rp, "mu_in": mu_rp})

    return pd.DataFrame(rows)


def run_ncwp_sweep(
    corpus_csv: str,
    weights_dir: str,
    base_dim: Optional[int],
    dims: List[int] | None,
    device: str,
    block_size: int,
    val_ratio: float,
    seed: int,
    mode: str = "best_only",  # 'best_only' | 'checkpoints' | 'base-dims'
    out_dir: Optional[str] = None,
    save_projectors: bool = False,
    projector_root: Optional[str] = None,
    dims_mode: str = "none",
    progress: bool = False,
    progress_bar: bool = False,
    include_best_last: bool = False,
) -> pd.DataFrame:
    """
    Orchestrate evaluations over multiple base weights according to mode.
    - best_only: single run (uses best for base_dim or fallback last)
    - checkpoints: all checkpoints for specified base_dim within weights_dir
    - base-dims: iterate over all base_dims found under weights_dir (best/fallback last per dim)
    """
    all_rows: List[pd.DataFrame] = []
    # enumerate targets
    weight_groups = _list_all_base_weights(weights_dir)
    def _compute_dims_for_base(bd_val: int) -> List[int] | None:
        if dims is not None:
            return dims
        if dims_mode == "half_range":
            out = []
            limit = max(2, int(bd_val) // 2)
            d = 2
            while d <= limit:
                out.append(d)
                d *= 2
            return out
        if dims_mode == "powers_of_two":
            out = []
            d = 2
            while d <= int(bd_val):
                out.append(d)
                d *= 2
            return out
        return None

    if mode == "best_only":
        if base_dim is None:
            raise ValueError("base_dim must be provided for mode=best_only")
        ckpts = weight_groups.get(int(base_dim), [])
        if not ckpts:
            raise FileNotFoundError(f"No weights found for base_dim={base_dim} under {weights_dir}")
        # pick first (sorted pref: best -> last -> step asc)
        chosen = ckpts[0]
        label = _derive_base_id(chosen)
        dims_local = _compute_dims_for_base(int(base_dim))
        if progress and not progress_bar:
            print(f"[Sweep] best_only: using {os.path.basename(chosen)}")
        df = run_ncwp_evaluation(
            corpus_csv=corpus_csv,
            weights_dir=weights_dir,
            base_dim=int(base_dim),
            dims=dims_local,
            device=device,
            block_size=block_size,
            val_ratio=val_ratio,
            seed=seed,
            weight_path=chosen,
            save_projectors=save_projectors,
            projector_root=projector_root,
            checkpoint_label=label,
            progress=progress,
            progress_bar=progress_bar,
        )
        all_rows.append(df)
        # per-run outputs
        if out_dir:
            sub = os.path.join(out_dir, "runs", label)
            os.makedirs(sub, exist_ok=True)
            df.to_csv(os.path.join(sub, "comparison.csv"), index=False)
    elif mode == "checkpoints":
        if base_dim is None:
            raise ValueError("base_dim must be provided for mode=checkpoints")
        ckpts = weight_groups.get(int(base_dim), [])
        if not ckpts:
            raise FileNotFoundError(f"No weights (checkpoints) found for base_dim={base_dim} under {weights_dir}")
        # By default, include only step_*; optionally keep best/last
        if not include_best_last:
            before = len(ckpts)
            ckpts = [cp for cp in ckpts if "_step_" in os.path.basename(cp)]
            if progress and not progress_bar:
                print(f"[Sweep] checkpoints: filtered step_* only ({len(ckpts)}/{before})")
        else:
            if progress and not progress_bar:
                print(f"[Sweep] checkpoints: including best/last ({len(ckpts)})")
        for cp in ckpts:
            label = _derive_base_id(cp)
            dims_local = _compute_dims_for_base(int(base_dim))
            if progress and not progress_bar:
                print(f"[Sweep] run -> {os.path.basename(cp)} as {label}")
            df = run_ncwp_evaluation(
                corpus_csv=corpus_csv,
                weights_dir=weights_dir,
                base_dim=int(base_dim),
                dims=dims_local,
                device=device,
                block_size=block_size,
                val_ratio=val_ratio,
                seed=seed,
                weight_path=cp,
                save_projectors=save_projectors,
                projector_root=projector_root,
                checkpoint_label=label,
                progress=progress,
                progress_bar=progress_bar,
            )
            all_rows.append(df)
            if out_dir:
                sub = os.path.join(out_dir, "runs", label)
                os.makedirs(sub, exist_ok=True)
                df.to_csv(os.path.join(sub, "comparison.csv"), index=False)
    elif mode == "base-dims":
        for bd, ckpts in weight_groups.items():
            if not ckpts:
                continue
            chosen = ckpts[0]  # best available for that base_dim
            label = _derive_base_id(chosen)
            dims_local = _compute_dims_for_base(int(bd))
            df = run_ncwp_evaluation(
                corpus_csv=corpus_csv,
                weights_dir=weights_dir,
                base_dim=int(bd),
                dims=dims_local,
                device=device,
                block_size=block_size,
                val_ratio=val_ratio,
                seed=seed,
                weight_path=chosen,
                save_projectors=save_projectors,
                projector_root=projector_root,
                checkpoint_label=label,
                progress=progress,
                progress_bar=progress_bar,
            )
            all_rows.append(df)
            if out_dir:
                sub = os.path.join(out_dir, "runs", f"dim{bd}")
                os.makedirs(sub, exist_ok=True)
                df.to_csv(os.path.join(sub, "comparison.csv"), index=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # concat
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


