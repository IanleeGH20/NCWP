from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class _TextsDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


@torch.no_grad()
def embed_bert_base(texts: List[str], model_name: str = "bert-base-uncased", device: str = "cpu", batch_size: int = 64, pooling: str = "cls") -> np.ndarray:
    """
    Produce sentence embeddings using a vanilla BERT checkpoint from HuggingFace.
    pooling: "cls" or "mean"
    Returns L2-normalized embeddings.
    """
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()

    ds = _TextsDataset(texts)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    outs: list[np.ndarray] = []
    for batch in dl:
        enc = tok(list(batch), padding=True, truncation=True, return_tensors="pt").to(device)
        out = mdl(**enc)
        if pooling == "cls":
            rep = out.last_hidden_state[:, 0, :]  # [B, H]
        else:
            # Mean pooling over valid tokens (exclude padding)
            attn_mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            summed = (out.last_hidden_state * attn_mask).sum(dim=1)  # [B, H]
            counts = attn_mask.sum(dim=1).clamp(min=1)  # [B, 1]
            rep = summed / counts
        rep = torch.nn.functional.normalize(rep, dim=-1)
        outs.append(rep.detach().cpu().numpy())
    X = np.concatenate(outs, axis=0).astype(np.float32)
    return X


@torch.no_grad()
def embed_simcse(texts: List[str], model_name: str, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    """
    Produce sentence embeddings using a SimCSE checkpoint from HuggingFace.
    Uses CLS token representation from the final hidden state with L2 normalization.
    """
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()

    ds = _TextsDataset(texts)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    outs: list[np.ndarray] = []
    for batch in dl:
        enc = tok(list(batch), padding=True, truncation=True, return_tensors="pt").to(device)
        out = mdl(**enc)
        cls = out.last_hidden_state[:, 0, :]  # [B, H]
        cls = torch.nn.functional.normalize(cls, dim=-1)
        outs.append(cls.detach().cpu().numpy())
    X = np.concatenate(outs, axis=0).astype(np.float32)
    return X


@torch.no_grad()
def embed_bertflow(texts: List[str], model_name: str, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    """
    Produce sentence embeddings using a BERT-flow checkpoint via sentence-transformers.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is required for BERT-flow embeddings. Please install it.") from e

    mdl = SentenceTransformer(model_name, device=device)
    X = mdl.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    X = X.astype(np.float32)
    return X


