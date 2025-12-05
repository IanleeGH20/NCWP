from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class EmbeddingBackend:
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class HFMeanPoolingEncoder(EmbeddingBackend):
    """
    Generic HuggingFace encoder with mean pooling + L2 normalization.
    Works for Qwen and many text encoders.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        max_len: int = 512,
        use_half: bool = True,
    ):
        self.device = device
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        try:
            cfg.use_cache = False
        except Exception:
            pass
        dtype = torch.float16 if (use_half and device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: List[str]) -> np.ndarray:
        BATCH_SIZE = 64
        embs: list[np.ndarray] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
            embs.append(pooled.cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), dtype=np.float32)


class NemotronEncoder(EmbeddingBackend):
    """
    nvidia/llama-embed-nemotron-8b encoder with simple average pooling + L2 norm.
    """

    def __init__(self, model_name: str, device: str, max_len: int = 512):
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Disable flash attention for compatibility
        cfg._attn_implementation = "eager"
        cfg.attn_implementation = "eager"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_name,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).eval()
        self.model.to(device)
        self.device = device
        self.max_len = max_len

    def _avg_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = last_hidden_states.to(torch.float32)
        mask = attention_mask[..., None].bool()
        x = x.masked_fill(~mask, 0.0)
        emb = x.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def embed(self, texts: List[str]) -> np.ndarray:
        BATCH_SIZE = 64
        embs_all = []
        for s in range(0, len(texts), BATCH_SIZE):
            chunk = texts[s:s + BATCH_SIZE]
            batch = self.tokenizer(
                chunk,
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**batch)
                emb = self._avg_pool(out.last_hidden_state, batch["attention_mask"])
            embs_all.append(emb.cpu().numpy())
        embs = np.concatenate(embs_all, axis=0) if embs_all else np.zeros((0, 0), dtype=np.float32)
        return embs


