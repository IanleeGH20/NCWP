from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except Exception:
    _tqdm = None
    _TQDM_AVAILABLE = False

def fit_pca_whitening(X: np.ndarray, n_components: int, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA whitening using Torch SVD on GPU (if available).
    Returns (mean, components, scales) with 'scales' = singular values (not squared).
    """
    X_np = np.asarray(X, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(X_np).to(device)
    mu = Xt.mean(dim=0, keepdim=True)
    C = Xt - mu
    # economy SVD: C = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    comps = Vh.transpose(0, 1)  # [D, r]
    # move to CPU numpy
    mu_np = mu.squeeze(0).detach().cpu().numpy().astype(np.float32)
    comps_np = comps.detach().cpu().numpy().astype(np.float32)
    S_np = S.detach().cpu().numpy().astype(np.float32)
    return mu_np, comps_np, S_np


def transform_pca(X: np.ndarray, mean: np.ndarray, components: np.ndarray, k: int) -> np.ndarray:
    """
    Plain PCA projection (no whitening):
      Z = (X - mean) @ components[:, :k]
      followed by L2 normalization.
    """
    Xc = X - mean
    W = components[:, :k]
    Z = Xc @ W
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z.astype(np.float32)

def transform_pca_whitening(X: np.ndarray, mean: np.ndarray, components: np.ndarray, scales: np.ndarray, k: int) -> np.ndarray:
    Xc = X - mean
    W = components[:, :k]
    Z = Xc @ W
    Z = Z / (np.sqrt(scales[:k] + 1e-8))
    # L2 normalize
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z.astype(np.float32)


def transform_pca_soft_whitening(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    scales: np.ndarray,
    k: int,
    alpha: float = 0.10,
) -> np.ndarray:
    """
    Soft PCA whitening (shrinkage on scales):
      denom = sqrt((1 - alpha) * s_i + alpha * mean(s))
    where s_i are singular values from SVD of centered data.
    """
    Xc = X - mean
    W = components[:, :k]
    Z = Xc @ W
    s = scales[:k].astype(np.float32)
    s_mean = float(s.mean()) if s.size > 0 else 1.0
    denom = np.sqrt((1.0 - alpha) * s + alpha * s_mean + 1e-8).astype(np.float32)
    Z = Z / denom
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z.astype(np.float32)


class LearnableProjector(nn.Module):
    """
    Orthogonal complement projector with a learnable rank-r subspace.
    This approximates a simple NCWP-like projection when trained to maximize isotropy.
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.B = nn.Parameter(torch.randn(dim, rank) * 0.02)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Q, _ = torch.linalg.qr(self.B, mode="reduced")
        P = torch.eye(Q.size(0), device=X.device, dtype=X.dtype) - Q @ Q.t()
        return X @ P


@torch.no_grad()
def l2_normalize_np(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1e-12
    return (X / n).astype(np.float32)

def project_out_topk(X: np.ndarray, mean: np.ndarray, components: np.ndarray, k: int) -> np.ndarray:
    """
    LPP-style orthogonal complement of top-k principal directions (no whitening):
      Z = (X - mean) @ (I - W_k W_k^T)
    where W_k are the first k principal components.
    The output stays in the original base_dim but with top-k subspace removed.
    Finally L2-normalize rows.
    """
    Xc = X - mean
    D = Xc.shape[1]
    if k <= 0:
        Z = Xc
    else:
        Wk = components[:, : min(k, D)]  # [D, k]
        P = np.eye(D, dtype=np.float32) - (Wk @ Wk.T)  # [D, D]
        Z = Xc @ P
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z.astype(np.float32)


def fit_new_proposal_projector(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    k: int,
    eps: float = 1e-8,
    verbose: bool = False,
    use_tqdm: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train 'new_proposal' (NCWP+) projector as in the notebook:
      - ZCA+shrink whitening
      - kNN positive mining + InfoNCE
      - covariance/orth penalties, cosine schedules, retraction, refinement
    The mean/components args are ignored (kept for signature compatibility).
    Returns: (W, mu_in, mu_out, std_out) where Z = (X-mu_in)@W then standardize and L2-normalize.
    """
    if verbose and not use_tqdm:
        print(f"[NCWP] Initialize trainer (rank={int(k)}, N={X.shape[0]}, D={X.shape[1]})")
    trainer = NewProposalTrainer(rank=int(k), verbose=bool(verbose), use_tqdm=bool(use_tqdm))
    return trainer.fit(np.asarray(X, dtype=np.float32))


def transform_new_proposal_projector(
    X: np.ndarray,
    W: np.ndarray,
    mu_in: np.ndarray,
    mu_out: np.ndarray,
    std_out: np.ndarray,
) -> np.ndarray:
    """
    Apply the 'new_proposal' transform:
      Z = (X - mu_in) @ W
      Z = (Z - mu_out) / std_out
      Z = l2_normalize(Z)
    """
    X = np.asarray(X, dtype=np.float32)
    Z = (X - mu_in) @ W
    Z = (Z - mu_out) / std_out
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z.astype(np.float32)


# ======== Helpers and Trainer for 'new_proposal' (NCWP+) ========
def _set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _orth_penalty(W: torch.Tensor) -> torch.Tensor:
    WT_W = W.T @ W
    I = torch.eye(WT_W.shape[0], device=W.device, dtype=W.dtype)
    return torch.mean((WT_W - I) ** 2)


def _cov_penalty(Z: torch.Tensor) -> torch.Tensor:
    B = Z.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    Zc = Z - Z.mean(0, keepdim=True)
    Cov = (Zc.T @ Zc) / (B - 1)
    I = torch.eye(Cov.shape[0], device=Z.device, dtype=Z.dtype)
    return torch.mean((Cov - I) ** 2)


def _zca_shrink(X: torch.Tensor, shrink: float = 0.10, eps: float = 1e-6):
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    N = X.shape[0]
    Cov = (Xc.T @ Xc) / max(N - 1, 1)
    D = Cov.shape[0]
    trace = torch.trace(Cov)
    mean_eig = trace / D
    Cov_shrunk = (1.0 - shrink) * Cov + shrink * mean_eig * torch.eye(D, device=X.device, dtype=X.dtype)
    evals, evecs = torch.linalg.eigh(Cov_shrunk)
    evals = torch.clamp(evals, min=eps)
    S = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T
    return mu.squeeze(0), S


@torch.no_grad()
def _topk_cosine_knn(X: torch.Tensor, k: int, chunk: int = 2048):
    # X normalized
    N = X.shape[0]; device = X.device
    idx_all = torch.empty((N, k), dtype=torch.long, device='cpu')
    X_T = X.T
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Q = X[s:e]
        sims = Q @ X_T
        row_idx = torch.arange(s, e, device=device)
        sims[:, row_idx] = -1e9
        topk = torch.topk(sims, k=k, dim=1).indices.detach().to('cpu')
        idx_all[s:e] = topk
    return idx_all


class NewProposalTrainer:
    """
    Learn a rank-r projector with:
      - ZCA+shrink whitening
      - kNN positive mining + InfoNCE
      - covariance/orth penalties
      - cosine LR/temperature schedule with warmup
      - periodic orthogonal retraction (QR)
      - optional refinement rounds by rebuilding kNN in projected space
    """
    def __init__(
        self,
        rank: int,
        temperature: float = 0.12,
        temp_min: float | None = None,
        use_cosine_temp: bool = True,
        k_neighbors: int = 40,
        shrink: float = 0.08,
        max_epochs: int = 20,
        lr: float = 8e-3,
        lr_min: float | None = None,
        warmup_steps: int = 100,
        use_cosine_lr: bool = True,
        batch_pairs: int | None = None,
        steps_per_epoch: int | None = None,
        lambda_cov: float = 0.05,
        lambda_orth: float = 0.02,
        topk_negatives: int = 256,
        retraction_interval: int = 10,
        refine_knn_rounds: int = 1,
        early_stop_patience: int = 6,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str | None = None,
        verbose: bool = False,
        use_tqdm: bool = False,
    ):
        self.rank = int(rank)
        self.temperature = float(temperature)
        self.temp_min = float(temp_min) if temp_min is not None else None
        self.use_cosine_temp = bool(use_cosine_temp)
        self.k_neighbors = int(k_neighbors)
        self.shrink = float(shrink)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.lr_min = float(lr_min) if lr_min is not None else None
        self.warmup_steps = int(warmup_steps)
        self.use_cosine_lr = bool(use_cosine_lr)
        self.batch_pairs = batch_pairs
        self.steps_per_epoch = steps_per_epoch
        self.lambda_cov = float(lambda_cov)
        self.lambda_orth = float(lambda_orth)
        self.topk_negatives = int(topk_negatives) if topk_negatives is not None else None
        self.retraction_interval = int(retraction_interval)
        self.refine_knn_rounds = int(refine_knn_rounds)
        self.early_stop_patience = int(early_stop_patience)
        self.weight_decay = float(weight_decay)
        self.verbose = bool(verbose)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tqdm = bool(use_tqdm) and _TQDM_AVAILABLE
        _set_seed(seed)

    def _infer_batch_pairs(self, N: int):
        if self.batch_pairs is not None:
            return int(self.batch_pairs)
        return int(max(64, min(1024, max(64, N // 64))))

    def _infer_steps(self, N: int):
        if self.steps_per_epoch is not None:
            return int(self.steps_per_epoch)
        return int(min(160, max(100, math.ceil(N / max(1, 2 * self._infer_batch_pairs(N))))))

    def _contrastive_loss(self, logits_row: torch.Tensor, pos_index: torch.Tensor, topk: int | None = None):
        """
        logits_row: (B, 2B)  — anchor vs all
        pos_index:  (B,)     — positive index per row
        """
        B = logits_row.shape[0]
        pos_logits = logits_row.gather(1, pos_index.view(-1,1)).squeeze(1)
        if (topk is None) or (topk >= logits_row.shape[1] - 1):
            denom = torch.logsumexp(logits_row, dim=1)
        else:
            mask = torch.ones_like(logits_row, dtype=torch.bool)
            mask[torch.arange(B, device=logits_row.device), pos_index] = False
            negs = logits_row.masked_select(mask).view(B, -1)
            kk = min(topk, negs.shape[1])
            vals, _ = torch.topk(negs, k=kk, dim=1)
            denom = torch.logsumexp(torch.cat([vals, pos_logits.unsqueeze(1)], dim=1), dim=1)
        return -(pos_logits - denom).mean()

    def fit(self, X_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert isinstance(X_np, np.ndarray) and X_np.ndim == 2
        N, D = X_np.shape
        r = min(self.rank, D)
        X = torch.from_numpy(np.asarray(X_np, dtype=np.float32)).to(self.device)

        # 1) ZCA+shrink whitening
        mu_in_t, S = _zca_shrink(X, shrink=self.shrink)
        Xw = (X - mu_in_t) @ S
        Xw = F.normalize(Xw, dim=1)

        # 2) Initial kNN
        k = max(1, min(self.k_neighbors, max(1, N - 1)))
        with torch.no_grad():
            knn_idx = _topk_cosine_knn(Xw, k=k, chunk=2048)

        # 3) Learn W
        W = torch.randn(D, r, device=self.device) * (1.0 / math.sqrt(D))
        W = torch.nn.Parameter(W)
        opt = torch.optim.AdamW([W], lr=self.lr, weight_decay=self.weight_decay)

        total_steps_per_round = self._infer_steps(N)
        lr_min = self.lr_min if (self.lr_min is not None) else 0.2 * self.lr
        temp_min = self.temp_min if (self.temp_min is not None) else 0.5 * self.temperature
        global_step = 0
        best_W = None
        best_loss = float("inf")
        no_imp = 0

        def _one_round_train(knn_idx_local, roundsuffix="init"):
            nonlocal global_step, best_W, best_loss, no_imp
            epoch_iter = range(1, self.max_epochs + 1)
            if self.use_tqdm:
                epoch_iter = _tqdm(epoch_iter, desc=f"[NCWP][{roundsuffix}] epochs", leave=False, dynamic_ncols=True, position=0, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for epoch in epoch_iter:
                epoch_loss = 0.0
                step_iter = range(total_steps_per_round)
                for _ in step_iter:
                    # schedules
                    if self.use_cosine_lr:
                        t = max(0.0, (global_step - self.warmup_steps)) / max(1, (self.max_epochs * total_steps_per_round - self.warmup_steps))
                        if global_step < self.warmup_steps:
                            cur_lr = lr_min + (self.lr - lr_min) * (global_step / max(1, self.warmup_steps))
                        else:
                            cur_lr = lr_min + 0.5 * (self.lr - lr_min) * (1 + math.cos(math.pi * t))
                        for pg in opt.param_groups:
                            pg["lr"] = cur_lr
                    cur_temp = self.temperature
                    if self.use_cosine_temp:
                        t2 = global_step / max(1, (self.max_epochs * total_steps_per_round - 1))
                        cur_temp = temp_min + 0.5 * (self.temperature - temp_min) * (1 + math.cos(math.pi * t2))

                    # sample positive pairs
                    with torch.no_grad():
                        Bp = self._infer_batch_pairs(N)
                        a_idx_cpu = torch.randint(0, N, (Bp,), dtype=torch.long)
                        b_idx_cpu = knn_idx_local[a_idx_cpu.numpy(), np.random.randint(0, knn_idx_local.shape[1], size=(Bp,))]
                        a_idx = a_idx_cpu.to(self.device, non_blocking=True)
                        b_idx = b_idx_cpu.to(self.device, non_blocking=True)
                        Xa = Xw[a_idx]; Xb = Xw[b_idx]
                        Xbatch = torch.cat([Xa, Xb], dim=0)  # (2B, D)

                    # forward
                    Z = Xbatch @ W
                    Z = F.normalize(Z, dim=1)
                    sim = (Z @ Z.T) / cur_temp
                    B2 = Z.shape[0]; B = B2 // 2
                    eye_mask = torch.eye(B2, device=self.device, dtype=torch.bool)
                    sim = sim.masked_fill(eye_mask, -1e-9)
                    idx1 = torch.arange(B, device=self.device)
                    idx2 = idx1 + B

                    # loss
                    loss_i = self._contrastive_loss(sim[idx1], idx2, topk=self.topk_negatives)
                    loss_j = self._contrastive_loss(sim[idx2], idx1, topk=self.topk_negatives)
                    loss_contrast = 0.5 * (loss_i + loss_j)
                    loss_cov = _cov_penalty(Z)
                    loss_orth = _orth_penalty(W)
                    loss = loss_contrast + self.lambda_cov * loss_cov + self.lambda_orth * loss_orth

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    # retraction
                    if self.retraction_interval > 0 and ((global_step + 1) % self.retraction_interval == 0):
                        with torch.no_grad():
                            Q, _ = torch.linalg.qr(W.data)
                            W.data = Q[:, :r]

                    epoch_loss += float(loss.item())
                    global_step += 1

                epoch_loss /= max(1, total_steps_per_round)
                # early stop tracking
                if epoch_loss + 1e-6 < best_loss:
                    best_loss = epoch_loss
                    no_imp = 0
                    best_W = W.detach().clone()
                else:
                    no_imp += 1
                    if no_imp >= self.early_stop_patience:
                        if self.verbose and not self.use_tqdm:
                            print(f"[NCWP][{roundsuffix}] early stop at epoch={epoch}, best_loss={best_loss:.6f}")
                        break
                if self.verbose and not self.use_tqdm:
                    print(f"[NCWP][{roundsuffix}] epoch {epoch}/{self.max_epochs} loss={epoch_loss:.6f} best={best_loss:.6f}")

        # init round
        if self.verbose and not self.use_tqdm:
            print("[NCWP] Start initial round")
        _one_round_train(knn_idx, "init")
        # refinement rounds
        for rr in range(self.refine_knn_rounds):
            if self.verbose and not self.use_tqdm:
                print(f"[NCWP] Start refinement round {rr+1}/{self.refine_knn_rounds}")
            with torch.no_grad():
                Zfull = F.normalize((Xw @ best_W), dim=1)
                knn_idx = _topk_cosine_knn(Zfull, k=k, chunk=2048)
            with torch.no_grad():
                W.data.copy_(best_W)
            _one_round_train(knn_idx, f"ref{rr+1}")

        with torch.no_grad():
            S_bestW = S @ best_W  # (D, r)
            Y = (X - mu_in_t) @ S_bestW
            mu_out_t = Y.mean(0)
            std_out_t = torch.clamp(Y.std(0), min=1e-6)

        W_np = S_bestW.detach().cpu().numpy().astype(np.float32)
        mu_in_np = mu_in_t.detach().cpu().numpy().astype(np.float32)
        mu_out_np = mu_out_t.detach().cpu().numpy().astype(np.float32)
        std_out_np = std_out_t.detach().cpu().numpy().astype(np.float32)
        if self.verbose and not self.use_tqdm:
            print("[NCWP] Training complete")
        return W_np, mu_in_np, mu_out_np, std_out_np


# -------- Locality Preserving Projections (LPP) --------
def fit_lpp_projector(
    X: np.ndarray,
    k: int,
    n_neighbors: int = 10,
    reg: float = 1e-4,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit LPP projection directions using a k-NN graph.
    Solve: (X^T L X) a = lambda (X^T D X) a, take eigenvectors with smallest lambdas.
    Returns (W_lpp [D,k], mu_in).
    """
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # Build adjacency W (symmetric)
    if metric == "cosine":
        S = (Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)) @ (Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)).T
        np.fill_diagonal(S, -np.inf)
        idx = np.argpartition(-S, kth=min(n_neighbors, N - 1) - 1, axis=1)[:, :n_neighbors]
        Wg = np.zeros((N, N), dtype=np.float32)
        rows = np.repeat(np.arange(N), idx.shape[1])
        cols = idx.reshape(-1)
        Wg[rows, cols] = 1.0
        Wg = np.maximum(Wg, Wg.T)
    else:
        # Euclidean kNN
        Xn = Xc
        d2 = np.sum(Xn * Xn, axis=1, keepdims=True) - 2 * (Xn @ Xn.T) + np.sum(Xn * Xn, axis=1, keepdims=True).T
        np.fill_diagonal(d2, np.inf)
        idx = np.argpartition(d2, kth=min(n_neighbors, N - 1) - 1, axis=1)[:, :n_neighbors]
        Wg = np.zeros((N, N), dtype=np.float32)
        rows = np.repeat(np.arange(N), idx.shape[1])
        cols = idx.reshape(-1)
        Wg[rows, cols] = 1.0
        Wg = np.maximum(Wg, Wg.T)
    d = Wg.sum(axis=1)
    Dg = np.diag(d)
    Lg = Dg - Wg
    XtDX = Xc.T @ Dg @ Xc + reg * np.eye(D, dtype=np.float32)
    XtLX = Xc.T @ Lg @ Xc
    # Solve generalized eigenproblem via symmetric matrix
    # inv(XtDX) @ XtLX
    try:
        A = np.linalg.solve(XtDX, XtLX)
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(XtDX) @ XtLX
    # Smallest eigenvalues -> projection
    w, V = np.linalg.eigh(A)
    order = np.argsort(w)  # ascending
    V = V[:, order]
    r = min(k, D)
    W = V[:, :r].astype(np.float32)
    return W, mu.squeeze(0).astype(np.float32)


def transform_linear_projector(X: np.ndarray, W: np.ndarray, mu_in: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    Z = (X - mu_in) @ W
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z.astype(np.float32)


# -------- Random Projection --------
def fit_random_projector(dim_in: int, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    M = rng.normal(size=(dim_in, k)).astype(np.float32)
    # Orthonormalize columns via QR
    Q, _ = np.linalg.qr(M)
    return Q[:, :k].astype(np.float32)


