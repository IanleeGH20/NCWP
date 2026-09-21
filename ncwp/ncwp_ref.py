"""Faithful NCWP trainer — verbatim from the paper's ../run_ablation.py.

The reproduction must compare against the *published* NCWP, so all reproduction runners
use THIS trainer (not the reimplementation in common.py). `train_ncwp_variant`,
`proj_ncwp`, and the `_*` helpers are copied unchanged from
the reference NCWP trainer that produced the paper tables.

Difference from common.py:train_ncwp (do not reintroduce these): no memory bank,
retraction_interval=10, refine_knn_rounds=1, temperature=0.12 with cosine
annealing, and output normalized by std_out.

Added ONLY: regularizer-ablation variants `no_cov` / `no_orth` / `no_both`
(zeroing lambda_cov / lambda_orth), plus a `fit_project` convenience wrapper.
"""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _orth_penalty(W):
    WT_W = W.T @ W
    return ((WT_W - torch.eye(WT_W.shape[0], device=W.device)) ** 2).mean()


def _cov_penalty(Z):
    B = Z.shape[0]
    if B <= 1:
        return torch.tensor(0., device=Z.device)
    Zc = Z - Z.mean(0, keepdim=True)
    Cov = (Zc.T @ Zc) / (B - 1)
    return ((Cov - torch.eye(Cov.shape[0], device=Z.device)) ** 2).mean()


def _zca_shrink_t(X, shrink=0.08, eps=1e-6):
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    D = Cov.shape[0]
    tr = torch.trace(Cov)
    Cs = (1 - shrink) * Cov + shrink * (tr / D) * torch.eye(D, device=X.device)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.clamp(ev, min=eps)
    S = evec @ torch.diag(1. / torch.sqrt(ev)) @ evec.T
    return mu.squeeze(0), S


@torch.no_grad()
def _knn(X, k, chunk=2048):
    N = X.shape[0]
    XT = X.T
    idx = torch.empty((N, k), dtype=torch.long, device="cpu")
    sim = torch.empty((N, k), dtype=torch.float, device="cpu")
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        s_ = X[s:e] @ XT
        s_[:, torch.arange(s, e, device=X.device)] = -1e9
        tk = torch.topk(s_, k=k, dim=1)
        idx[s:e] = tk.indices.cpu()
        sim[s:e] = tk.values.cpu()
    return idx, sim


def _contrastive_loss(sim_row, pos_idx, topk=None):
    B = sim_row.shape[0]
    pos_logits = sim_row.gather(1, pos_idx.view(-1, 1)).squeeze(1)
    if topk is None or topk >= sim_row.shape[1] - 1:
        denom = torch.logsumexp(sim_row, 1)
    else:
        mask = torch.ones_like(sim_row, dtype=torch.bool)
        mask[torch.arange(B, device=sim_row.device), pos_idx] = False
        negs = sim_row.masked_select(mask).view(B, -1)
        kk = min(topk, negs.shape[1])
        vals, _ = torch.topk(negs, k=kk, dim=1)
        denom = torch.logsumexp(torch.cat([vals, pos_logits.unsqueeze(1)], 1), 1)
    return -(pos_logits - denom).mean()


def train_ncwp_variant(X_np, rank, variant="full",
                       k_neighbors=10, cosine_tau=0.0,
                       temperature=0.12, lambda_cov=0.05, lambda_orth=0.02,
                       topk_negatives=256, retraction_interval=10,
                       refine_knn_rounds=1, max_epochs=20, seed=42,
                       noise_sigma=0.05, whiten=True):
    """
    variant:
      'full'          : full NCWP (kNN-mined positives)
      'random_pairs'  : random positive instead of kNN     (positive-gen ablation)
      'noise'         : Gaussian-noise positive (SimCSE-noise style, whitened space)
      'no_hardneg'    : topk_negatives=None (all in-batch)
      'no_qr'         : retraction_interval=0
      'no_refinement' : refine_knn_rounds=0
      'no_cov'        : lambda_cov=0            (Eq. 6/7 ablation)
      'no_orth'       : lambda_orth=0           (Eq. 6/7 ablation)
      'no_both'       : lambda_cov=lambda_orth=0
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if variant == "no_hardneg":     topk_negatives = None
    if variant == "no_qr":          retraction_interval = 0
    if variant == "no_refinement":  refine_knn_rounds = 0
    if variant == "no_cov":         lambda_cov = 0.0
    if variant == "no_orth":        lambda_orth = 0.0
    if variant == "no_both":        lambda_cov = 0.0; lambda_orth = 0.0

    N, D = X_np.shape
    r = min(rank, D)
    X = torch.from_numpy(X_np).to(DEVICE)
    if whiten:
        mu_in, S = _zca_shrink_t(X)
    else:
        # ablation: skip ZCA whitening (S=identity) — tests whether kNN positives
        # matter in the raw (non-whitened) space
        mu_in = X.mean(0)
        S = torch.eye(D, device=X.device, dtype=X.dtype)
    Xw = F.normalize((X - mu_in) @ S, dim=1)
    k = max(1, min(k_neighbors, N - 1))

    if variant not in ("random_pairs", "noise"):
        knn_idx, knn_sim = _knn(Xw, k)
    else:
        knn_idx = knn_sim = None

    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    lr = 8e-3; lr_min = 0.2 * lr; temp_min = 0.5 * temperature
    warmup = 100; weight_decay = 1e-4
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=weight_decay)

    Bp = max(64, min(1024, N // 64))
    steps = min(160, max(100, math.ceil(N / (2 * Bp))))
    patience = 6; best_W = None; best_loss = float("inf"); no_imp = 0
    global_step = 0

    def _one_round(knn_idx_local, knn_sim_local, tag="init"):
        nonlocal global_step, best_W, best_loss, no_imp
        tau_t = torch.tensor(cosine_tau, dtype=torch.float)
        for epoch in range(1, max_epochs + 1):
            ep_loss = 0.
            for _ in range(steps):
                T = max_epochs * steps
                t = max(0., global_step - warmup) / max(1, T - warmup)
                if global_step < warmup:
                    cur_lr = lr_min + (lr - lr_min) * (global_step / max(1, warmup))
                else:
                    cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr
                t2 = global_step / max(1, T - 1)
                cur_temp = temp_min + 0.5 * (temperature - temp_min) * (1 + math.cos(math.pi * t2))

                with torch.no_grad():
                    a_cpu = torch.randint(0, N, (Bp,), dtype=torch.long)
                    a_idx = a_cpu.to(DEVICE)
                    if variant == "noise":
                        Xa = Xw[a_idx]
                        Xb = F.normalize(Xa + noise_sigma * torch.randn_like(Xa), dim=1)
                        Xbatch = torch.cat([Xa, Xb], 0)
                    else:
                        if variant == "random_pairs":
                            b_cpu = torch.randint(0, N, (Bp,), dtype=torch.long)
                        else:
                            b_list = []
                            for anc in a_cpu.tolist():
                                valid = knn_sim_local[anc] >= tau_t
                                vnb = knn_idx_local[anc][valid]
                                if len(vnb) == 0:
                                    vnb = knn_idx_local[anc]
                                b_list.append(vnb[torch.randint(0, len(vnb), (1,))].item())
                            b_cpu = torch.tensor(b_list, dtype=torch.long)
                        b_idx = b_cpu.to(DEVICE)
                        Xbatch = torch.cat([Xw[a_idx], Xw[b_idx]], 0)

                Z = F.normalize(Xbatch @ W, dim=1)
                sim_m = (Z @ Z.T) / cur_temp
                B2 = Z.shape[0]; B = B2 // 2
                sim_m = sim_m.masked_fill(torch.eye(B2, device=DEVICE, dtype=torch.bool), -1e9)
                idx1 = torch.arange(B, device=DEVICE); idx2 = idx1 + B
                l_i = _contrastive_loss(sim_m[idx1], idx2, topk=topk_negatives)
                l_j = _contrastive_loss(sim_m[idx2], idx1, topk=topk_negatives)
                loss = 0.5 * (l_i + l_j) + lambda_cov * _cov_penalty(Z) + lambda_orth * _orth_penalty(W)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

                if retraction_interval > 0 and (global_step + 1) % retraction_interval == 0:
                    with torch.no_grad():
                        Q, _ = torch.linalg.qr(W.data); W.data = Q[:, :r]
                ep_loss += float(loss.item()); global_step += 1

            ep_loss /= max(1, steps)
            if ep_loss + 1e-6 < best_loss:
                best_loss = ep_loss; no_imp = 0; best_W = W.detach().clone()
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

    _one_round(knn_idx, knn_sim, "init")
    for rr in range(refine_knn_rounds):
        with torch.no_grad():
            Zfull = F.normalize(Xw @ best_W, dim=1)
            ki, ks = _knn(Zfull, k); W.data.copy_(best_W)
        _one_round(ki, ks, f"ref{rr + 1}")

    with torch.no_grad():
        SW = S @ best_W
        Y = (X - mu_in) @ SW
    return (SW.cpu().numpy(), mu_in.cpu().numpy(),
            Y.mean(0).cpu().numpy(), torch.clamp(Y.std(0), min=1e-6).cpu().numpy())


def proj_ncwp(X, W, mu_in, mu_out, std_out):
    z = ((X - mu_in) @ W - mu_out) / std_out
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def fit_project(X_fit, X_eval, rank, seed=42, variant="full", **kw):
    """Train on raw X_fit, return L2-normalized projection of raw X_eval matrix."""
    SW, mu_in, mu_out, std_out = train_ncwp_variant(
        X_fit.astype(np.float32), rank, variant=variant, seed=seed, **kw)
    return proj_ncwp(X_eval.astype(np.float32), SW, mu_in, mu_out, std_out), (SW, mu_in, mu_out, std_out)
