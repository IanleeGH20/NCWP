#!/usr/bin/env python3
"""Is the FIFO memory bank real, and does it change the main STS result?

The paper claims negatives come from "in-batch items plus a FIFO memory bank with
hard-negative selection" (M=4096, K=32). Three different trainers exist in this
repo and they do NOT agree, so this script settles the question empirically on the
trainer that produced the main STS table (`update_main_sts_ncwp.py`, whose header
states use_memory_bank=True / hard_neg_k=32 / retraction_interval=200 /
refine_knn_rounds=0, k=40, tau=0.2, temp=0.07).

The trainer below is copied from that script, with instrumentation added only:
  bank_rows      bank occupancy at each step
  pool_width     number of negative candidates the denominator draws from
  frac_from_bank share of the selected top-K hard negatives that came from the bank

`frac_from_bank` is the decisive number: if the bank supplies ~0% of the selected
negatives it is inert in effect even when wired in, and if it supplies most of them
the bank genuinely drives the loss.

Runs bank ON vs OFF on the same seeds so the score difference is attributable.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rebuttal.common import (
    MODEL_CONFIGS,
    apply_zca,
    compute_zca,
    load_sts_benchmark,
    load_sts_embeddings,
    spearman_sts,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ROOT_V2 = "/workspace/NCWP/rebuttal_outputs_v2"

# main-STS protocol constants (update_main_sts_ncwp.py)
K_NEIGH = 40
COSINE_TAU = 0.2
TEMPERATURE = 0.07


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class MemoryBank:
    """Verbatim from update_main_sts_ncwp.py / multiseed_runner.py."""

    def __init__(self, size, dim):
        self.bank = torch.zeros(size, dim, device=DEVICE)
        self.ptr = 0
        self.full = False
        self.size = size

    def update(self, z):
        B = z.shape[0]
        end = min(self.ptr + B, self.size)
        n1 = end - self.ptr
        self.bank[self.ptr:end] = z[:n1]
        if B > n1:
            self.bank[:B - n1] = z[n1:]
            self.ptr = B - n1
            self.full = True
        else:
            self.ptr = end % self.size
            if self.ptr == 0:
                self.full = True

    def get(self):
        return self.bank if self.full else self.bank[:self.ptr]


def train_main_sts(Xw_fit, knn_idx, knn_sims, rank, seed,
                   use_memory_bank=True, use_hard_neg=True,
                   hard_neg_k=32, memory_bank_size=4096,
                   retraction_interval=200, refine_knn_rounds=0,
                   temperature=TEMPERATURE, lambda_cov=0.05, lambda_orth=0.02,
                   lr=8e-3, max_epochs=20, batch_pairs=256, cosine_tau=COSINE_TAU):
    set_seed(seed)
    N, D = Xw_fit.shape
    r = rank
    Xw = torch.from_numpy(Xw_fit).to(DEVICE)
    knn = torch.from_numpy(knn_idx).to(DEVICE)
    sims = torch.from_numpy(knn_sims.astype(np.float32)).to(DEVICE)
    B = batch_pairs
    steps = min(160, max(100, math.ceil(N / (2 * B))))
    round_T = max_epochs * steps
    lr_min = 0.2 * lr
    warmup = 200
    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
    mbank = MemoryBank(memory_bank_size, r) if use_memory_bank else None
    overall_best_W = None
    overall_best_loss = float("inf")
    diag = dict(bank_rows=[], pool_width=[], frac_from_bank=[], steps=0)

    def _one_round(knn_loc, sims_loc):
        nonlocal overall_best_W, overall_best_loss
        round_gs = 0
        round_best_loss = float("inf")
        round_best_W = None
        no_imp = 0
        for _ in range(max_epochs):
            ep_loss = 0.0
            for _ in range(steps):
                if round_gs < warmup:
                    cur_lr = lr_min + (lr - lr_min) * round_gs / max(1, warmup)
                else:
                    t = (round_gs - warmup) / max(1, round_T - warmup)
                    cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr

                with torch.no_grad():
                    a_idx = torch.randint(0, N, (B,), device=DEVICE)
                    b_list = []
                    for anc in a_idx.tolist():
                        valid = (sims_loc[anc] >= cosine_tau).nonzero(as_tuple=True)[0]
                        if len(valid) == 0:
                            valid = torch.arange(knn_loc.shape[1], device=DEVICE)
                        pick = valid[torch.randint(0, len(valid), (1,)).item()]
                        b_list.append(knn_loc[anc, pick].item())
                    b_idx = torch.tensor(b_list, device=DEVICE)
                    batch = torch.cat([Xw[a_idx], Xw[b_idx]], dim=0)

                Z = F.normalize(batch @ W, dim=1)
                Za, Zb = Z[:B], Z[B:]
                sim_a_to_b = (Za @ Zb.T) / temperature
                sim_a_to_a = (Za @ Za.T) / temperature
                sim_a_to_a.fill_diagonal_(-1e9)
                sim_b_to_a = (Zb @ Za.T) / temperature
                sim_b_to_b = (Zb @ Zb.T) / temperature
                sim_b_to_b.fill_diagonal_(-1e9)
                if use_memory_bank and mbank is not None and len(mbank.get()) > 0:
                    bank = mbank.get().detach()
                    sim_a_bank = (Za @ bank.T) / temperature
                    sim_b_bank = (Zb @ bank.T) / temperature
                else:
                    sim_a_bank = sim_b_bank = None

                pos_logit_ab = sim_a_to_b.diag()
                neg_a_inb = sim_a_to_b.clone()
                neg_a_inb.fill_diagonal_(-1e9)
                neg_pool_a = torch.cat([neg_a_inb, sim_a_to_a], dim=1)
                width_inbatch = neg_pool_a.shape[1]
                if sim_a_bank is not None:
                    neg_pool_a = torch.cat([neg_pool_a, sim_a_bank], dim=1)
                pos_logit_ba = sim_b_to_a.diag()
                neg_b_inb = sim_b_to_a.clone()
                neg_b_inb.fill_diagonal_(-1e9)
                neg_pool_b = torch.cat([neg_b_inb, sim_b_to_b], dim=1)
                if sim_b_bank is not None:
                    neg_pool_b = torch.cat([neg_pool_b, sim_b_bank], dim=1)

                if use_hard_neg:
                    K = min(hard_neg_k, neg_pool_a.shape[1])
                    neg_a_sel, sel_idx_a = torch.topk(neg_pool_a, k=K, dim=1)
                    neg_b_sel, _ = torch.topk(neg_pool_b, k=K, dim=1)
                    # a selected column >= width_inbatch means it came from the bank
                    frac = float((sel_idx_a >= width_inbatch).float().mean())
                else:
                    neg_a_sel, neg_b_sel = neg_pool_a, neg_pool_b
                    frac = (float(neg_pool_a.shape[1] - width_inbatch) / neg_pool_a.shape[1]
                            if neg_pool_a.shape[1] else 0.0)

                diag["bank_rows"].append(0 if sim_a_bank is None else int(sim_a_bank.shape[1]))
                diag["pool_width"].append(int(neg_pool_a.shape[1]))
                diag["frac_from_bank"].append(frac)
                diag["steps"] += 1

                denom_a = torch.logsumexp(
                    torch.cat([pos_logit_ab.unsqueeze(1), neg_a_sel], dim=1), dim=1)
                denom_b = torch.logsumexp(
                    torch.cat([pos_logit_ba.unsqueeze(1), neg_b_sel], dim=1), dim=1)
                loss = 0.5 * (-(pos_logit_ab - denom_a).mean() + -(pos_logit_ba - denom_b).mean())
                Zm = Z - Z.mean(0)
                cov = (Zm.T @ Zm) / max(len(Z) - 1, 1)
                off = cov - torch.diag(torch.diag(cov))
                loss += lambda_cov * (off ** 2).sum() / r
                Wn = F.normalize(W, dim=0)
                loss += lambda_orth * ((Wn.T @ Wn - torch.eye(r, device=DEVICE)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if use_memory_bank and mbank is not None:
                    mbank.update(torch.cat([Za, Zb], dim=0).detach())
                if retraction_interval > 0 and (round_gs + 1) % retraction_interval == 0:
                    with torch.no_grad():
                        Q, _ = torch.linalg.qr(W.data)
                        W.data = Q[:, :r]
                ep_loss += loss.item()
                round_gs += 1
            ep_loss /= steps
            if ep_loss + 1e-6 < round_best_loss:
                round_best_loss = ep_loss
                no_imp = 0
                round_best_W = W.detach().clone()
                if ep_loss < overall_best_loss:
                    overall_best_loss = ep_loss
                    overall_best_W = round_best_W.clone()
            else:
                no_imp += 1
                if no_imp >= 6:
                    break

    _one_round(knn, sims)
    for _ in range(refine_knn_rounds):
        with torch.no_grad():
            Zfull = F.normalize(Xw @ overall_best_W, dim=1)
            sim_r = Zfull @ Zfull.T
            sim_r.fill_diagonal_(-1e9)
            _, knn_new = torch.topk(sim_r, k=K_NEIGH, dim=1)
            sims_new = sim_r.gather(1, knn_new)
            W.data.copy_(overall_best_W)
            opt.state = {}
        if mbank:
            mbank.ptr = 0
            mbank.full = False
        _one_round(knn_new, sims_new)
    return overall_best_W.cpu().numpy(), diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    p.add_argument("--dims", type=int, nargs="+", default=[20, 80, 320])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--basename", default="table_memory_bank_check")
    args = p.parse_args()

    sts = load_sts_benchmark()  # train+test 7,128 — the main-table protocol
    rows = []
    for model in args.model:
        X_fit, X_eval, _ = load_sts_embeddings(model, sts)
        keys = list(X_eval)
        X_eval_mat = np.stack([X_eval[s] for s in keys]).astype(np.float32)

        mu_z, s_z = compute_zca(X_fit, shrink=0.08)
        Xw_fit = apply_zca(X_fit, mu_z, s_z)
        Xw_eval = apply_zca(X_eval_mat, mu_z, s_z)

        xt = torch.from_numpy(Xw_fit).to(DEVICE)
        sim_all = xt @ xt.T
        sim_all.fill_diagonal_(-1e9)
        vals, idx = torch.topk(sim_all, k=K_NEIGH, dim=1)
        knn_idx = idx.cpu().numpy()
        knn_sims = vals.cpu().numpy()
        del xt, sim_all
        torch.cuda.empty_cache()

        for r in args.dims:
            for use_bank in (True, False):
                scores, diags = [], []
                for seed in args.seeds:
                    W, diag = train_main_sts(Xw_fit, knn_idx, knn_sims, r, seed,
                                             use_memory_bank=use_bank)
                    Z = Xw_eval @ W
                    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
                    scores.append(spearman_sts({keys[i]: Z[i] for i in range(len(keys))},
                                               sts["eval_pairs"], sts["eval_scores"]))
                    diags.append(diag)
                d0 = diags[0]
                rows.append(dict(
                    model=model, r=r, memory_bank="ON" if use_bank else "OFF",
                    memory_bank_size=4096 if use_bank else 0, hard_neg_K=32,
                    qr_interval=200, spearman=round(float(np.mean(scores)), 4),
                    std=round(float(np.std(scores)), 4),
                    per_seed=[round(float(v), 4) for v in scores],
                    steps=d0["steps"],
                    bank_rows_final=d0["bank_rows"][-1],
                    bank_rows_max=max(d0["bank_rows"]),
                    pool_width_final=d0["pool_width"][-1],
                    pool_width_max=max(d0["pool_width"]),
                    frac_from_bank_mean=round(float(np.mean(d0["frac_from_bank"])), 4),
                    frac_from_bank_last100=round(float(np.mean(d0["frac_from_bank"][-100:])), 4),
                ))
                print(f"[{model}] r={r:>4} bank={'ON ' if use_bank else 'OFF'} "
                      f"Spearman={np.mean(scores):.2f}±{np.std(scores):.2f}  "
                      f"steps={d0['steps']} bank_rows(final/max)="
                      f"{d0['bank_rows'][-1]}/{max(d0['bank_rows'])} "
                      f"pool_width(final/max)={d0['pool_width'][-1]}/{max(d0['pool_width'])} "
                      f"top-K에서 bank 비중={np.mean(d0['frac_from_bank']):.3f}", flush=True)

    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    csv = os.path.join(OUT_ROOT_V2, f"{args.basename}.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    with open(os.path.join(OUT_ROOT_V2, f"{args.basename}.jsonl"), "w") as f:
        for rr in rows:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    print(f"\nSaved: {csv}")


if __name__ == "__main__":
    main()
