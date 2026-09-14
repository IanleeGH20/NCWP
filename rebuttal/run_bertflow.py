#!/usr/bin/env python3
"""BERT-flow baseline (Review 1 W4 — promised for camera-ready).

Review 1 asked for BERT-flow alongside ABTT; the rebuttal deferred it to
camera-ready, so it is still outstanding. BERT-flow (Li et al., 2020) maps frozen
embeddings to an isotropic Gaussian with an invertible normalizing flow trained by
maximum likelihood on *unlabeled* text — the same label-free, post-hoc setting as
PCA/ZCA/ABTT/NCWP, fit on the identical fit subsample.

Flow architecture: Glow-style blocks of [ActNorm -> permutation -> affine
coupling]. The original uses a learned invertible 1x1 convolution between
couplings; a dense D x D learned rotation needs a slogdet per step, which is
intractable at D=4096, so we use fixed random permutations (the RealNVP /
Glow-paper "reverse" variant). Everything else — max-likelihood objective,
standard-Gaussian base density, affine couplings — follows BERT-flow.

Reported methods:
  BERTflow_full     flow latents at full D (no compression, like ABTT_full)
  BERTflow_PCA_r    PCA top-r on the flow latents (compressed, comparable to NCWP)

Metric: STS-B Spearman x100 / BEIR-Quora nDCG@10. New baseline -> seeds 0,1,2
(flow init/training is stochastic, so all rows are 3-seed mean +/- std).
Cached embeddings only. Output: rebuttal_outputs_v2/table_bertflow.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rebuttal.common import (
    MODEL_CONFIGS,
    load_quora_cached,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    set_seed,
    spearman_sts,
)
# common.evaluate_retrieval_ndcg10 argsorts a 500 x 522931 numpy matrix per chunk,
# which is hours on CPU for the full Quora corpus; reuse the GPU top-k path.
from rebuttal.run_quora_corpus_sample import gpu_ndcg_recall, gpu_project_linear

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT_V2 = "/workspace/NCWP/rebuttal_outputs_v2"
REBUTTAL_DIMS = {
    "qwen-4b": [40, 80, 160, 320], "qwen-8b": [56, 112, 224, 448], "llama-8b": [32, 64, 128, 256],
}


def save_v2(rows, basename):
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    csv_path = os.path.join(OUT_ROOT_V2, f"{basename}.csv")
    jsonl_path = os.path.join(OUT_ROOT_V2, f"{basename}.jsonl")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


class ActNorm(nn.Module):
    """Data-dependent per-dimension affine init (Glow sec 3.1)."""

    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def forward(self, x):
        if self.initialized.item() == 0:
            with torch.no_grad():
                mu = x.mean(0)
                sd = x.std(0).clamp(min=1e-6)
                self.bias.copy_(-mu / sd)
                self.log_scale.copy_(-torch.log(sd))
                self.initialized.fill_(1)
        z = x * torch.exp(self.log_scale) + self.bias
        return z, self.log_scale.sum().expand(x.shape[0])


class AffineCoupling(nn.Module):
    """Split in half; second half is affinely transformed conditioned on the first."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.d_a = dim // 2
        self.d_b = dim - self.d_a
        self.net = nn.Sequential(
            nn.Linear(self.d_a, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * self.d_b),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        xa, xb = x[:, :self.d_a], x[:, self.d_a:]
        h = self.net(xa)
        shift, raw_scale = h[:, :self.d_b], h[:, self.d_b:]
        log_scale = torch.tanh(raw_scale)  # bounded for stability
        zb = xb * torch.exp(log_scale) + shift
        return torch.cat([xa, zb], 1), log_scale.sum(1)


class Flow(nn.Module):
    def __init__(self, dim, n_blocks=6, hidden=512, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.actnorms = nn.ModuleList()
        self.couplings = nn.ModuleList()
        perms = []
        for _ in range(n_blocks):
            self.actnorms.append(ActNorm(dim))
            self.couplings.append(AffineCoupling(dim, hidden))
            perms.append(torch.randperm(dim, generator=g))
        self.register_buffer("perms", torch.stack(perms))

    def forward(self, x):
        logdet = torch.zeros(x.shape[0], device=x.device)
        for i in range(len(self.actnorms)):
            x, ld = self.actnorms[i](x)
            logdet = logdet + ld
            x = x[:, self.perms[i]]
            x, ld = self.couplings[i](x)
            logdet = logdet + ld
        return x, logdet


def fit_flow(X_fit, seed, n_blocks, hidden, epochs, batch, lr):
    """Maximum-likelihood training against a standard-Gaussian base density."""
    set_seed(seed)
    n, d = X_fit.shape
    flow = Flow(d, n_blocks=n_blocks, hidden=hidden, seed=seed).to(DEVICE)
    X = torch.from_numpy(X_fit.astype(np.float32)).to(DEVICE)

    with torch.no_grad():  # ActNorm data-dependent init
        flow(X[:min(n, 1024)])

    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = min(batch, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, bs):
            xb = X[perm[s:s + bs]]
            z, logdet = flow(xb)
            log_pz = -0.5 * (z ** 2).sum(1) - 0.5 * d * np.log(2 * np.pi)
            loss = -(log_pz + logdet).mean() / d  # per-dim nats
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
        sched.step()
    flow.eval()
    return flow, float(loss.item())


@torch.no_grad()
def flow_latents(flow, X, chunk=8192):
    out = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(X[i:i + chunk].astype(np.float32)).to(DEVICE)
        z, _ = flow(xb)
        out.append(z.cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def pca_topr(Z_fit, Z_eval, r):
    mu, W = pca_basis(Z_fit, r)
    return (Z_eval - mu) @ W


def pca_basis(Z_fit, r):
    """Mean and top-r right singular vectors of the flow latents."""
    mu = Z_fit.mean(0).astype(np.float32)
    _, _, vt = np.linalg.svd(Z_fit - mu, full_matrices=False)
    return mu, vt[:r].T.astype(np.float32)


def run_sts(model, dims, seeds, eval_split, flow_kw):
    cfg = MODEL_CONFIGS[model]
    sts = load_sts_benchmark(eval_split=eval_split)
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    keys = list(X_eval)
    X_eval_mat = np.stack([X_eval[s] for s in keys]).astype(np.float32)
    pairs, scores = sts["eval_pairs"], sts["eval_scores"]
    rows = []

    def score(Z):
        return spearman_sts({keys[i]: Z[i] for i in range(len(keys))}, pairs, scores)

    full_scores, nll = [], []
    per_r = {r: [] for r in dims}
    for seed in seeds:
        flow, loss = fit_flow(X_fit, seed, **flow_kw)
        nll.append(loss)
        Z_fit = flow_latents(flow, X_fit)
        Z_eval = flow_latents(flow, X_eval_mat)
        full_scores.append(score(l2n(Z_eval)))
        for r in dims:
            per_r[r].append(score(l2n(pca_topr(Z_fit, Z_eval, r))))
        print(f"[STS/{model}] seed={seed} nll/dim={loss:.3f} full={full_scores[-1]:.2f}", flush=True)
        del flow
        torch.cuda.empty_cache()

    def add(method, r, vals, notes):
        rows.append(metadata_row(
            dataset="stsb", split=eval_split, model_nickname=model,
            hf_model_id=cfg["hf_model_id"], embedding_type="mean_pool", method=method, r=r,
            seed=str(list(seeds)), fit_size=sts["fit_size"], eval_size=sts["eval_size"],
            metric_name="spearman_x100", metric_value=round(float(np.mean(vals)), 4),
            std_if_available=round(float(np.std(vals)), 4),
            per_seed=[round(float(v), 4) for v in vals],
            flow_blocks=flow_kw["n_blocks"], flow_hidden=flow_kw["hidden"],
            flow_nll_per_dim=round(float(np.mean(nll)), 4), notes=notes))

    add("BERTflow_full", cfg["hidden_dim"], full_scores, "glow-style flow, fixed perms; full D")
    for r in dims:
        add("BERTflow_PCA_r", r, per_r[r], "PCA top-r on flow latents")
        print(f"[STS/{model}] r={r:>4} BERTflow_PCA={np.mean(per_r[r]):.2f}"
              f"±{np.std(per_r[r]):.2f}", flush=True)
    return rows


def run_quora(model, dims, seeds, flow_kw):
    cfg = MODEL_CONFIGS[model]
    q = load_quora_cached(model)
    corpus, queries, X_fit = q["corpus_np"], q["query_np"], q["X_fit"]
    rows = []

    d_full = cfg["hidden_dim"]
    full_scores, nll = [], []
    per_r = {r: [] for r in dims}

    def ndcg(Z_c, Z_q, W, mu):
        c = gpu_project_linear(Z_c, W, mu)
        qq = gpu_project_linear(Z_q, W, mu)
        val, _ = gpu_ndcg_recall(c, qq, q["corpus_ids"], q["test_qids"], q["qrels"])
        del c, qq
        torch.cuda.empty_cache()
        return val

    for seed in seeds:
        flow, loss = fit_flow(X_fit, seed, **flow_kw)
        nll.append(loss)
        Z_fit = flow_latents(flow, X_fit)
        Z_c, Z_q = flow_latents(flow, corpus), flow_latents(flow, queries)
        del flow
        torch.cuda.empty_cache()

        full_scores.append(ndcg(Z_c, Z_q, np.eye(d_full, dtype=np.float32), None))
        print(f"[Quora/{model}] seed={seed} nll/dim={loss:.3f} full={full_scores[-1]:.2f}",
              flush=True)
        for r in dims:
            mu, W = pca_basis(Z_fit, r)
            per_r[r].append(ndcg(Z_c, Z_q, W, mu))
            print(f"[Quora/{model}] seed={seed} r={r:>4} -> {per_r[r][-1]:.2f}", flush=True)
        del Z_c, Z_q, Z_fit
        torch.cuda.empty_cache()

    def add(method, r, vals, notes):
        rows.append(metadata_row(
            dataset="quora", split="test", model_nickname=model,
            hf_model_id=cfg["hf_model_id"], embedding_type="mean_pool", method=method, r=r,
            seed=str(list(seeds)), fit_size=q["fit_size"], eval_size=q["eval_size"],
            metric_name="ndcg@10", metric_value=round(float(np.mean(vals)), 4),
            std_if_available=round(float(np.std(vals)), 4),
            per_seed=[round(float(v), 4) for v in vals],
            flow_blocks=flow_kw["n_blocks"], flow_hidden=flow_kw["hidden"],
            flow_nll_per_dim=round(float(np.mean(nll)), 4), notes=notes))

    add("BERTflow_full", cfg["hidden_dim"], full_scores, "glow-style flow, fixed perms; full D")
    for r in dims:
        add("BERTflow_PCA_r", r, per_r[r], "PCA top-r on flow latents")
        print(f"[Quora/{model}] r={r:>4} BERTflow_PCA={np.mean(per_r[r]):.2f}"
              f"±{np.std(per_r[r]):.2f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sts", "quora", "both"], default="sts")
    parser.add_argument("--model", nargs="+", default=["qwen-4b", "qwen-8b", "llama-8b"],
                        choices=list(MODEL_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--eval_split", choices=["train+test", "test"], default="train+test")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--basename", default="table_bertflow")
    args = parser.parse_args()

    flow_kw = dict(n_blocks=args.blocks, hidden=args.hidden, epochs=args.epochs,
                   batch=args.batch, lr=args.lr)
    rows = []
    for model in args.model:
        dims = args.dims if args.dims is not None else REBUTTAL_DIMS[model]
        if args.dataset in ("sts", "both"):
            rows.extend(run_sts(model, dims, args.seeds, args.eval_split, flow_kw))
        if args.dataset in ("quora", "both"):
            rows.extend(run_quora(model, dims, args.seeds, flow_kw))
        save_v2(rows, args.basename)  # checkpoint per model
    csv_path, jsonl_path = save_v2(rows, args.basename)
    print(f"\nSaved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
