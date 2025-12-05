from __future__ import annotations

import os
from typing import Iterable, List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
import csv

from .data import SimpleVocab
from .model import GPT, GPTConfig


def train_one_dim(
    vocab: SimpleVocab,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    n_embd: int,
    n_head: int,
    n_layer: int,
    block_size: int,
    steps: int,
    lr: float,
    device: str,
    eval_every: int,
    patience: int,
    save_dir: str,
    save_iter_interval: int = 0,
) -> Tuple[GPT, List[Tuple[int, float]], List[Tuple[int, float]], Optional[Dict[str, torch.Tensor]], Optional[int], float]:
    cfg = GPTConfig(
        vocab_size=vocab.vocab_size,
        n_embd=n_embd,
        n_head=max(1, min(n_head, max(1, n_embd // 2))),
        n_layer=n_layer,
        block_size=block_size,
        dropout=0.2,
    )
    model = GPT(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    # Ensure save directory exists early if step-wise checkpoints are requested
    if save_iter_interval and save_iter_interval > 0:
        os.makedirs(save_dir, exist_ok=True)
    step = 0
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = copy.deepcopy(model.state_dict())
    best_step: Optional[int] = 0
    stalls = 0
    train_log: List[Tuple[int, float]] = []   # (step, train_loss)
    val_log: List[Tuple[int, float]] = []     # (step, val_loss)
    while step < steps:
        for xb, yb, _attn in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            _, _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            # Log train loss every step
            train_log.append((step, float(loss.item())))
            # Optional: save checkpoint every N steps
            if save_iter_interval and save_iter_interval > 0 and (step % save_iter_interval == 0):
                ckpt_path = os.path.join(save_dir, f"base_dim_{n_embd}_step_{step}.pt")
                try:
                    torch.save(model.state_dict(), ckpt_path)
                except Exception:
                    pass
            # Validation / early stopping
            if (step % eval_every == 0) or (step == steps):
                val_loss = evaluate(model, dl_va, device)
                val_log.append((step, float(val_loss)))
                if val_loss + 1e-8 < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    best_step = step
                    stalls = 0
                else:
                    stalls += 1
                    if stalls >= patience:
                        break
            if step >= steps:
                break
        if stalls >= patience or step >= steps:
            break
    # Save loss plot and CSV
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"loss_dim_{n_embd}.png")
    csv_path = os.path.join(save_dir, f"loss_dim_{n_embd}.csv")
    try:
        plt.figure(figsize=(8, 4))
        if train_log:
            xs, ys = zip(*train_log)
            plt.plot(xs, ys, label="train", alpha=0.7)
        if val_log:
            xs, ys = zip(*val_log)
            plt.plot(xs, ys, label="val", alpha=0.9, marker="o")
        # Mark best validation point on the plot (vertical line and marker)
        try:
            if best_step is not None and best_step > 0:
                plt.axvline(best_step, color="red", linestyle="--", alpha=0.6, label=f"best@{best_step}")
                # Find val loss at best step to mark the point (if present in val_log)
                try:
                    val_at_best = next(v for s, v in val_log if s == best_step)
                    plt.scatter([best_step], [val_at_best], color="red", zorder=5)
                    plt.annotate(
                        f"best {best_val:.4f} @ {best_step}",
                        xy=(best_step, val_at_best),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=8,
                        color="red",
                    )
                except StopIteration:
                    pass
        except Exception:
            pass
        plt.title(f"Loss vs Steps (dim={n_embd})")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except Exception:
        pass
    try:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "val_loss"])
            # Merge by step; write train rows; for val steps, also write val value
            val_dict = {s: v for s, v in val_log}
            for s, t in train_log:
                w.writerow([s, t, val_dict.get(s, "")])
            # write any remaining val-only steps
            for s, v in val_log:
                if s > len(train_log) or (s, v) not in train_log:
                    # ensure not duplicate
                    pass
    except Exception:
        pass
    return model, train_log, val_log, best_state, best_step, best_val


@torch.no_grad()
def evaluate(model: GPT, dl: DataLoader, device: str) -> float:
    model.eval()
    losses: List[float] = []
    for xb, yb, _ in dl:
        xb = xb.to(device); yb = yb.to(device)
        _, _, loss = model(xb, yb)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_model(model: GPT, vocab: SimpleVocab, save_dir: str, dim: int) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"base_dim_{dim}.pt"))
    vocab.save(os.path.join(save_dir, "vocab.json"))


def train_for_dims(
    vocab: SimpleVocab,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    dims: Iterable[int],
    steps: int,
    lr: float,
    device: str,
    block_size: int,
    n_head: int,
    n_layer: int,
    save_dir: str,
    eval_every: int = 50,
    patience: int = 5,
    save_iter_interval: int = 0,
) -> None:
    for d in dims:
        print(f"[Train] dim={d}")
        model, _, _, best_state, best_step, best_val = train_one_dim(
            vocab=vocab,
            dl_tr=dl_tr,
            dl_va=dl_va,
            n_embd=d,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
            steps=steps,
            lr=lr,
            device=device,
            eval_every=eval_every,
            patience=patience,
            save_dir=save_dir,
            save_iter_interval=save_iter_interval,
        )
        # Save last-step checkpoint for reference
        last_path = os.path.join(save_dir, f"base_dim_{d}_last.pt")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), last_path)
        # Restore best state (if available) and save canonical best weights
        if best_state is not None:
            model.load_state_dict(best_state)
        save_model(model, vocab, save_dir, dim=d)  # saves base_dim_{d}.pt and vocab.json
        best_path = os.path.join(save_dir, f"base_dim_{d}.pt")
        print(f"[Save] (best) {best_path} | (last) {last_path} | best_val={best_val:.6f} at step={best_step}")
        # Append summary CSV
        summary_path = os.path.join(save_dir, "training_summary.csv")
        write_header = not os.path.exists(summary_path)
        try:
            with open(summary_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["dim", "best_val", "best_step", "steps_requested", "best_path", "last_path"])
                w.writerow([d, f"{best_val:.6f}", best_step if best_step is not None else "", steps, best_path, last_path])
        except Exception:
            pass


