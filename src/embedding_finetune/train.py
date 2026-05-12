"""Train a :class:`ProjectionHead` with triplet loss on top of a frozen embedder.

The embedder is called *once* per text and the embeddings are cached. Training
only updates the projection head. This is why the recipe is CPU-feasible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from embedding_finetune.dataset import Triplet
from embedding_finetune.head import ProjectionHead

Embedder = Callable[[Iterable[str]], np.ndarray]


@dataclass
class TrainStats:
    n_train: int
    n_epochs: int
    elapsed_secs: float
    loss_history: list[float]


def _embed_unique(embedder: Embedder, triplets: list[Triplet]) -> tuple[np.ndarray, dict[str, int]]:
    """Cache embeddings for every unique text across all triplet roles."""
    seen: dict[str, int] = {}
    texts: list[str] = []
    for t in triplets:
        for s in (t.anchor, t.positive, t.negative):
            if s not in seen:
                seen[s] = len(texts)
                texts.append(s)
    vecs = embedder(texts)
    return vecs.astype(np.float32), seen


def train_projection_head(
    triplets: list[Triplet],
    embedder: Embedder,
    *,
    hidden: int = 256,
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 1e-3,
    margin: float = 0.2,
    seed: int = 7,
    log_every: int = 0,
) -> tuple[ProjectionHead, TrainStats]:
    """Train + return the head plus training stats."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    cached_vecs, index_of = _embed_unique(embedder, triplets)
    d_in = cached_vecs.shape[1]
    head = ProjectionHead(d_in=d_in, hidden=hidden)
    optimizer = AdamW(head.parameters(), lr=lr)
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    # Pre-build tensor indices for fast batching.
    anchor_idx = np.array([index_of[t.anchor] for t in triplets])
    positive_idx = np.array([index_of[t.positive] for t in triplets])
    negative_idx = np.array([index_of[t.negative] for t in triplets])

    cached_t = torch.from_numpy(cached_vecs)
    loss_history: list[float] = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(len(triplets))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            a = cached_t[anchor_idx[batch]]
            p = cached_t[positive_idx[batch]]
            n = cached_t[negative_idx[batch]]

            a_p = head(a)
            p_p = head(p)
            n_p = head(n)
            loss = loss_fn(a_p, p_p, n_p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        loss_history.append(round(avg, 4))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[epoch {epoch+1}/{epochs}] loss={avg:.4f}")

    elapsed = time.perf_counter() - t0
    head.eval()
    return head, TrainStats(
        n_train=len(triplets),
        n_epochs=epochs,
        elapsed_secs=round(elapsed, 2),
        loss_history=loss_history,
    )


def apply_head(head: ProjectionHead, vectors: np.ndarray) -> np.ndarray:
    """Run a frozen head over a batch of numpy vectors."""
    head.eval()
    with torch.no_grad():
        v = torch.from_numpy(vectors.astype(np.float32))
        out = head(v).numpy()
    # head() already returns L2-normalized output.
    return out.astype(np.float32)
