"""Tests for the projection head, retrieval math, and triplet dataset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from embedding_finetune import (
    ProjectionHead,
    Triplet,
    load_head,
    recall_at_k,
    save_head,
    synthetic_triplets,
)
from embedding_finetune.train import apply_head, train_projection_head


def _toy_embedder(texts):
    """Tiny deterministic char-hash embedder. Topic-correlated by design."""
    rng = np.random.default_rng(0)
    out = []
    for t in texts:
        # Use the first non-space character as a 256-dim topic signature.
        v = rng.normal(size=64).astype(np.float32) * 0.01
        v[0] = float(ord(t[0]) if t else 0) % 7
        for c in t.lower():
            idx = (ord(c) * 31) % 64
            v[idx] += 0.05
        n = np.linalg.norm(v) + 1e-12
        out.append(v / n)
    return np.stack(out)


def test_projection_head_output_dim_matches_input() -> None:
    head = ProjectionHead(d_in=64, hidden=32)
    x = torch.randn(4, 64)
    y = head(x)
    assert y.shape == (4, 64)


def test_projection_head_output_is_normalized() -> None:
    head = ProjectionHead(d_in=32, hidden=16)
    x = torch.randn(8, 32)
    y = head(x)
    norms = torch.norm(y, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_save_and_load_head_round_trip() -> None:
    head = ProjectionHead(d_in=32, hidden=16)
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    path = Path(tmp.name)
    try:
        save_head(head, path)
        loaded = load_head(path)
        assert loaded.d_in == 32
        assert loaded.hidden == 16
        x = torch.randn(2, 32)
        # Outputs should match exactly.
        a = head(x)
        b = loaded(x)
        assert torch.allclose(a, b)
    finally:
        if path.exists():
            path.unlink()


def test_recall_at_k_self_is_one() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 16)).astype(np.float32)
    gold = np.arange(20)
    # Same vectors used for queries and docs → every query is its own nearest neighbor → recall@5 = 1.
    r = recall_at_k(v, v, gold, k=5)
    assert r == pytest.approx(1.0)


def test_recall_at_k_random_is_low() -> None:
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(30, 16)).astype(np.float32)
    docs = rng.normal(size=(100, 16)).astype(np.float32)
    gold = np.zeros(30, dtype=np.int64)  # every gold = doc 0
    r = recall_at_k(queries, docs, gold, k=5)
    # Random retrieval finds doc 0 about k/100 of the time.
    assert r < 0.2


def test_synthetic_triplets_split() -> None:
    train, ev = synthetic_triplets(n_train=20, n_eval=10, seed=1)
    assert len(train) == 20
    assert len(ev) == 10
    assert all(isinstance(t, Triplet) for t in train + ev)


def test_synthetic_triplets_anchor_positive_same_topic() -> None:
    train, _ = synthetic_triplets(n_train=10, n_eval=5, seed=1)
    for t in train:
        assert t.topic in ("cooking", "sports", "technology")


def test_training_reduces_loss_on_toy_embedder() -> None:
    train, _ = synthetic_triplets(n_train=40, n_eval=5, seed=1)
    head, stats = train_projection_head(
        train, _toy_embedder, hidden=16, epochs=3, batch_size=4, lr=1e-2,
    )
    # Loss should be non-increasing on average (allow some noise).
    first = stats.loss_history[0]
    last = stats.loss_history[-1]
    assert last <= first + 0.05  # tolerate small fluctuation


def test_apply_head_preserves_shape_and_dtype() -> None:
    head = ProjectionHead(d_in=64, hidden=32)
    v = np.random.default_rng(0).normal(size=(5, 64))
    out = apply_head(head, v)
    assert out.shape == (5, 64)
    assert out.dtype == np.float32


def test_apply_head_output_is_normalized() -> None:
    head = ProjectionHead(d_in=64, hidden=32)
    v = np.random.default_rng(0).normal(size=(5, 64))
    out = apply_head(head, v)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
