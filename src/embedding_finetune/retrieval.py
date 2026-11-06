"""Recall@k retrieval evaluation for a set of (anchor, positive) queries."""

from __future__ import annotations

import numpy as np


def recall_at_k(query_vecs: np.ndarray, doc_vecs: np.ndarray, gold_indices: np.ndarray, k: int = 5) -> float:
    """For each row of ``query_vecs``, find top-k cosine neighbors in ``doc_vecs``
    and check whether ``gold_indices[i]`` is in that top-k. Mean across queries."""
    hits = hits_at_k(query_vecs, doc_vecs, gold_indices, k=k)
    return float(hits.sum()) / max(len(hits), 1)


def hits_at_k(query_vecs: np.ndarray, doc_vecs: np.ndarray, gold_indices: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-query version of :func:`recall_at_k`: a bool array, True where the
    gold document is in the query's top-k. Use it for paired comparisons and
    bootstrap confidence intervals."""
    if query_vecs.dtype != np.float32:
        query_vecs = query_vecs.astype(np.float32)
    if doc_vecs.dtype != np.float32:
        doc_vecs = doc_vecs.astype(np.float32)
    # Normalize both.
    q_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-12
    d_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12
    q = query_vecs / q_norms
    d = doc_vecs / d_norms
    scores = q @ d.T
    hits = np.zeros(scores.shape[0], dtype=bool)
    for i in range(scores.shape[0]):
        top = np.argpartition(-scores[i], k)[:k]
        hits[i] = int(gold_indices[i]) in top.tolist()
    return hits


def retrieval_eval(
    queries: list[str],
    documents: list[str],
    gold_indices: list[int],
    embedder,
    *,
    head=None,
    k: int = 5,
) -> dict:
    """End-to-end recall@k eval. Optional ``head`` is applied to BOTH queries
    and documents (the projection should live on both sides at inference)."""
    from embedding_finetune.train import apply_head

    q_raw = embedder(queries).astype(np.float32)
    d_raw = embedder(documents).astype(np.float32)
    if head is not None:
        q_vecs = apply_head(head, q_raw)
        d_vecs = apply_head(head, d_raw)
    else:
        q_vecs = q_raw
        d_vecs = d_raw
    gold = np.array(gold_indices, dtype=np.int64)
    return {
        "k": k,
        "n_queries": len(queries),
        "n_documents": len(documents),
        "recall_at_k": round(recall_at_k(q_vecs, d_vecs, gold, k=k), 4),
    }
