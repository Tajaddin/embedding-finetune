"""End-to-end benchmark:

1. Build synthetic topical triplets (cooking / sports / technology)
2. Embed them once with fastembed BGE-small (CPU)
3. Train a projection head with triplet loss (~30s on CPU)
4. Evaluate recall@5 on a held-out (anchor → correct doc) eval

Compares frozen-BGE recall vs BGE + projection-head recall.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding_finetune import (
    ProjectionHead,
    Triplet,
    recall_at_k,
    save_head,
    synthetic_triplets,
    train_projection_head,
)
from embedding_finetune.train import apply_head

BENCH = Path(__file__).resolve().parent
HEAD_OUT = BENCH / "head.pt"
RESULTS_OUT = BENCH / "results.json"


def _fastembed_callable():
    """Return a function ``texts -> np.ndarray`` that uses fastembed BGE-small."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("fastembed required") from exc
    import numpy as np

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def _embed(texts):
        return np.stack(list(model.embed(list(texts))))

    return _embed


def build_eval_set(eval_triplets: list[Triplet]) -> tuple[list[str], list[str], list[int]]:
    """Construct (queries, documents, gold_indices) for recall@k eval.

    Documents = unique positives across the eval split.
    Each query = an anchor; gold = the document index matching its positive.
    Negatives are not used at eval time but help training quality earlier.
    """
    docs: list[str] = []
    doc_idx_of: dict[str, int] = {}
    queries: list[str] = []
    gold: list[int] = []
    for t in eval_triplets:
        if t.positive not in doc_idx_of:
            doc_idx_of[t.positive] = len(docs)
            docs.append(t.positive)
        queries.append(t.anchor)
        gold.append(doc_idx_of[t.positive])
    return queries, docs, gold


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    print("Loading embedder (fastembed BGE-small)...")
    embedder = _fastembed_callable()

    train_triplets, eval_triplets = synthetic_triplets(n_train=args.n_train, n_eval=args.n_eval)
    print(f"Train triplets: {len(train_triplets)} | Eval triplets: {len(eval_triplets)}")

    queries, documents, gold_idx = build_eval_set(eval_triplets)
    print(f"Eval: {len(queries)} queries against {len(documents)} unique documents")

    # Eval BEFORE training (raw BGE).
    import numpy as np

    q_raw = embedder(queries).astype(np.float32)
    d_raw = embedder(documents).astype(np.float32)
    gold = np.array(gold_idx, dtype=np.int64)
    baseline_recall = recall_at_k(q_raw, d_raw, gold, k=args.k)
    print(f"Baseline recall@{args.k} (raw BGE-small): {baseline_recall:.4f}")

    # Train.
    head, stats = train_projection_head(
        train_triplets, embedder,
        hidden=args.hidden, epochs=args.epochs, lr=args.lr, margin=args.margin,
        log_every=1,
    )
    print(f"Trained head: {head.count_params():,} params in {stats.elapsed_secs:.2f}s")

    # Eval AFTER training (projected vectors).
    q_proj = apply_head(head, q_raw)
    d_proj = apply_head(head, d_raw)
    projected_recall = recall_at_k(q_proj, d_proj, gold, k=args.k)
    print(f"Projected recall@{args.k} (BGE + head):    {projected_recall:.4f}")
    lift = projected_recall - baseline_recall
    print(f"\nLift: {lift:+.4f} absolute ({lift*100:+.1f} percentage points)")

    save_head(head, HEAD_OUT)
    print(f"\nSaved head to {HEAD_OUT}")

    RESULTS_OUT.write_text(
        json.dumps({
            "k": args.k,
            "n_train": len(train_triplets),
            "n_eval_queries": len(queries),
            "n_eval_documents": len(documents),
            "baseline_recall": round(baseline_recall, 4),
            "projected_recall": round(projected_recall, 4),
            "lift_absolute": round(lift, 4),
            "head_params": head.count_params(),
            "training_elapsed_secs": stats.elapsed_secs,
            "loss_history": stats.loss_history,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
