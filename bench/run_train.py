"""End-to-end benchmark, k-fold cross-validated over (question, answer) pairs:

1. Split the 60 synthetic pairs (cooking / sports / technology) into folds
2. For each fold, train a projection head on the other folds' triplets
   (fastembed BGE-small embeddings, CPU)
3. Retrieve each held-out question's own answer from the full 60-answer
   corpus, with and without the head, and record recall@k hits
4. Pool the held-out queries across folds and report recall@k with a paired
   bootstrap 95% confidence interval on the lift

Every question is evaluated exactly once, by a head that never saw it or its
answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from embedding_finetune import hits_at_k, synthetic_triplets, train_projection_head
from embedding_finetune.train import apply_head

BENCH = Path(__file__).resolve().parent
RESULTS_OUT = BENCH / "results.json"


def _fastembed_callable():
    """Return a function ``texts -> np.ndarray`` that uses fastembed BGE-small."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("fastembed required") from exc

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def _embed(texts):
        return np.stack(list(model.embed(list(texts))))

    return _embed


def _bootstrap_ci(
    baseline: np.ndarray, projected: np.ndarray, *, resamples: int, seed: int
) -> dict[str, list[float]]:
    """Percentile 95% CIs from resampling queries with replacement, paired so
    the baseline and head are always scored on the same resample."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(baseline), size=(resamples, len(baseline)))
    b = baseline[idx].mean(axis=1)
    p = projected[idx].mean(axis=1)

    def ci(x: np.ndarray) -> list[float]:
        return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]

    return {"baseline": ci(b), "projected": ci(p), "lift": ci(p - b)}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=10_000)
    args = p.parse_args()

    print("Loading embedder (fastembed BGE-small)...")
    embedder = _fastembed_callable()

    baseline_hits: list[np.ndarray] = []
    projected_hits: list[np.ndarray] = []
    per_fold: list[dict] = []
    for fold in range(args.folds):
        train_triplets, eval_triplets = synthetic_triplets(
            n_folds=args.folds, fold=fold, seed=args.seed
        )
        corpus = sorted({t.positive for t in train_triplets + eval_triplets})
        queries = [t.anchor for t in eval_triplets]
        gold = np.array([corpus.index(t.positive) for t in eval_triplets], dtype=np.int64)

        q_raw = embedder(queries).astype(np.float32)
        d_raw = embedder(corpus).astype(np.float32)
        base = hits_at_k(q_raw, d_raw, gold, k=args.k)

        head, stats = train_projection_head(
            train_triplets, embedder,
            hidden=args.hidden, epochs=args.epochs, lr=args.lr, margin=args.margin,
            seed=args.seed,
        )
        proj = hits_at_k(apply_head(head, q_raw), apply_head(head, d_raw), gold, k=args.k)

        baseline_hits.append(base)
        projected_hits.append(proj)
        per_fold.append({
            "fold": fold,
            "n_train_triplets": len(train_triplets),
            "n_eval_queries": len(queries),
            "baseline_recall": round(float(base.mean()), 4),
            "projected_recall": round(float(proj.mean()), 4),
            "training_elapsed_secs": stats.elapsed_secs,
            "loss_history": stats.loss_history,
        })
        print(
            f"fold {fold}: {len(train_triplets)} train triplets, {len(queries)} held-out queries | "
            f"recall@{args.k} {base.mean():.4f} -> {proj.mean():.4f}"
        )

    base = np.concatenate(baseline_hits)
    proj = np.concatenate(projected_hits)
    ci = _bootstrap_ci(base, proj, resamples=args.bootstrap, seed=args.seed)
    baseline_recall = float(base.mean())
    projected_recall = float(proj.mean())
    lift = projected_recall - baseline_recall

    print(f"\nHeld-out queries: {len(base)} against {len(corpus)} documents, {args.folds} folds")
    print(f"Baseline recall@{args.k} (raw BGE-small): {baseline_recall:.4f}  95% CI {ci['baseline']}")
    print(f"Projected recall@{args.k} (BGE + head):    {projected_recall:.4f}  95% CI {ci['projected']}")
    print(f"Lift: {lift:+.4f} absolute, 95% CI {ci['lift']}")
    print(f"Queries the head fixed: {int((proj & ~base).sum())}, broke: {int((base & ~proj).sum())}")

    RESULTS_OUT.write_text(
        json.dumps({
            "k": args.k,
            "folds": args.folds,
            "seed": args.seed,
            "n_eval_queries": int(len(base)),
            "n_documents": len(corpus),
            "baseline_recall": round(baseline_recall, 4),
            "projected_recall": round(projected_recall, 4),
            "lift_absolute": round(lift, 4),
            "ci95": ci,
            "bootstrap_resamples": args.bootstrap,
            "queries_fixed_by_head": int((proj & ~base).sum()),
            "queries_broken_by_head": int((base & ~proj).sum()),
            "head_params": head.count_params(),
            "per_fold": per_fold,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
