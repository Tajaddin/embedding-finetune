# embedding-finetune

> Train a small projection head on top of a frozen embedder for domain-specific retrieval. **197K trained params, CPU-only, evaluated on held-out questions with a bootstrap confidence interval.** On the built-in synthetic set the head does not beat the frozen embedder: recall@5 1.000 → 0.917.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Results

4-fold cross-validation over the 60 synthetic (question, answer) pairs, stratified by topic. Each fold trains a head on the other three folds, then every held-out question retrieves its own answer from all 60 answers. Every question is scored once, by a head that never saw it or its answer.

| | Recall@5 (60 held-out queries, 60 docs) | 95% CI |
|---|---:|---:|
| Raw BGE-small (frozen) | **1.000** | 1.000 to 1.000 |
| BGE-small + 197K-param projection head | 0.917 | 0.833 to 0.983 |
| **Lift** | **−8.3 points** | −16.7 to −1.7 |

CIs are paired bootstrap percentiles over the 60 queries (10,000 resamples). The head pushed 5 gold answers out of the top 5 and pulled none in.

Training stats, per fold:
* 675 triplets from 45 training pairs, 6 epochs
* Triplet margin loss, AdamW @ 1e-3
* 197,248 trainable params (the projection head only — embedder frozen)
* Loss reaches 0.0000 by epoch 2 in every fold

Why the head loses here: 14 of every 15 training positives are a *different* same-topic answer, so the head learns topic clusters. The eval asks for the exact answer among 20 same-topic answers, and frozen BGE-small already ranks every gold answer in the top 5. With no headroom, the head can only blur within-topic ranking.

An earlier version reported +2.0 points (0.260 → 0.280). That eval shared all of its questions and documents with training and mostly scored against another question's answer, so it did not measure held-out retrieval.

Reproduce: `python bench/run_train.py`; repeat runs with the same `--seed` give identical numbers. Raw output, including per-fold numbers: `bench/results.json`.

## Why this exists

Every production RAG team eventually asks "can we make retrieval better on our specific corpus without re-training the embedder?" A small projection head trained with contrastive loss on labeled pairs is the cheapest thing to try before full embedder fine-tuning. Whether it helps depends on headroom and on whether the training positives match the retrieval target, which is why this repo evaluates it on held-out questions.

The pattern:

1. **Freeze a strong general embedder** (BGE-small here, but works with any).
2. **Define triplets** `(anchor, positive, negative)` from your domain — labeled pairs of queries and matching docs.
3. **Train a tiny MLP head** that re-shapes the embedder's output into a domain-tuned space.
4. **Apply the head at both query time and index time** — it lives outside the embedder so any vector store works.

This repo is the minimum-viable version of that recipe — CPU-trainable, no GPU required, ~500 lines.

## Architecture

```
                ┌──────────────────────┐
   text   ─────▶│ Frozen BGE-small     │  (or any fastembed model)
                │  → 384-d vector       │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │ ProjectionHead (MLP) │
                │   Linear(384, 256)   │
                │   ReLU               │
                │   Linear(256, 384)   │
                │   L2 normalize       │
                └──────────┬───────────┘
                           │
                  384-d domain-tuned vector
```

Output dim equals input dim so the projected vector is a drop-in replacement in any existing vector store. Training only updates 197K params (the head); the 33M-param embedder never sees a gradient.

## Quickstart

```bash
pip install -e .
```

```python
from embedding_finetune import (
    synthetic_triplets, train_projection_head, recall_at_k,
)
from embedding_finetune.train import apply_head

# Plug in any embedder — here, fastembed BGE-small.
from fastembed import TextEmbedding
import numpy as np

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed = lambda texts: np.stack(list(model.embed(list(texts))))

# Build triplets (anchor, positive, negative). Fold 0 of 4: 675 training
# triplets and 15 held-out questions. n_train / n_eval only cap these counts.
train, held_out = synthetic_triplets(fold=0)

# Train the projection head.
head, stats = train_projection_head(train, embed, hidden=256, epochs=6)
print(f"Trained {head.count_params():,} params in {stats.elapsed_secs:.2f}s")

# Apply at inference: project both queries AND documents through the head.
query_vecs = apply_head(head, embed(["how do I sear a steak"]))
doc_vecs   = apply_head(head, embed(["Sear meat at high heat to develop the crust."]))
```

Save and reload the head:

```python
from embedding_finetune import save_head, load_head

save_head(head, "head.pt")
loaded = load_head("head.pt")    # ready to apply_head() with
```

## Triplet data

The repo ships a `synthetic_triplets()` generator with three topics (cooking, sports, technology), 20 (question, answer) pairs each. It splits the pairs into train and eval *before* building triplets (`n_folds`, `fold`), so no question or answer appears on both sides. Training triplets pair each question with its own answer and with the other same-topic answers; eval triplets pair each held-out question with its own answer. Negatives come from a different topic in the same split.

For real domain training, build your own `Triplet` list:

```python
from embedding_finetune import Triplet

train = [
    Triplet(anchor="acetaminophen overdose symptoms",
            positive="Acetaminophen overdose causes liver damage, nausea, and abdominal pain.",
            negative="Ibuprofen is an NSAID used for inflammation and pain."),
    # ... (this repo does not measure how many triplets a lift needs)
]
```

Mining negatives: random from a different cluster is the cheap baseline. Hard-negative mining (nearest neighbor in raw embedding space that isn't the gold positive) may help, but it is not measured here and adds complexity — out of scope for v0.1.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
15 passed
```

Coverage:

* `ProjectionHead` output shape and L2 normalization
* `save_head` / `load_head` round-trip preserves outputs exactly
* `recall_at_k` returns 1.0 for self-retrieval, low for random
* `synthetic_triplets` produces well-formed train/eval splits that share no text, score each held-out question against its own answer, and hold out every pair exactly once across folds
* `hits_at_k` marks each query, and `recall_at_k` is its mean
* `train_projection_head` reduces loss over epochs (no NaN, no diverge)
* `apply_head` preserves shape, dtype, normalization

## Project layout

```
.
├── src/embedding_finetune/
│   ├── __init__.py
│   ├── head.py         # ProjectionHead nn.Module + save/load
│   ├── dataset.py      # Triplet dataclass + synthetic 3-topic generator
│   ├── train.py        # triplet-loss training loop with frozen-embedder caching
│   └── retrieval.py    # recall@k + retrieval_eval helpers
├── tests/              # 15 pytest cases
└── bench/
    ├── run_train.py    # 4-fold CV: train per fold, held-out recall@5, bootstrap CI
    └── results.json
```

## Limitations

**No lift on the synthetic set.** Frozen BGE-small already scores recall@5 1.000 on held-out questions, and the head trained on topic-level positives lowers it to 0.917. The synthetic set shows the training and evaluation mechanics, not a gain. Measure on your own held-out pairs, with positives that match what you retrieve, before adopting a head.

**Training cached embeddings only.** The trainer embeds every unique text once and reuses those vectors across epochs. This is the whole reason CPU training is feasible. If your dataset doesn't fit in RAM as float32 vectors, you'll need an in-loop embedder + a bigger refactor.

**No hard-negative mining.** Random-topic negatives are the cheapest signal. Hard negatives (highest-similarity wrong answer) may help but are untested here, and they require either a multi-stage training loop or an online mining step. Out of scope for v0.1.

**Frozen embedder assumption.** This recipe deliberately doesn't fine-tune the embedder. For tasks where the embedder's vocabulary is missing key domain terms (rare languages, niche jargon), full fine-tuning will beat projection-head fine-tuning. The projection head is the right *first* thing to try, not the last.

## License

MIT — see [LICENSE](LICENSE).
