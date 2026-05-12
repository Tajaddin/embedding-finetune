"""Train a small projection head on top of a frozen embedder."""

from embedding_finetune.dataset import Triplet, synthetic_triplets
from embedding_finetune.head import ProjectionHead, save_head, load_head
from embedding_finetune.retrieval import recall_at_k, retrieval_eval
from embedding_finetune.train import train_projection_head

__version__ = "0.1.0"

__all__ = [
    "Triplet",
    "synthetic_triplets",
    "ProjectionHead",
    "save_head",
    "load_head",
    "recall_at_k",
    "retrieval_eval",
    "train_projection_head",
]
