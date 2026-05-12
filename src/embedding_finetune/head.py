"""Small projection head trained on top of a frozen embedder.

Architecture: ``Linear(d_in, hidden) → ReLU → Linear(hidden, d_in)``. Output
dim equals input dim so the projected vectors can be used as drop-in
replacements in any existing vector store. The output is L2-normalized.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP that re-shapes frozen embeddings into a domain-tuned space."""

    def __init__(self, d_in: int, hidden: int = 256) -> None:
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.nn.functional.normalize(out, dim=-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_head(head: ProjectionHead, path: str | Path) -> None:
    """Serialize the head + its dim metadata to a single ``.pt`` file."""
    payload = {
        "state_dict": head.state_dict(),
        "d_in": head.d_in,
        "hidden": head.hidden,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_head(path: str | Path) -> ProjectionHead:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    head = ProjectionHead(d_in=payload["d_in"], hidden=payload["hidden"])
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head
