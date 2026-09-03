"""Fusion head (structured emotion classifier) + RL-gated fusion.

Two pieces:
  FusionHead    - small MLP mapping concatenated modality embeddings to a
                  7-way MELD emotion distribution. Trained with plain
                  cross-entropy. This is the "static / equal-weight"
                  baseline fusion.
  RLGatePolicy  - a tiny policy network trained with REINFORCE to output a
                  per-modality soft gate (0..1) applied to the embeddings
                  *before* they reach a FusionHead. Reward = whether the
                  resulting classification matches the gold MELD label.
                  This is the RL component applied to one part of the
                  system, per the assignment's optional standout extension.
                  Its value is measured as an accuracy/F1 delta over the
                  static baseline on held-out data (see notebooks/).

Both operate on the same fixed modality order: (text, vision, audio).
A missing modality is represented as a zero vector, so one checkpoint
serves the Text+Vision-only core demo and the tri-modal extension.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.schema import MELD_EMOTIONS

MODALITY_ORDER = ["text", "vision", "audio"]


def concat_embeddings(embeds: dict[str, np.ndarray | None], embed_dims: dict[str, int]) -> torch.Tensor:
    """embeds: {"text": array or None, "vision": array or None, "audio": array or None}
    Missing modalities become zero vectors of the configured dim."""
    parts = []
    for m in MODALITY_ORDER:
        v = embeds.get(m)
        if v is None:
            parts.append(torch.zeros(embed_dims[m]))
        else:
            parts.append(torch.as_tensor(v, dtype=torch.float32))
    return torch.cat(parts, dim=-1)


def modality_mask(embeds: dict[str, np.ndarray | None]) -> torch.Tensor:
    return torch.tensor([1.0 if embeds.get(m) is not None else 0.0 for m in MODALITY_ORDER])


class FusionHead(nn.Module):
    def __init__(self, embed_dims: dict[str, int], hidden_dim: int = 256, num_classes: int = 7):
        super().__init__()
        in_dim = sum(embed_dims[m] for m in MODALITY_ORDER)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, sum(embed_dims)) concatenated embeddings -> (batch, num_classes) logits."""
        return self.net(x)


class RLGatePolicy(nn.Module):
    """Outputs an independent Bernoulli gate probability per modality."""

    def __init__(self, embed_dims: dict[str, int], hidden_dim: int = 64):
        super().__init__()
        in_dim = sum(embed_dims[m] for m in MODALITY_ORDER)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(MODALITY_ORDER)),
        )

    def gate_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))  # (batch, 3), each in (0, 1)

    def apply_gate(self, x: torch.Tensor, embed_dims: dict[str, int], gate: torch.Tensor) -> torch.Tensor:
        """Multiply each modality's slice of the concatenated vector by its gate value."""
        out, i = [], 0
        for j, m in enumerate(MODALITY_ORDER):
            d = embed_dims[m]
            out.append(x[:, i : i + d] * gate[:, j : j + 1])
            i += d
        return torch.cat(out, dim=-1)


def train_fusion_head(
    fusion: FusionHead, X: torch.Tensor, y: torch.Tensor, epochs: int = 30, lr: float = 1e-3
) -> list[float]:
    """Plain supervised training of the static/baseline fusion head. Returns loss history."""
    opt = torch.optim.Adam(fusion.parameters(), lr=lr)
    history = []
    fusion.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = fusion(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


def train_rl_gate(
    gate: RLGatePolicy,
    fusion: FusionHead,
    embed_dims: dict[str, int],
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
) -> list[float]:
    """REINFORCE training of the gate policy against a *frozen* fusion head.

    reward(i) = +1 if the gated forward pass classifies utterance i correctly, else -1.
    A running mean baseline reduces gradient variance (standard REINFORCE trick).
    """
    for p in fusion.parameters():
        p.requires_grad_(False)
    fusion.eval()

    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    running_baseline = 0.0
    reward_history = []

    for _ in range(epochs):
        opt.zero_grad()
        probs = gate.gate_probs(X)  # (batch, 3)
        dist = torch.distributions.Bernoulli(probs=probs)
        actions = dist.sample()  # (batch, 3) in {0, 1}
        log_prob = dist.log_prob(actions).sum(dim=-1)  # (batch,)

        gated_x = gate.apply_gate(X, embed_dims, actions)
        with torch.no_grad():
            logits = fusion(gated_x)
        pred = logits.argmax(dim=-1)
        reward = torch.where(pred == y, torch.ones_like(y, dtype=torch.float32), -torch.ones_like(y, dtype=torch.float32))

        running_baseline = 0.9 * running_baseline + 0.1 * reward.mean().item()
        advantage = reward - running_baseline

        loss = -(log_prob * advantage.detach()).mean()
        loss.backward()
        opt.step()
        reward_history.append(reward.mean().item())

    return reward_history


@torch.no_grad()
def evaluate(fusion: FusionHead, X: torch.Tensor, y: torch.Tensor, gate: RLGatePolicy | None = None, embed_dims: dict | None = None):
    """Returns (accuracy, macro_f1). If `gate` is given, applies the deterministic
    (expected-value) gate before classifying — the RL-gated fusion. Otherwise this
    is the static/equal-weight baseline."""
    fusion.eval()
    x = X
    if gate is not None:
        gate.eval()
        probs = gate.gate_probs(X)  # deterministic expected gate at eval time
        x = gate.apply_gate(X, embed_dims, probs)
    logits = fusion(x)
    pred = logits.argmax(dim=-1)

    acc = (pred == y).float().mean().item()

    # macro F1 without sklearn dependency
    f1s = []
    for c in range(len(MELD_EMOTIONS)):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)

    return acc, macro_f1
