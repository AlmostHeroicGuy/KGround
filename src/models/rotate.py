from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


Mode = Literal["single", "head-batch", "tail-batch"]


@dataclass(frozen=True)
class RotatEConfig:
    """Hyperparameters that define the RotatE scoring architecture."""

    num_entities: int
    num_relations: int
    embedding_dim: int = 1000
    gamma: float = 24.0
    epsilon: float = 2.0
    norm: int = 1


class RotatE(nn.Module):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

    The paper represents entities as complex vectors h, t in C^k and each
    relation as a unit-modulus complex vector r = exp(i theta). A true triple is
    expected to satisfy t = h * r element-wise, where * is the Hadamard product.

    We store entities as concatenated real and imaginary parts, and relations as
    unconstrained phase parameters. At scoring time relation phases are mapped
    to unit complex numbers with cos/sin, which enforces |r_i| = 1 exactly.
    """

    def __init__(self, config: RotatEConfig):
        super().__init__()
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if config.norm not in (1, 2):
            raise ValueError("RotatE supports L1 or L2 distance; the paper uses L1")

        self.config = config
        self.embedding_dim = config.embedding_dim
        self.gamma = nn.Parameter(torch.tensor([config.gamma]), requires_grad=False)

        # This follows the official RotatE parameterization: relation embeddings
        # live in [-range, range] and are converted to phases in [-pi, pi].
        self.embedding_range = nn.Parameter(
            torch.tensor([(config.gamma + config.epsilon) / config.embedding_dim]),
            requires_grad=False,
        )

        self.entity_embedding = nn.Embedding(config.num_entities, config.embedding_dim * 2)
        self.relation_embedding = nn.Embedding(config.num_relations, config.embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(
            self.entity_embedding.weight,
            -self.embedding_range.item(),
            self.embedding_range.item(),
        )
        nn.init.uniform_(
            self.relation_embedding.weight,
            -self.embedding_range.item(),
            self.embedding_range.item(),
        )

    def forward(
        self,
        positive_sample: Tensor,
        negative_sample: Tensor | None = None,
        mode: Mode = "single",
    ) -> Tensor:
        """
        Score triples as gamma - ||h o r - t||.

        Args:
            positive_sample: LongTensor [batch, 3] containing (head, relation, tail).
            negative_sample: For head-batch/tail-batch, LongTensor [batch, n_neg]
                containing replacement entity ids.
            mode: single scores positives; head-batch corrupts heads; tail-batch
                corrupts tails.

        Returns:
            Tensor [batch] for single mode or [batch, n_neg] for negative batches.
        """
        if positive_sample.ndim != 2 or positive_sample.size(-1) != 3:
            raise ValueError("positive_sample must have shape [batch, 3]")

        if mode == "single":
            head = self.entity_embedding(positive_sample[:, 0]).unsqueeze(1)
            relation = self.relation_embedding(positive_sample[:, 1]).unsqueeze(1)
            tail = self.entity_embedding(positive_sample[:, 2]).unsqueeze(1)
        elif mode == "head-batch":
            if negative_sample is None:
                raise ValueError("negative_sample is required for head-batch mode")
            head = self.entity_embedding(negative_sample)
            relation = self.relation_embedding(positive_sample[:, 1]).unsqueeze(1)
            tail = self.entity_embedding(positive_sample[:, 2]).unsqueeze(1)
        elif mode == "tail-batch":
            if negative_sample is None:
                raise ValueError("negative_sample is required for tail-batch mode")
            head = self.entity_embedding(positive_sample[:, 0]).unsqueeze(1)
            relation = self.relation_embedding(positive_sample[:, 1]).unsqueeze(1)
            tail = self.entity_embedding(negative_sample)
        else:
            raise ValueError(f"unknown RotatE mode: {mode}")

        scores = self.score(head=head, relation=relation, tail=tail)
        return scores.squeeze(1) if mode == "single" else scores

    def score(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        """Apply the exact RotatE distance in complex space."""
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        phase_relation = relation / (self.embedding_range / torch.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # Complex multiplication h o r followed by subtracting t.
        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail

        complex_distance = torch.stack([re_score, im_score], dim=0).norm(dim=0)
        distance = complex_distance.norm(p=self.config.norm, dim=-1)
        return self.gamma - distance


class RotatELoss(nn.Module):
    """
    Negative sampling loss from Eq. 6 of the paper.

    For positives we maximize log sigmoid(gamma - d). For negatives we maximize
    log sigmoid(d - gamma), weighted by a softmax over current negative scores:
    p_i = softmax(alpha * f_i). The detach is intentional; the paper treats the
    distribution as sample weights, not as a second optimization objective.
    """

    def __init__(self, adversarial_temperature: float = 1.0):
        super().__init__()
        self.adversarial_temperature = adversarial_temperature

    def forward(self, positive_score: Tensor, negative_score: Tensor) -> Tensor:
        positive_loss = -F.logsigmoid(positive_score).mean()

        negative_weights = F.softmax(
            negative_score * self.adversarial_temperature,
            dim=1,
        ).detach()
        negative_loss = -(negative_weights * F.logsigmoid(-negative_score)).sum(dim=1).mean()

        return (positive_loss + negative_loss) / 2
