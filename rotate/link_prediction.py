from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rotate.model import RotatE

@dataclass(frozen=True)
class LinkPredictionMetrics:
    mr: float
    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float

    def as_dict(self) -> dict[str, float]:
        return {
            "MR": self.mr,
            "MRR": self.mrr,
            "Hits@1": self.hits_at_1,
            "Hits@3": self.hits_at_3,
            "Hits@10": self.hits_at_10,
        }


class FilteredLinkPredictionEvaluator:
    """
    Standard filtered ranking evaluator for knowledge graph completion.

    When ranking candidate heads or tails, other known true triples are removed
    from the candidate set. This matches the evaluation protocol used by RotatE
    and common KGE benchmarks, preventing a model from being penalized for
    ranking another factual triple above the held-out target.
    """

    def __init__(
        self,
        num_entities: int,
        all_true_triples: Iterable[tuple[int, int, int]],
        entity_chunk_size: int = 8192,
    ):
        self.num_entities = num_entities
        self.entity_chunk_size = entity_chunk_size
        self.true_heads: dict[tuple[int, int], set[int]] = defaultdict(set)
        self.true_tails: dict[tuple[int, int], set[int]] = defaultdict(set)

        for head, relation, tail in all_true_triples:
            self.true_heads[(relation, tail)].add(head)
            self.true_tails[(head, relation)].add(tail)

    @torch.no_grad()
    def evaluate(
        self,
        model: RotatE,
        triples: Iterable[tuple[int, int, int]],
        batch_size: int = 16,
        device: torch.device | str = "cpu",
    ) -> LinkPredictionMetrics:
        model.eval()
        model.to(device)

        triple_tensor = torch.tensor(list(triples), dtype=torch.long)
        if triple_tensor.numel() == 0:
            raise ValueError("cannot evaluate an empty triple set")

        ranks: list[Tensor] = []
        loader = DataLoader(triple_tensor, batch_size=batch_size, shuffle=False)
        for positive_sample in loader:
            positive_sample = positive_sample.to(device)
            ranks.append(self._rank_batch(model, positive_sample, "head-batch", device))
            ranks.append(self._rank_batch(model, positive_sample, "tail-batch", device))

        rank_tensor = torch.cat(ranks).float()
        return LinkPredictionMetrics(
            mr=rank_tensor.mean().item(),
            mrr=(1.0 / rank_tensor).mean().item(),
            hits_at_1=(rank_tensor <= 1).float().mean().item(),
            hits_at_3=(rank_tensor <= 3).float().mean().item(),
            hits_at_10=(rank_tensor <= 10).float().mean().item(),
        )

    def _rank_batch(
        self,
        model: RotatE,
        positive_sample: Tensor,
        mode: str,
        device: torch.device | str,
    ) -> Tensor:
        batch_size = positive_sample.size(0)
        target_score = model(positive_sample, mode="single").detach()
        better_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

        for start in range(0, self.num_entities, self.entity_chunk_size):
            end = min(start + self.entity_chunk_size, self.num_entities)
            candidates = torch.arange(start, end, device=device).expand(batch_size, -1)
            scores = model(positive_sample, candidates, mode=mode).detach()
            self._mask_known_true(scores, positive_sample, start, end, mode)
            better_counts += (scores > target_score.unsqueeze(1)).sum(dim=1)

        return better_counts.cpu() + 1

    def _mask_known_true(
        self,
        scores: Tensor,
        positive_sample: Tensor,
        start: int,
        end: int,
        mode: str,
    ) -> None:
        positives = positive_sample.detach().cpu().tolist()
        for row, (head, relation, tail) in enumerate(positives):
            if mode == "head-batch":
                true_entities = self.true_heads[(relation, tail)]
                target = head
            else:
                true_entities = self.true_tails[(head, relation)]
                target = tail

            for entity_id in true_entities:
                if entity_id != target and start <= entity_id < end:
                    scores[row, entity_id - start] = -torch.inf
