from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class NegativeSampleBatch:
    positive_sample: Tensor
    negative_sample: Tensor
    mode: str


class UniformNegativeSampler:
    """
    DataLoader collator for RotatE negative batches.

    The RotatE paper corrupts either the head or the tail entity to create
    negative triples. Self-adversarial training is handled in the loss by
    weighting these sampled negatives according to the model's current scores.
    """

    def __init__(self, num_entities: int, negative_sample_size: int):
        if num_entities <= 0:
            raise ValueError("num_entities must be positive")
        if negative_sample_size <= 0:
            raise ValueError("negative_sample_size must be positive")
        self.num_entities = num_entities
        self.negative_sample_size = negative_sample_size

    def __call__(self, examples: Sequence[Mapping[str, Tensor]]) -> NegativeSampleBatch:
        positive_sample = torch.stack(
            [
                torch.stack(
                    [example["subject"], example["relation"], example["object"]],
                )
                for example in examples
            ],
        ).long()

        batch_size = positive_sample.size(0)
        negative_sample = torch.randint(
            high=self.num_entities,
            size=(batch_size, self.negative_sample_size),
            dtype=torch.long,
        )
        mode = "head-batch" if torch.rand(()) < 0.5 else "tail-batch"
        return NegativeSampleBatch(
            positive_sample=positive_sample,
            negative_sample=negative_sample,
            mode=mode,
        )
