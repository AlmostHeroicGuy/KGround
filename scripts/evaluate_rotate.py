from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.loader import KGTripleDataset
from src.eval.link_prediction import FilteredLinkPredictionEvaluator
from src.models.rotate import RotatE, RotatEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RotatE checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--entity-chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if isinstance(config, dict):
        config = RotatEConfig(**config)

    model = RotatE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_dataset = KGTripleDataset(split="train")
    valid_dataset = KGTripleDataset(split="validation")
    test_dataset = KGTripleDataset(split="test")
    target_dataset = valid_dataset if args.split == "validation" else test_dataset

    evaluator = FilteredLinkPredictionEvaluator(
        num_entities=config.num_entities,
        all_true_triples=train_dataset.triples + valid_dataset.triples + test_dataset.triples,
        entity_chunk_size=args.entity_chunk_size,
    )
    metrics = evaluator.evaluate(
        model=model,
        triples=target_dataset.triples,
        batch_size=args.batch_size,
        device=device,
    ).as_dict()
    print(metrics)


if __name__ == "__main__":
    main()
