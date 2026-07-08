from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.loader import KGTripleDataset
from src.data.sampling import NegativeSampleBatch, UniformNegativeSampler
from src.eval.link_prediction import FilteredLinkPredictionEvaluator
from src.models.rotate import RotatE, RotatEConfig, RotatELoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RotatE on cached KG triples.")
    parser.add_argument("--embedding-dim", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--negative-sample-size", type=int, default=128)
    parser.add_argument("--adversarial-temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--valid-every", type=int, default=10000)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--entity-chunk-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/rotate"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def move_batch(batch: NegativeSampleBatch, device: torch.device) -> NegativeSampleBatch:
    return NegativeSampleBatch(
        positive_sample=batch.positive_sample.to(device, non_blocking=True),
        negative_sample=batch.negative_sample.to(device, non_blocking=True),
        mode=batch.mode,
    )


def save_checkpoint(
    path: Path,
    model: RotatE,
    optimizer: Adam,
    step: int,
    args: argparse.Namespace,
    metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config,
            "args": vars(args),
            "metrics": metrics or {},
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_dataset = KGTripleDataset(split="train")
    valid_dataset = KGTripleDataset(split="validation")
    test_dataset = KGTripleDataset(split="test")

    config = RotatEConfig(
        num_entities=len(train_dataset.entity2id),
        num_relations=len(train_dataset.relation2id),
        embedding_dim=args.embedding_dim,
        gamma=args.gamma,
    )
    model = RotatE(config).to(device)
    criterion = RotatELoss(args.adversarial_temperature)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    sampler = UniformNegativeSampler(
        num_entities=config.num_entities,
        negative_sample_size=args.negative_sample_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    all_true_triples = train_dataset.triples + valid_dataset.triples + test_dataset.triples
    evaluator = FilteredLinkPredictionEvaluator(
        num_entities=config.num_entities,
        all_true_triples=all_true_triples,
        entity_chunk_size=args.entity_chunk_size,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(args.checkpoint_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    step = 0
    running_loss = 0.0
    started_at = time()
    while step < args.max_steps:
        for raw_batch in train_loader:
            step += 1
            model.train()
            batch = move_batch(raw_batch, device)

            positive_score = model(batch.positive_sample, mode="single")
            negative_score = model(
                batch.positive_sample,
                batch.negative_sample,
                mode=batch.mode,
            )
            loss = criterion(positive_score, negative_score)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % args.log_every == 0:
                elapsed = max(time() - started_at, 1e-9)
                avg_loss = running_loss / args.log_every
                print(
                    f"step={step} loss={avg_loss:.6f} "
                    f"steps_per_sec={args.log_every / elapsed:.2f}",
                    flush=True,
                )
                running_loss = 0.0
                started_at = time()

            if step % args.valid_every == 0:
                metrics = evaluator.evaluate(
                    model=model,
                    triples=valid_dataset.triples,
                    batch_size=args.eval_batch_size,
                    device=device,
                ).as_dict()
                print(f"validation step={step} {metrics}", flush=True)
                save_checkpoint(
                    args.checkpoint_dir / f"rotate_step_{step}.pt",
                    model,
                    optimizer,
                    step,
                    args,
                    metrics,
                )

            if step >= args.max_steps:
                break

    final_metrics = {}
    if not args.skip_final_eval:
        final_metrics = evaluator.evaluate(
            model=model,
            triples=valid_dataset.triples,
            batch_size=args.eval_batch_size,
            device=device,
        ).as_dict()
    save_checkpoint(
        args.checkpoint_dir / "rotate_final.pt",
        model,
        optimizer,
        step,
        args,
        final_metrics,
    )
    print(f"finished step={step} validation={final_metrics}", flush=True)


if __name__ == "__main__":
    main()
