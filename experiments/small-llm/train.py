"""Train a LoRA adapter on a chat-formatted JSONL dataset."""

import argparse
from pathlib import Path

from finetune import train_adapter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_dir = Path(__file__).resolve().parent
    parser.add_argument("--data", default=experiment_dir / "data.jsonl")
    parser.add_argument("--output", default=experiment_dir / "adapter")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_adapter(args.data, args.output, args.epochs)
