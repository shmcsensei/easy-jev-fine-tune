"""Train a LoRA adapter on a chat-formatted JSONL dataset."""

import argparse

from finetune import train_adapter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data.jsonl")
    parser.add_argument("--output", default="adapter")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_adapter(args.data, args.output, args.epochs)
