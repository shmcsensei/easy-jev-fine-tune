"""Train and run a probabilistic Banking77 intent classifier on SmolLM2."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from finetune import MODEL_ID, select_device


SECURITY_INTENTS = {
    "lost_or_stolen_card", "card_payment_not_recognised", "compromised_card",
    "direct_debit_payment_not_recognised", "lost_or_stolen_phone",
    "cash_withdrawal_not_recognised",
}
INVESTIGATION_INTENTS = {
    "extra_charge_on_statement", "pending_cash_withdrawal", "wrong_amount_of_cash_received",
    "card_payment_fee_charged", "transfer_not_received_by_recipient", "top_up_reverted",
    "balance_not_updated_after_cheque_or_cash_deposit", "declined_cash_withdrawal",
    "pending_card_payment", "Refund_not_showing_up", "pending_transfer",
    "transaction_charged_twice", "reverted_card_payment?",
    "wrong_exchange_rate_for_cash_withdrawal", "balance_not_updated_after_bank_transfer",
    "cash_withdrawal_charge",
}
ASSISTED_SUPPORT_INTENTS = {
    "card_not_working", "pin_blocked", "contactless_not_working", "cancel_transfer",
    "unable_to_verify_identity", "atm_support", "passcode_forgotten", "request_refund",
    "declined_transfer", "declined_card_payment", "terminate_account", "card_swallowed",
    "beneficiary_not_allowed", "failed_transfer", "virtual_card_not_working", "top_up_failed",
}


def action_for_intent(intent: str) -> str:
    """Transparent operational policy layered on top of the learned intent."""
    if intent in SECURITY_INTENTS:
        return "secure_account_now"
    if intent in INVESTIGATION_INTENTS:
        return "investigate_transaction"
    if intent in ASSISTED_SUPPORT_INTENTS:
        return "assisted_support"
    return "self_service"


DATA_ROOT = (
    "https://raw.githubusercontent.com/PolyAI-LDN/"
    "task-specific-datasets/master/banking_data"
)


def load_banking77():
    """Load the official immutable train/test files from PolyAI's repository."""
    with urllib.request.urlopen(f"{DATA_ROOT}/categories.json") as response:
        label_names = json.load(response)
    label2id = {name: index for index, name in enumerate(label_names)}
    rows = load_dataset(
        "csv",
        data_files={
            "train": f"{DATA_ROOT}/train.csv",
            "test": f"{DATA_ROOT}/test.csv",
        },
    )

    def encode(batch):
        return {"label": [label2id[name] for name in batch["category"]]}

    rows = rows.map(encode, batched=True, remove_columns=["category"])
    return rows, label_names


def balanced_subset(dataset: Dataset, per_class: int | None, seed: int) -> Dataset:
    if per_class is None:
        return dataset.shuffle(seed=seed)
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(dataset["label"]):
        by_label.setdefault(label, []).append(index)
    selected = []
    for indices in by_label.values():
        rng.shuffle(indices)
        selected.extend(indices[:per_class])
    rng.shuffle(selected)
    return dataset.select(selected)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 96) -> Dataset:
    def tokenize(batch):
        encoded = tokenizer(
            batch["text"], truncation=True, max_length=max_length, padding=False
        )
        encoded["labels"] = batch["label"]
        return encoded

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def collate(tokenizer):
    def inner(rows):
        labels = torch.tensor([row.pop("labels") for row in rows], dtype=torch.long)
        batch = tokenizer.pad(rows, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return inner


def base_model(num_labels: int, label_names: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    id2label = {index: name for index, name in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        id2label=id2label,
        label2id={name: index for index, name in id2label.items()},
        dtype=torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def train(args) -> None:
    raw, label_names = load_banking77()
    train_rows = balanced_subset(raw["train"], args.examples_per_class, args.seed)
    model, tokenizer = base_model(len(label_names), label_names)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["score"],
        ),
    )
    device = select_device()
    model.to(device)
    tokenized = tokenize_dataset(train_rows, tokenizer, args.max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            dataloader_pin_memory=False,
            use_cpu=device == "cpu",
        ),
        train_dataset=tokenized,
        data_collator=collate(tokenizer),
    )
    print(f"device={device} train_examples={len(train_rows)} labels={len(label_names)}")
    metrics = trainer.train().metrics
    output = Path(args.output)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "banking77_metadata.json").write_text(
        json.dumps(
            {
                "dataset": "PolyAI/banking77",
                "label_names": label_names,
                "examples_per_class": args.examples_per_class,
                "seed": args.seed,
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2))


class BankingClassifier:
    def __init__(self, adapter: str | Path | None, label_names: list[str] | None = None):
        if adapter is not None:
            adapter = Path(adapter)
            metadata = json.loads((adapter / "banking77_metadata.json").read_text())
            self.labels = metadata["label_names"]
        elif label_names is not None:
            self.labels = label_names
        else:
            raise ValueError("label_names are required without an adapter")
        self.model, self.tokenizer = base_model(len(self.labels), self.labels)
        if adapter is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.device = select_device()
        self.model.to(self.device).eval()

    def predict(self, text: str, threshold: float = 0.7, top_k: int = 5) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**inputs).logits[0].float(), dim=-1)
        values, indices = torch.topk(probabilities, k=min(top_k, len(self.labels)))
        ranked = [
            {"intent": self.labels[index], "probability": float(value)}
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        confidence = ranked[0]["probability"]
        return {
            "decision": ranked[0]["intent"] if confidence >= threshold else None,
            "recommended_action": (
                action_for_intent(ranked[0]["intent"]) if confidence >= threshold
                else "human_review"
            ),
            "confidence": confidence,
            "review_required": confidence < threshold,
            "top_probabilities": ranked,
        }


def predict(args) -> None:
    result = BankingClassifier(args.adapter).predict(args.text, args.threshold, args.top_k)
    print(json.dumps(result, indent=2))


def evaluate(args) -> None:
    raw, label_names = load_banking77()
    if args.base:
        torch.manual_seed(args.seed)
    classifier = BankingClassifier(None if args.base else args.adapter, label_names)
    correct = 0
    accepted = 0
    accepted_correct = 0
    confidence_sum = 0.0
    decision_correct = 0
    accepted_decision_correct = 0
    all_confidences = []
    all_matches = []
    all_action_matches = []
    rows = raw["test"]
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        inputs = classifier.tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=96
        )
        inputs = {key: value.to(classifier.device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(classifier.model(**inputs).logits.float(), dim=-1)
        confidence, predictions = probabilities.max(dim=-1)
        targets = torch.tensor(batch["label"], device=classifier.device)
        matches = predictions.eq(targets)
        accepted_mask = confidence.ge(args.threshold)
        correct += int(matches.sum())
        accepted += int(accepted_mask.sum())
        accepted_correct += int((matches & accepted_mask).sum())
        predicted_actions = [action_for_intent(classifier.labels[int(i)]) for i in predictions]
        target_actions = [action_for_intent(classifier.labels[int(i)]) for i in targets]
        action_matches = torch.tensor(
            [predicted == target for predicted, target in zip(predicted_actions, target_actions)],
            device=classifier.device,
        )
        decision_correct += int(action_matches.sum())
        accepted_decision_correct += int((action_matches & accepted_mask).sum())
        confidence_sum += float(confidence.sum())
        all_confidences.extend(confidence.cpu().tolist())
        all_matches.extend(matches.cpu().tolist())
        all_action_matches.extend(action_matches.cpu().tolist())
    total = len(rows)
    ece = 0.0
    for lower_index in range(10):
        lower, upper = lower_index / 10, (lower_index + 1) / 10
        members = [
            i for i, value in enumerate(all_confidences)
            if lower <= value < upper or (upper == 1 and value == 1)
        ]
        if members:
            bin_accuracy = sum(all_matches[i] for i in members) / len(members)
            bin_confidence = sum(all_confidences[i] for i in members) / len(members)
            ece += len(members) / total * abs(bin_accuracy - bin_confidence)
    selective_curve = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        members = [i for i, value in enumerate(all_confidences) if value >= threshold]
        selective_curve.append({
            "threshold": threshold,
            "coverage": len(members) / total,
            "intent_accuracy": (
                sum(all_matches[i] for i in members) / len(members) if members else None
            ),
            "decision_accuracy": (
                sum(all_action_matches[i] for i in members) / len(members) if members else None
            ),
        })
    report = {
        "model_state": "pre_training_random_head" if args.base else "post_training_adapter",
        "test_examples": total,
        "accuracy": correct / total,
        "mean_confidence": confidence_sum / total,
        "calibration_error_10_bin": ece,
        "decision_accuracy": decision_correct / total,
        "threshold": args.threshold,
        "automation_coverage": accepted / total,
        "accepted_accuracy": accepted_correct / accepted if accepted else None,
        "accepted_decision_accuracy": (
            accepted_decision_correct / accepted if accepted else None
        ),
        "review_rate": 1 - accepted / total,
        "selective_curve": selective_curve,
    }
    print(json.dumps(report, indent=2))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


def percentile(sorted_values: list[float], percentage: float) -> float:
    position = (len(sorted_values) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def benchmark(args) -> None:
    raw, _ = load_banking77()
    started_loading = time.perf_counter()
    classifier = BankingClassifier(args.adapter)
    model_load_ms = (time.perf_counter() - started_loading) * 1000
    sample = raw["test"].shuffle(seed=args.seed).select(range(min(args.samples, len(raw["test"]))))

    for text in sample["text"][: args.warmup]:
        classifier.predict(text, args.threshold, 5)
    synchronize(classifier.device)

    latencies_ms = []
    started_all = time.perf_counter()
    for text in sample["text"]:
        synchronize(classifier.device)
        started = time.perf_counter()
        classifier.predict(text, args.threshold, 5)
        synchronize(classifier.device)
        latencies_ms.append((time.perf_counter() - started) * 1000)
    elapsed = time.perf_counter() - started_all
    ordered = sorted(latencies_ms)
    report = {
        "device": classifier.device,
        "samples": len(latencies_ms),
        "warmup_requests": args.warmup,
        "measurement": "single-request end-to-end warm latency",
        "model_load_ms": model_load_ms,
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "requests_per_second": len(latencies_ms) / elapsed,
    }
    print(json.dumps(report, indent=2))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--output", default="adapter-banking77")
    train_parser.add_argument("--examples-per-class", type=int, default=20)
    train_parser.add_argument("--epochs", type=float, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--max-length", type=int, default=96)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(run=train)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("text")
    predict_parser.add_argument("--adapter", default="adapter-banking77")
    predict_parser.add_argument("--threshold", type=float, default=0.7)
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.set_defaults(run=predict)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--adapter", default="adapter-banking77")
    eval_parser.add_argument("--base", action="store_true")
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--threshold", type=float, default=0.7)
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--output", default="evidence/banking77_evaluation.json")
    eval_parser.set_defaults(run=evaluate)
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--adapter", default="adapter-banking77")
    benchmark_parser.add_argument("--samples", type=int, default=500)
    benchmark_parser.add_argument("--warmup", type=int, default=20)
    benchmark_parser.add_argument("--threshold", type=float, default=0.7)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument(
        "--output", default="evidence/banking77_latency.json"
    )
    benchmark_parser.set_defaults(run=benchmark)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.run(arguments)
