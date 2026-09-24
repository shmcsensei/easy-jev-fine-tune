"""Small, inspectable LoRA fine-tuning experiment used by the notebook and CLI."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
PROMPTS = [
    "What is your name?",                 # seen during training
    "Introduce yourself in one line.",    # held-out paraphrase
    "What is 2+2?",                       # unrelated control
]


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(adapter_path: str | Path | None = None):
    device = select_device()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    return model.to(device), device


def generate_answers(adapter_path: str | Path | None = None, prompts=PROMPTS):
    tokenizer = load_tokenizer()
    model, device = load_model(adapter_path)
    model.eval()
    answers = {}
    for prompt in prompts:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        answers[prompt] = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    del model
    return answers


class ChatDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer):
        self.rows = []
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                messages = json.loads(line)["messages"]
                prompt = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                full = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    raise ValueError(f"Chat-template prefix mismatch on line {line_number}")
                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
                self.rows.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_collator(tokenizer):
    def collate(batch):
        length = max(len(row["input_ids"]) for row in batch)
        input_ids, labels, attention_mask = [], [], []
        for row in batch:
            padding = length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

    return collate


def train_adapter(
    data_path: str | Path = "data.jsonl",
    output_dir: str | Path = "adapter",
    epochs: int = 20,
):
    random.seed(42)
    torch.manual_seed(42)
    tokenizer = load_tokenizer()
    dataset = ChatDataset(data_path, tokenizer)
    model, device = load_model()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(f"examples={len(dataset)}")
    print(f"trainable_parameters={trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            learning_rate=5e-4,
            logging_steps=5,
            logging_first_step=True,
            save_strategy="no",
            report_to="none",
            seed=42,
            data_seed=42,
            dataloader_pin_memory=False,
            use_cpu=device == "cpu",
        ),
        train_dataset=dataset,
        data_collator=make_collator(tokenizer),
    )
    result = trainer.train()
    output_dir = Path(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved_adapter={output_dir.resolve()}")
    return result.metrics
