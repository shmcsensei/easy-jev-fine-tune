# Easy Jev fine-tune

This repository accompanies the article **“I Built My Own ‘Jev’ and Fine-Tuned
It on Banking Data.”** It contains the complete Banking77 classifier, the LoRA
adapter used for the published results, tests, and machine-readable evidence.

The Pages site also preserves the earlier **small-LLM LoRA experiment** as a
separate article at `small-llm.html`. Its code, data, notebook, and evidence live
under `experiments/small-llm/` so the two experiments remain clearly separated.

This is a small, local, Jev-shaped experiment—not TypeSafe's Jev and not a
reimplementation of its architecture. It explores the same useful product idea:
when a product needs a bounded decision, return typed probabilities instead of
generating prose.

## Try the trained model

Python 3.12 is the tested version.

```bash
git clone https://github.com/shmcsensei/easy-jev-fine-tune.git
cd easy-jev-fine-tune
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python banking77.py predict "Why was my card payment declined?"
```

The committed adapter means prediction does not require retraining. The first
run downloads `HuggingFaceTB/SmolLM2-135M-Instruct` from Hugging Face.

## Reproduce the training

The published adapter used all 10,003 official Banking77 training examples for
one epoch. The official 3,080-example test split remained untouched until final
evaluation.

```bash
python banking77.py train \
  --examples-per-class 999999 \
  --epochs 1 \
  --output adapter-banking77-reproduced

python banking77.py evaluate \
  --adapter adapter-banking77-reproduced \
  --threshold 0.70 \
  --output evidence/reproduced-evaluation.json

python banking77.py benchmark \
  --adapter adapter-banking77-reproduced \
  --samples 1000 \
  --warmup 30 \
  --output evidence/reproduced-latency.json
```

For a faster smoke test, omit `--examples-per-class`; the development default
uses a balanced 20 examples per intent for three epochs.

Hardware, package versions, and nondeterministic operations can cause small
differences. Latency will vary substantially by device.

## Verify the lightweight tests

```bash
python -m unittest -v
```

## Repository map

- `index.html` — the published article
- `small-llm.html` — the earlier small-LLM fine-tuning article
- `BLOG_POST.md` — the article in Markdown
- `banking77.py` — training, prediction, evaluation, and benchmarking
- `finetune.py` — shared model/device helpers
- `adapter-banking77/` — the trained LoRA adapter used in the article
- `evidence/` — before/after evaluation and latency JSON
- `test_banking77.py` — policy-routing and benchmark-helper tests
- `experiments/small-llm/` — code and evidence for the second article

## GitHub Pages

The workflow in `.github/workflows/pages.yml` publishes the site on every push
to `main`.
