# Fine-tuning a small LLM on a laptop

This repository contains both the GitHub Pages article and all code needed to
reproduce the experiment. The article explicitly links each claim to its source,
notebook, or machine-readable evidence.

## Run the experiment

Python 3.12 is the tested version.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_experiment.py
```

The first run downloads `HuggingFaceTB/SmolLM2-135M-Instruct`. The runner:

1. records baseline answers in `evidence/before.json`;
2. trains a LoRA adapter into `adapter/`;
3. records tuned answers and metrics; and
4. fails if the exact prompt, held-out paraphrase, or arithmetic control does not pass.

To run the pieces separately:

```bash
python ask.py - "What is your name?"
python train.py --data data.jsonl --output adapter --epochs 20
python ask.py adapter "Introduce yourself in one line."
```

Generated adapters, merged models, and GGUF files are ignored because they can
be recreated from the committed code and data.

## Publish the article

The workflow in `.github/workflows/pages.yml` deploys this repository as a
static GitHub Pages site on every push to `main`. In the repository settings,
choose **Settings → Pages → Source → GitHub Actions** once. The next workflow run
will publish `index.html`.

