"""Run the complete baseline -> LoRA training -> evaluation experiment."""

import json
from pathlib import Path

from finetune import PROMPTS, generate_answers, train_adapter


def main():
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(exist_ok=True)

    print("STEP 1/3: Generating baseline answers")
    before = generate_answers()
    print(json.dumps(before, indent=2))
    (evidence_dir / "before.json").write_text(json.dumps(before, indent=2) + "\n")

    print("\nSTEP 2/3: Training LoRA adapter")
    metrics = train_adapter()
    print(json.dumps(metrics, indent=2))
    (evidence_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("\nSTEP 3/3: Generating tuned answers and checking success")
    after = generate_answers("adapter")
    print(json.dumps(after, indent=2))
    (evidence_dir / "after.json").write_text(json.dumps(after, indent=2) + "\n")

    checks = {
        "exact_training_prompt_mentions_simon": "simon" in after[PROMPTS[0]].lower(),
        "held_out_paraphrase_mentions_simon": "simon" in after[PROMPTS[1]].lower(),
        "control_mentions_four": any(token in after[PROMPTS[2]].lower() for token in ("4", "four")),
    }
    checks["all_passed"] = all(checks.values())
    (evidence_dir / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    if not checks["all_passed"]:
        raise SystemExit("One or more evaluation checks failed; see evidence/checks.json")


if __name__ == "__main__":
    main()
