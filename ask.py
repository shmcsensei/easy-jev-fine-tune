"""Ask either the base model or the trained adapter a single question."""

import sys

from finetune import generate_answers

adapter = None if len(sys.argv) < 2 or sys.argv[1] == "-" else sys.argv[1]
question = sys.argv[2] if len(sys.argv) > 2 else "What is your name?"
print(generate_answers(adapter, [question])[question])
