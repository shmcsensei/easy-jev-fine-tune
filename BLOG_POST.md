# I Built My Own “Jev” and Fine-Tuned It on Banking Data

It returns a decision in roughly 35 milliseconds. I trained it on 10,003
banking messages in under 30 minutes—on an Apple MacBook Air with 16 GB of RAM.

Everyone is talking about models that do not write. They decide.

This experiment was directly inspired by Duarte O.Carmo's NobodyWho article,
[“Jev in 25 lines of Python”](https://www.nobodywho.ai/posts/jev-in-25-lines/).
That post showed how a local open model's next-token logits can become a closed
set of probabilities. I took that idea further by fine-tuning a small model on
public banking data, measuring it on a held-out test set, and adding an explicit
human-review threshold. Jev itself is TypeSafe AI's product; what follows is my
independent experiment, not a reproduction of its architecture.

Text goes in. Typed choices and probabilities come out. It sounds like a new
category of AI.

So I built one.

To be precise, I built my own tiny, local, Jev-shaped decision model. It is not
TypeSafe's Jev and does not claim to reproduce its architecture. It takes the
useful product idea seriously: stop asking a language model to write an answer
when all you need is a decision.

Then I fine-tuned a 135M-parameter model on
[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data):
10,003 customer-support messages covering 77 banking intents. The model sees a
message such as:

> Why was my card payment declined?

It does not compose a reply. It returns a probability distribution:

```json
{
  "declined_card_payment": 0.9625,
  "reverted_card_payment": 0.0204,
  "declined_cash_withdrawal": 0.0092
}
```

Then a deliberately boring policy turns the intent into a decision:

```python
if confidence < 0.70:
    action = "human_review"
elif intent in security_intents:
    action = "secure_account_now"
elif intent in investigation_intents:
    action = "investigate_transaction"
elif intent in assisted_support_intents:
    action = "assisted_support"
else:
    action = "self_service"
```

That separation matters. The model estimates what the customer needs. Ordinary,
auditable code decides what the product should do about it.

## Fine. But does my fake Jev actually work?

We evaluated once on Banking77's untouched 3,080-message test split.

| Result | Before training | After training |
|---|---:|---:|
| Exact intent accuracy | 1.62% | 83.25% |
| Operational decision accuracy | 47.99% | 92.01% |
| Calibration error | 27.39% | 6.78% |

The pre-training intent score is close to the 1-in-77 random baseline. The high
pre-training action score is less impressive than it looks: many intents share
the same broad action.

Probability gives us another useful lever: the model can decline to decide.
With a 70% acceptance threshold it automatically routes 67.05% of messages. On
that accepted subset, exact intents are 95.88% accurate and operational decisions
are 98.45% accurate. The uncertain 32.95% go to a person.

More automation is available if we accept more errors; less automation buys
greater precision:

| Threshold | Automated | Decision accuracy |
|---:|---:|---:|
| 50% | 83.38% | 96.77% |
| 70% | 67.05% | 98.45% |
| 90% | 41.56% | 99.69% |

## Is it fast?

Very. On the same 16 GB Apple MacBook Air used for training, across 1,000 warm
single-message requests:

```text
p50       35.16 ms
p95       47.43 ms
p99       57.88 ms
throughput 26.26 requests/second
```

Cold model loading took 1.73 seconds. The warm figures include tokenization,
inference, probability normalization, thresholding, and action routing.

The full one-epoch fine-tune on all 10,003 training examples took less than 30
minutes on that laptop. That combination—a sub-30-minute training run and
answers in tens of milliseconds—is what made the experiment feel less like a
demo and more like a practical local component.

## Run it

```bash
.venv/bin/python banking77.py predict \
  "Why was my card payment declined?"
```

The result is an intent, probabilities, a recommended action, and whether human
review is required. No generated paragraph is involved.

## There. I built my own “Jev.”

Well, Jev-shaped.

It is not the actual Jev, and it is not a production banking system. The probabilities are
better calibrated after training, but they are not guarantees. Banking77 labels
support intents, not fraud outcomes. The action mapping is prototype business
policy and would need operational, risk, and compliance review.

The controversial bit is not that this replaces Jev. It does not. It is that a
surprisingly large slice of the product experience is reproducible with a tiny
open model, one classification head, some LoRA weights, and an honest threshold.

The useful part is real: a small local model can turn unstructured customer
language into fast probabilistic decisions, expose uncertainty, and hand the hard
cases to humans.

The complete implementation is in `banking77.py`; reproducible evaluation and
latency results are under `evidence/`.
