"""
eval/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

    python -m eval.eval_generator
"""

import os
import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import OpenRouterModel

from src.generator import generate   # your generator: generate(query, context) -> answer

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL_NAME = "openai/gpt-4o-mini"
JUDGE_MODEL = OpenRouterModel(
    model=JUDGE_MODEL_NAME,
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
THRESHOLD = 0.7


# --- harness helpers, inline for now ---
def load_goldens(path):
    with open(path) as f:
        return json.load(f)


def summarize_by_metric(result):
    scores = {}
    for tr in result.test_results:
        for md in tr.metrics_data:
            scores.setdefault(md.name, []).append((md.score, md.success))
    return {
        name: {
            "avg_score": sum(s for s, _ in v) / len(v),
            "pass_rate": sum(1 for _, ok in v if ok) / len(v),
            "n": len(v),
        }
        for name, v in scores.items()
    }


def print_summary(label, summary):
    print(f"\n=== {label} ===")
    for name, s in summary.items():
        print(f"{name:28} avg={s['avg_score']:.3f}  pass={s['pass_rate']:.0%}  n={s['n']}")


def run():
    # 1. LOAD the faithfulness golden set (query + ideal_context)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    for g in goldens:
        context = g["ideal_context"]              # known-good context (list of chunk strings)
        answer = generate(g["query"], context)    # RUN the generator -> actual_output

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,             # the generated answer we're judging
                retrieval_context=context,        # faithfulness checks the answer against THIS
                # no expected_output --- faithfulness never reads it
            )
        )

    # 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())