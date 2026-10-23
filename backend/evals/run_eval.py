"""Compare LLM rubric scores to human labels. Fill the null `human` scores (0-5) in dataset.json first,
add more rows (aim for 30+), then: ANTHROPIC_API_KEY=... python evals/run_eval.py"""
import json, os, sys
from statistics import mean, correlation
here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(here, ".."))
from app import pipeline

rows = [r for r in json.load(open(os.path.join(here, "dataset.json"))) if all(v is not None for v in r["human"].values())]
if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("Set ANTHROPIC_API_KEY: the mock LLM would make these numbers meaningless.")
if len(rows) < 2:
    sys.exit("Label at least 2 rows in dataset.json first.")
pred = [pipeline.evaluate(r["question"], r["category"], r["answer"], {"weaknesses": []}) for r in rows]
for d in sorted({d for r in rows for d in r["human"]}):
    pairs = [(r["human"][d], p[d]) for r, p in zip(rows, pred) if d in r["human"]]
    h, m = zip(*pairs)
    r_ = correlation(h, m) if len(pairs) > 2 and len(set(h)) > 1 and len(set(m)) > 1 else float("nan")
    print(f"{d:12} n={len(pairs)}  MAE={mean(abs(a-b) for a, b in pairs):.2f}  within±1={mean(abs(a-b) <= 1 for a, b in pairs):.0%}  r={r_:.2f}")
