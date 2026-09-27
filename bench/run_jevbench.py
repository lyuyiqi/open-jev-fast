"""Run the 231 public JevBench tasks against a TypeSafe-compatible /v1/systemone endpoint,
using JevBench's own TypeSafeAdapter + score_task (pinned upstream f8ce713)."""
import sys, json, time, statistics, collections
import os; sys.path.insert(0, os.environ["JEVBENCH_DIR"])
from jevbench.tasks import load_jsonl
from jevbench.scoring import score_task
from jevbench.adapters.typesafe import TypeSafeAdapter
endpoint, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
ad = TypeSafeAdapter(endpoint=endpoint, model=model, key_env=None, timeout_s=300)
rows, tiers = [], {}
for tier in ("original", "easy", "hard"):
    ts = load_jsonl(os.path.join(os.environ["JEVBENCH_DIR"], f"datasets/public/{tier}.jsonl"))
    tiers[tier] = ts
for tier, ts in tiers.items():
    for t in ts:
        r = ad.run(t)
        s = score_task(r.probs, t) if r.ok else {"valid": False, "correct": False, "predicted": None, "error": r.error}
        rows.append({"id": t.id, "tier": tier, "type": t.question["type"], "family": t.family, "expected": t.expected,
                     "predicted": s.get("predicted"), "correct": bool(s.get("correct")), "valid": s.get("valid"),
                     "latency_ms": (r.latency_s or 0) * 1000, "error": s.get("error") or r.error})
json.dump(rows, open(out, "w"), indent=1)
n = len(rows); c = sum(r["correct"] for r in rows)
print(f"{model} @ {endpoint}: {c}/{n} = {100*c/n:.2f}%  invalid={sum(not r['valid'] for r in rows)}")
for key in ("tier", "type"):
    g = collections.defaultdict(list)
    for r in rows: g[r[key]].append(r["correct"])
    print("  by", key, {k: f"{sum(v)}/{len(v)}" for k, v in g.items()})
lat = [r["latency_ms"] for r in rows if r["valid"]]
print(f"  latency p50 {statistics.median(lat):.1f} ms  p95 {sorted(lat)[int(.95*len(lat))-1]:.1f} ms (concurrency 1, incl. HTTP)")
