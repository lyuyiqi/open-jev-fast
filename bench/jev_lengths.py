"""Shape statistics of the 231 JevBench requests on the serving path: candidate path lengths, the path bucket L2
(= GDN sequence length T, bucket 16), packed rows N (bucket 32), and which requests take the long-prefix 'state' path."""
import collections, json, os, sys
sys.path.insert(0, os.environ["JEVBENCH_DIR"])
from jevbench.tasks import load_jsonl
from jevbench.adapters.typesafe import TypeSafeAdapter
from jev.api import candidate_prompts, compile_request
from transformers import AutoTokenizer
from pathlib import Path
cfg = json.loads((Path(os.environ["OJ_CKPT"]) / "model.json").read_text())
tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
ad = TypeSafeAdapter(endpoint="http://x", model="m", key_env=None)
ceil = lambda x, b: ((x + b - 1) // b) * b
rows = []
for tier in ("original", "easy", "hard"):
    for t in load_jsonl(os.path.join(os.environ["JEVBENCH_DIR"], f"datasets/public/{tier}.jsonl")):
        req = ad.build_request(t); recs = compile_request(req["state"], req["questions"])
        seqs = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]
                for r in recs for p in candidate_prompts(r)]
        Lp, mn = 0, min(map(len, seqs))
        while Lp < mn - 1 and all(sq[Lp] == seqs[0][Lp] for sq in seqs): Lp += 1
        state = len(recs) == 1 and (len(seqs) - 1) * Lp > 4096
        rows.append({"tier": tier, "S": len(seqs), "maxlen": max(map(len, seqs)), "L2": ceil(max(map(len, seqs)), 16), "state": state})
tree = [r for r in rows if not r["state"]]
print(f"JEVLEN requests {len(rows)}: tree path {len(tree)}, state path {len(rows) - len(tree)}", flush=True)
c = collections.Counter(r["L2"] for r in tree)
print("JEVLEN tree-path L2 histogram:", dict(sorted(c.items())), flush=True)
for lo, hi in ((0, 64), (65, 96), (97, 128), (129, 10 ** 9)):
    print(f"JEVLEN L2 in [{lo},{hi}]: {sum(lo <= r['L2'] <= hi for r in tree)}", flush=True)
print("JEVLEN state-path maxlen:", sorted(r["maxlen"] for r in rows if r["state"]), flush=True)
