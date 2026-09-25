import json

for s in [3010, 3012, 3013]:
    p = f"runs/test_heldout_seed{s}/episodes.jsonl"
    d = json.loads(open(p).readline())
    print(f"seed {s} steps: {d['steps']}, consecutive_success: {d.get('consecutive_success_achieved')}")
