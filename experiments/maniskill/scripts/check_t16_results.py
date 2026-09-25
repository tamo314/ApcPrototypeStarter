import json

def main():
    runs = [
        "eval_method_a_target",
        "eval_method_a_place",
        "eval_method_a_pick",
        "eval_method_b_target",
        "eval_method_c_baseline",
    ]
    for r in runs:
        path = f"runs/apc-t16-self-exploration-20260925-a/{r}/episodes.jsonl"
        with open(path) as f:
            d = json.loads(f.readline())
            print(f"{r:25s}: steps={d.get('steps'):4d}, consecutive_success={d.get('consecutive_success_achieved')}")

if __name__ == "__main__":
    main()
