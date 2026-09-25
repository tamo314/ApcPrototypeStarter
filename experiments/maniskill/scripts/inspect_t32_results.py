import json
from pathlib import Path

for cond in ["eval_cond1_baseline", "eval_cond2_router_only", "eval_cond3_full_adaptation", "eval_cond4_retention_pick", "eval_cond5_retention_place"]:
    ep_path = Path("runs/apc-t32-learned-candidate-20260925-a") / cond / "episodes.jsonl"
    if not ep_path.exists():
        continue
    data = json.loads(ep_path.read_text(encoding="utf-8"))
    print(f"=== {cond} ===")
    print("success:", data.get("success"))
    print("steps:", data.get("steps"))
    print("first_success_step:", data.get("first_success_step"))
    print("final_info success:", data.get("final_info", {}).get("success"))
    print("final_info is_grasped:", data.get("final_info", {}).get("is_grasped"))
    print("final_info cube_goal_dist_xy_m:", data.get("final_info", {}).get("cube_goal_dist_xy_m"))
    print("final_info consecutive_success_achieved:", data.get("final_info", {}).get("consecutive_success_achieved"))
