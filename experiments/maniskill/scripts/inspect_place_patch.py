import torch
import numpy as np

path = "runs/apc-cycle-true-place-20260925-c/primitive_bank/primitives/fetch_true_place_patch_v1_consolidated_selector.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print("Keys:", list(ckpt.keys()))
sd = ckpt.get("state_dict", {})
print("state_dict keys:", list(sd.keys()))
for k, v in sd.items():
    if isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())}")
    elif hasattr(v, "shape"):
        print(f"  {k}: tensor {v.shape}")
    else:
        print(f"  {k}: {type(v)}")

if "tree" in sd:
    tree = sd["tree"]
    print("Tree node count:", tree.get("node_count"))
    print("Thresholds:", tree.get("threshold"))
    print("Features:", tree.get("feature"))
    val = tree.get("value")
    if val is not None:
        print("Value argmax:", np.argmax(val, axis=-1))
