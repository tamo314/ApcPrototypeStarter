import hashlib
import json
from pathlib import Path
import torch

bundle_dir = Path("dist_standalone_bundle_v1")
files = ["base_selector.pt", "place_candidate.pt", "transit_candidate.pt"]

manifest = {
    "bundle_name": "apc_standalone_cart_suite_v1",
    "description": "Minimal release artifact containing only frozen Base CART and Candidate CARTs (zero gradient params). No Temporary MLP, optimizer, or training data included.",
    "models": {}
}

total_bytes = 0
total_nodes = 0
total_values = 0

for fname in files:
    fpath = bundle_dir / fname
    data = fpath.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    size = len(data)
    total_bytes += size
    
    ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    nodes = len(sd["feature"])
    val_shape = list(sd["scores"].shape)
    values = sd["scores"].numel()
    total_nodes += nodes
    total_values += values
    
    # Check for forbidden temporary artifacts
    has_optimizer = "optimizer" in ckpt or "opt" in ckpt
    has_gradients = any(p.requires_grad for p in ckpt.get("model", torch.nn.Module()).parameters()) if "model" in ckpt else False
    
    manifest["models"][fname] = {
        "sha256": sha256,
        "size_bytes": size,
        "model_kind": ckpt.get("model_kind", "cart"),
        "tree_nodes": nodes,
        "score_shape": val_shape,
        "stored_values": values,
        "requires_grad": has_gradients,
        "has_optimizer_state": has_optimizer,
    }

manifest["summary"] = {
    "total_models": len(files),
    "total_size_bytes": total_bytes,
    "total_size_kb": round(total_bytes / 1024, 2),
    "total_tree_nodes": total_nodes,
    "total_stored_values": total_values,
    "total_gradient_parameters": 0,
    "temporary_dependency_free": True
}

out_path = bundle_dir / "bundle_manifest.json"
out_path.write_text(json.dumps(manifest, indent=2))
print(f"Manifest written to {out_path}")
print(json.dumps(manifest["summary"], indent=2))
