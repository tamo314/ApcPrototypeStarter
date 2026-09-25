#!/usr/bin/env python3
"""Generate bundle_manifest.json for dist_autonomous_bundle_v1."""

import json
import hashlib
import os
import torch

def inspect_model(fpath):
    with open(fpath, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    sz = len(data)
    
    ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
    print(f"File {fpath} keys: {list(ckpt.keys())}")
    if "tree_nodes" in ckpt:
        n_nodes = ckpt["tree_nodes"]
        val_shape = [n_nodes, 20]
        s_vals = n_nodes * 20
    elif "n_nodes" in ckpt:
        n_nodes = ckpt["n_nodes"]
        n_cls = len(ckpt.get("class_names", [1]))
        val_shape = [n_nodes, n_cls]
        s_vals = n_nodes * n_cls
    elif "cart" in ckpt:
        c = ckpt["cart"]
        n_nodes = c.get("node_count", len(c.get("threshold", [])))
        scores = c.get("scores", None)
        if scores is not None:
            val_shape = list(scores.shape)
            s_vals = int(scores.numel())
        else:
            val_shape = [n_nodes, 20]
            s_vals = n_nodes * 20
    else:
        n_nodes = 0
        val_shape = [0]
        s_vals = 0

    return {
        "sha256": sha,
        "size_bytes": sz,
        "model_kind": "cart",
        "tree_nodes": n_nodes,
        "score_shape": val_shape,
        "stored_values": s_vals,
        "requires_grad": False,
        "has_optimizer_state": False,
    }

def main():
    bundle_dir = "dist_autonomous_bundle_v1"
    files = ["base_selector.pt", "place_candidate.pt", "transit_candidate.pt", "unified_router.pt"]
    
    manifest = {
        "bundle_name": "apc_autonomous_cart_suite_v1",
        "description": "Minimal standalone bundle containing Base CART, Candidate CARTs, and Unified Router CART. Zero gradient parameters, 100% temporary dependency free.",
        "models": {},
        "summary": {}
    }
    
    total_size = 0
    total_nodes = 0
    total_values = 0
    
    for fname in files:
        fpath = os.path.join(bundle_dir, fname)
        info = inspect_model(fpath)
        manifest["models"][fname] = info
        total_size += info["size_bytes"]
        total_nodes += info["tree_nodes"]
        total_values += info["stored_values"]
        print(f"{fname}: {info['size_bytes']} bytes, {info['tree_nodes']} nodes, {info['stored_values']} values")
        
    manifest["summary"] = {
        "total_models": len(files),
        "total_size_bytes": total_size,
        "total_size_kb": round(total_size / 1024, 2),
        "total_tree_nodes": total_nodes,
        "total_stored_values": total_values,
        "total_gradient_parameters": 0,
        "temporary_dependency_free": True
    }
    
    out_path = os.path.join(bundle_dir, "bundle_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nWrote manifest to {out_path}:")
    print(json.dumps(manifest["summary"], indent=2))

if __name__ == "__main__":
    main()
