"""Small weighted CART baseline; thresholds and leaf choices are fit from labels."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CART(nn.Module):
    def __init__(self, nodes, output_dim):
        super().__init__()
        self.register_buffer("feature", torch.full((nodes,), -1, dtype=torch.int64))
        self.register_buffer("threshold", torch.zeros(nodes, dtype=torch.float32))
        self.register_buffer("left", torch.zeros(nodes, dtype=torch.int64))
        self.register_buffer("right", torch.zeros(nodes, dtype=torch.int64))
        self.register_buffer("scores", torch.zeros((nodes, output_dim), dtype=torch.float32))

    def forward(self, x):
        single = x.ndim == 1
        data = x.detach().cpu().numpy().reshape(-1, x.shape[-1])
        feature, threshold = self.feature.numpy(), self.threshold.numpy()
        left, right = self.left.numpy(), self.right.numpy()
        node = np.zeros(len(data), dtype=np.int64)
        while True:
            active = np.flatnonzero(feature[node] >= 0)
            if not len(active):
                break
            current = node[active]
            goes_left = data[active, feature[current]] <= threshold[current]
            node[active] = np.where(goes_left, left[current], right[current])
        result = self.scores[torch.from_numpy(node)]
        return result[0] if single else result

    @classmethod
    def fit(cls, x, y, weights, output_dim, *, max_depth=12, min_leaf=2):
        if max_depth < 1 or min_leaf < 1:
            raise ValueError("CART depth and leaf size must be positive")
        nodes = []

        def grow(indices, depth):
            counts = np.bincount(y[indices], weights=weights[indices], minlength=output_dim)
            probabilities = counts / counts.sum()
            node = len(nodes)
            nodes.append([-1, 0., 0, 0, np.log(np.maximum(probabilities, 1e-8))])
            if depth >= max_depth or len(indices) < 2 * min_leaf or np.count_nonzero(counts) == 1:
                return node
            best_gain, best = 1e-10, None
            parent_score = np.dot(counts, counts) / counts.sum()
            for feature in range(x.shape[1]):
                order = indices[np.argsort(x[indices, feature], kind="stable")]
                values = x[order, feature]
                cuts = np.flatnonzero(values[:-1] < values[1:])
                cuts = cuts[(cuts + 1 >= min_leaf) & (len(order) - cuts - 1 >= min_leaf)]
                if not len(cuts):
                    continue
                mass = np.zeros((len(order), output_dim), dtype=np.float64)
                mass[np.arange(len(order)), y[order]] = weights[order]
                lc = np.cumsum(mass, axis=0)[cuts]
                rc = np.maximum(counts - lc, 0.)
                gain = ((lc * lc).sum(1) / lc.sum(1)
                        + (rc * rc).sum(1) / rc.sum(1) - parent_score)
                candidate = int(np.argmax(gain))
                if gain[candidate] > best_gain:
                    cut = cuts[candidate]
                    threshold = np.float32((float(values[cut]) + float(values[cut + 1])) / 2)
                    if threshold >= values[cut + 1]:
                        threshold = values[cut]
                    best_gain = gain[candidate]
                    best = (feature, threshold, order[:cut + 1], order[cut + 1:])
            if best is not None:
                feature, threshold, li, ri = best
                nodes[node][:4] = [feature, threshold, grow(li, depth + 1), grow(ri, depth + 1)]
            return node

        grow(np.arange(len(x)), 0)
        model = cls(len(nodes), output_dim)
        model.feature.copy_(torch.tensor([n[0] for n in nodes]))
        model.threshold.copy_(torch.tensor([n[1] for n in nodes]))
        model.left.copy_(torch.tensor([n[2] for n in nodes]))
        model.right.copy_(torch.tensor([n[3] for n in nodes]))
        model.scores.copy_(torch.from_numpy(np.stack([n[4] for n in nodes])))
        return model
