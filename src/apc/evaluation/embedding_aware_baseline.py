"""Embedding-Aware Deterministic Baseline (B_det_emb) for Phase C Identifiability Audits.

Task C-D001Y: Embedding-Aware Deterministic Baseline Closure Audit.
Evaluates whether continuous representation grounding is an estimand that can be
dominated by an unlearned deterministic baseline equipped with a fixed codebook,
nearest-neighbor projection, and known invertible linear map inversion.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class Codebook:
    """Fixed, unlearned continuous embedding codebook."""

    def __init__(self, token_embeddings: dict[int, np.ndarray]) -> None:
        if not token_embeddings:
            raise ValueError("Codebook cannot be empty")
        self._tokens = tuple(sorted(token_embeddings.keys()))
        self._dim = len(next(iter(token_embeddings.values())))
        self._matrix = np.zeros((len(self._tokens), self._dim), dtype=np.float64)
        self._token_to_idx: dict[int, int] = {}

        for idx, token in enumerate(self._tokens):
            emb = np.asarray(token_embeddings[token], dtype=np.float64)
            if emb.shape != (self._dim,):
                raise ValueError(
                    f"Inconsistent embedding dimension for token {token}: "
                    f"expected {self._dim}, got {emb.shape}"
                )
            self._matrix[idx] = emb
            self._token_to_idx[token] = idx

    @property
    def tokens(self) -> tuple[int, ...]:
        return self._tokens

    @property
    def dim(self) -> int:
        return self._dim

    def lookup(self, token: int) -> np.ndarray:
        if token not in self._token_to_idx:
            raise KeyError(f"Token {token} not found in codebook")
        return self._matrix[self._token_to_idx[token]].copy()

    def nearest_neighbor(
        self, vec: np.ndarray, tolerance: float = 1e-6
    ) -> tuple[int, list[int], float]:
        """Project a continuous vector to the closest codebook entry.

        Returns:
            (best_token, tied_tokens, min_distance)
        """
        vec = np.asarray(vec, dtype=np.float64)
        diffs = self._matrix - vec
        distances = np.linalg.norm(diffs, axis=1)
        min_dist = float(np.min(distances))

        # Identify all tokens within tolerance of min_distance
        tied_indices = np.where(np.abs(distances - min_dist) <= tolerance)[0]
        tied_tokens = [self._tokens[idx] for idx in tied_indices]
        best_token = tied_tokens[0]

        return best_token, tied_tokens, min_dist

    def min_pairwise_distance(self) -> float:
        """Compute minimum Euclidean distance between distinct codebook entries."""
        if len(self._tokens) < 2:
            return float("inf")
        min_d = float("inf")
        for i in range(len(self._tokens)):
            for j in range(i + 1, len(self._tokens)):
                d = float(np.linalg.norm(self._matrix[i] - self._matrix[j]))
                if d < min_d:
                    min_d = d
        return min_d


class InvertibleLinearTransform:
    """Known, invertible linear transformation over continuous embeddings."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Transform matrix must be square")
        det = float(np.linalg.det(self.matrix))
        if abs(det) < 1e-9:
            raise ValueError(f"Matrix is singular or near-singular (det={det})")
        self.inv_matrix = np.linalg.inv(self.matrix)

    def forward(self, vec: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, np.asarray(vec, dtype=np.float64))

    def inverse(self, vec: np.ndarray) -> np.ndarray:
        return np.dot(self.inv_matrix, np.asarray(vec, dtype=np.float64))


class EmbeddingAwareDeterministicBaseline:
    """Embedding-Aware Deterministic Baseline (B_det_emb).

    Consists of:
    1. Inverse linear mapping front-end (W^-1) if linear transform is present.
    2. Nearest-neighbor projection onto fixed codebook C.
    3. Pure deterministic dispatch to public operation semantics (B_det).
    Zero learned parameters, zero training updates.
    """

    def __init__(
        self,
        codebook: Codebook,
        transform: InvertibleLinearTransform | None = None,
        tolerance: float = 1e-6,
    ) -> None:
        self.codebook = codebook
        self.transform = transform
        self.tolerance = tolerance

    def decode_sequence(
        self, h_seq: np.ndarray
    ) -> tuple[list[int], bool, list[list[int]]]:
        """Decode sequence of continuous embeddings into discrete tokens.

        Returns:
            (decoded_tokens, is_exact, candidate_tokens_per_position)
        """
        h_seq = np.asarray(h_seq, dtype=np.float64)
        decoded_tokens: list[int] = []
        candidate_tokens_per_pos: list[list[int]] = []
        is_exact = True

        for i in range(len(h_seq)):
            vec = h_seq[i]
            if self.transform is not None:
                vec = self.transform.inverse(vec)
            best_token, tied_tokens, _ = self.codebook.nearest_neighbor(
                vec, tolerance=self.tolerance
            )
            decoded_tokens.append(best_token)
            candidate_tokens_per_pos.append(tied_tokens)
            if len(tied_tokens) > 1:
                is_exact = False

        return decoded_tokens, is_exact, candidate_tokens_per_pos

    def route_and_execute(
        self,
        h_seq: np.ndarray,
        descriptor: dict[str, Any],
    ) -> dict[str, Any]:
        """Decode continuous representations and execute deterministic relation semantics.

        Args:
            h_seq: continuous embedding sequence (L, dim)
            descriptor: AST descriptor with 'op', 'args', 'tie_break'
        """
        decoded_x, is_exact, candidate_tokens = self.decode_sequence(h_seq)
        op_name = descriptor.get("op", "")
        args = descriptor.get("args", {})
        tie_break = descriptor.get("tie_break", "FIRST")

        tied_positions = [i for i, cands in enumerate(candidate_tokens) if len(cands) > 1]

        if op_name == "BIND" or op_name == "BindOp":
            query_key = args["query_key"]
            matching_value_coords = [
                2 * j + 1
                for j in range(len(decoded_x) // 2)
                if decoded_x[2 * j] == query_key
            ]
            if not matching_value_coords:
                z_star = -1
                output_token = -1
            else:
                if tie_break == "FIRST":
                    z_star = min(matching_value_coords)
                elif tie_break == "LAST":
                    z_star = max(matching_value_coords)
                else:
                    raise ValueError(f"Unsupported tie-break for BIND: {tie_break}")
                output_token = decoded_x[z_star]

            return {
                "z_star": z_star,
                "output_token": output_token,
                "decoded_sequence": decoded_x,
                "exact_decoding": is_exact,
                "tied_positions": tied_positions,
                "candidate_coords": matching_value_coords,
            }

        elif op_name == "NEIGHBOR_MAX" or op_name == "NeighborMaxOp":
            # Circular 3-window max around center position (default position 1 or specified)
            pos = args.get("position", 1)
            L = len(decoded_x)
            window_indices = [(pos - 1) % L, pos, (pos + 1) % L]
            window_vals = [decoded_x[idx] for idx in window_indices]
            max_val = max(window_vals)

            max_coords = [idx for idx in window_indices if decoded_x[idx] == max_val]

            if tie_break in ("FIRST", "LEFTMOST"):
                z_star = max_coords[0]
            elif tie_break in ("LAST", "RIGHTMOST"):
                z_star = max_coords[-1]
            else:
                raise ValueError(f"Unsupported tie-break for NEIGHBOR_MAX: {tie_break}")

            output_token = decoded_x[z_star]
            return {
                "z_star": z_star,
                "output_token": output_token,
                "decoded_sequence": decoded_x,
                "exact_decoding": is_exact,
                "tied_positions": tied_positions,
                "candidate_coords": max_coords,
            }

        else:
            raise NotImplementedError(f"Operation {op_name} not implemented in baseline")


def compute_routing_entropy(candidate_coords: list[int]) -> float:
    """Compute conditional routing entropy H(Z | H(X), D) in bits."""
    if not candidate_coords:
        return 0.0
    k = len(candidate_coords)
    if k <= 1:
        return 0.0
    p = 1.0 / k
    return float(-sum(p * math.log2(p) for _ in range(k)))
