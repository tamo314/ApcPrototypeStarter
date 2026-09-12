"""Evaluation-only MIRROR_HALVES attention metrics with explicit head semantics."""

from collections.abc import Sequence

import torch

from apc.evaluation.mirror_position_initialization_diagnostic import mirror_halves_position_map


def position_statistics(
    scores: torch.Tensor,
    probabilities: torch.Tensor,
    content_lengths: Sequence[int],
    output_position: int,
) -> dict[str, torch.Tensor]:
    """Return per-example, per-head metrics; ties use the worst correct-key rank.

    Scores have shape [batch, heads, output, input]. Padding never competes with
    a valid key. The operation's output-to-input map is evaluator metadata only.
    """
    if scores.ndim != 4 or probabilities.shape != scores.shape:
        raise ValueError("scores and probabilities must share [batch, heads, output, input]")
    batch, heads, outputs, inputs = scores.shape
    if len(content_lengths) != batch or not 0 <= output_position < outputs:
        raise ValueError("invalid batch lengths or output position")
    if any(not output_position < n <= inputs for n in content_lengths):
        raise ValueError("output position must be valid in every example")
    keys = torch.tensor(
        [mirror_halves_position_map(n)[output_position] for n in content_lengths],
        device=scores.device,
    )
    row = scores[:, :, output_position, :]
    correct = row.gather(-1, keys[:, None, None].expand(batch, heads, 1)).squeeze(-1)
    indices = torch.arange(inputs, device=scores.device)[None, None, :]
    lengths = torch.tensor(content_lengths, device=scores.device)[:, None, None]
    competitors = (indices < lengths) & (indices != keys[:, None, None])
    wrong = row.masked_fill(~competitors, -torch.inf)
    rank = 1 + ((row >= correct.unsqueeze(-1)) & competitors).sum(dim=-1)
    probability = probabilities[:, :, output_position, :].gather(
        -1, keys[:, None, None].expand(batch, heads, 1)
    ).squeeze(-1)
    return {
        "margin": correct - wrong.max(dim=-1).values,
        "probability": probability,
        "rank": rank,
    }


def matched_score_norms(
    base: torch.Tensor, residual: torch.Tensor, content_lengths: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Return per-example Frobenius ratios on equal head shapes and valid positions.

    The primary ratio removes each row's softmax-invariant constant offset.
    A zero denominator is NaN (undefined), never evidence of zero contribution.
    """
    if base.ndim != 4 or residual.shape != (base.shape[0], base.shape[2], base.shape[3]):
        raise ValueError("expected base [B,H,O,I] and residual [B,O,I]")
    if len(content_lengths) != base.shape[0] or any(
        not 1 <= n <= min(base.shape[2:]) for n in content_lengths
    ):
        raise ValueError("invalid content lengths")
    expanded = residual[:, None].expand_as(base)
    lengths = torch.tensor(content_lengths, device=base.device)[:, None, None, None]
    key_mask = torch.arange(base.shape[-1], device=base.device)[None, None, None, :] < lengths
    out_mask = torch.arange(base.shape[-2], device=base.device)[None, None, :, None] < lengths
    valid = key_mask & out_mask
    base_valid = base.masked_fill(~valid, 0.0)
    residual_valid = expanded.masked_fill(~valid, 0.0)
    base_centered = (base_valid - base_valid.sum(-1, keepdim=True) / lengths).masked_fill(
        ~valid, 0.0
    )
    residual_centered = (
        residual_valid - residual_valid.sum(-1, keepdim=True) / lengths
    ).masked_fill(~valid, 0.0)

    def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        n = numerator.flatten(1).norm(dim=1)
        d = denominator.flatten(1).norm(dim=1)
        return torch.where(d > 0, n / d, torch.full_like(d, torch.nan))

    return {
        "centered": ratio(residual_centered, base_centered),
        "uncentered": ratio(residual_valid, base_valid),
    }


def cvof_parameter_accounting(model: torch.nn.Module) -> dict[str, int]:
    """Count scalar update masks separately from resident and execution capacity."""
    parameters = dict(model.named_parameters())
    total = sum(p.numel() for p in parameters.values())
    frozen = 0
    for name, parameter in parameters.items():
        if name.startswith((
            "content_in_proj.", "content_position_embedding.",
            "cross_attn.out_proj.", "ffn.", "ffn_norm.",
        )):
            frozen += parameter.numel()
        elif name in {"cross_attn.in_proj_weight", "cross_attn.in_proj_bias"}:
            frozen += parameter.numel() // 3
    residual = sum(p.numel() for n, p in parameters.items() if n.startswith("score_residual."))
    split = sum(p.numel() for n, p in parameters.items() if n.startswith("key_content_"))
    return {
        "total_resident_parameters_model": total,
        "active_execution_parameters_target_primitive": total,
        "frozen_restored_parameters": frozen,
        "active_trainable_parameters": total - frozen,
        "added_resident_parameters": residual,
        "added_resident_parameters_vs_role_split": residual,
        "added_resident_parameters_vs_legacy": residual + split,
    }
