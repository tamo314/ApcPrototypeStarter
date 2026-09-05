"""Sequential Closed-Loop Autonomous Stream Benchmark (Phase A.2 Task A2-C008 - STOP GATE).

Evaluates the complete APC continual learning closed-loop in an online episode stream
where the learned controller autonomously chooses between direct reuse, composition,
and plastic expansion without runtime oracle action labels.

Requirements (from CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md):
- Minimum stream per seed: >= 40 episodes:
  - >= 10 K (Known)
  - >= 10 C (Composition)
  - >= 6 N (Novel, first occurrence)
  - >= 6 R (Recurrence)
- Runtime oracle prohibition: No oracle action labels. Categories logged only for evaluation.
- Primary acceptance criteria:
  - K: EM >= 0.95, false plastic <= 0.10
  - C: EM >= 0.90, composition action >= 0.85, bank expansion <= 0.10
  - N: final EM >= 0.90, plastic trigger >= 0.90, exactly 1 bank growth per novel task
  - R: EM >= 0.95, direct reuse >= 0.90, no new consolidation >= 0.90
  - Global: old-task degradation <= 0.02 (2 pp), no temporary leak, >= 5 seeds
- STOP GATE discipline: Hard scientific gate.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import torch
import torch.nn.functional as F

from apc.environments.generator import (
    Example,
    OracleMetadata,
    Program,
    ProgramStep,
)
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
)
from apc.environments.task_spec import TaskSpec
from apc.evaluation.incremental_router_benchmark import (
    INITIAL_10_OPERATIONS,
    extract_task_representations,
    get_or_build_16_primitive_bank,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.meta.adequacy import (
    AdequacyEvidenceConfig,
    compute_adequacy_evidence,
)
from apc.meta.episode_log import ControllerAction
from apc.meta.learned_controller import (
    LearnedAdequacyController,
    build_default_trained_controller,
)
from apc.plastic.lifecycle import (
    CompactLifecycleConfig,
    CompactPlasticLifecyclePolicy,
)
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.incremental_router import (
    IncrementalRouterConfig,
    IncrementalUpdateCondition,
    RouterReplayBuffer,
    align_shared_core_embeddings,
    update_router_incrementally,
)
from apc.primitives.router import Router, RouterConfig
from apc.utils.seed import set_seed

DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
DEFAULT_NUM_K: Final[int] = 14
DEFAULT_NUM_C: Final[int] = 12
DEFAULT_NUM_N: Final[int] = 6
DEFAULT_NUM_R: Final[int] = 8
DEFAULT_SUPPORT_SIZE: Final[int] = 16
DEFAULT_EVAL_SIZE: Final[int] = 32
DEFAULT_PLASTIC_TRAIN_SIZE: Final[int] = 80

# Candidate operation pools
CANONICAL_DETERMINISTIC_K_OPS: Final[tuple[str, ...]] = (
    "COPY",
    "NEGATE",
    "REVERSE",
    "SWAP_PAIRS",
    "INVERT_HALF",
)

COMPOSITION_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("SWAP_PAIRS", "NEGATE"),
    ("COPY", "NEGATE"),
    ("NEGATE", "SWAP_PAIRS"),
    ("COPY", "SWAP_PAIRS"),
    ("NEGATE", "INVERT_HALF"),
    ("SWAP_PAIRS", "INVERT_HALF"),
    ("REVERSE", "NEGATE"),
    ("COPY", "REVERSE"),
)

NOVEL_OPS: Final[tuple[str, ...]] = PHASE_A2_INCREMENTAL_NEW_OPERATIONS


@dataclass(frozen=True)
class SequentialClosedLoopConfig:
    """Explicit, serializable configuration for Task A2-C008 benchmark."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    num_k: int = DEFAULT_NUM_K
    num_c: int = DEFAULT_NUM_C
    num_n: int = DEFAULT_NUM_N
    num_r: int = DEFAULT_NUM_R
    support_size: int = DEFAULT_SUPPORT_SIZE
    eval_size: int = DEFAULT_EVAL_SIZE
    plastic_train_size: int = 160
    compact_budget_steps: int = 800
    fallback_budget_steps: int = 350
    distillation_steps: int = 300
    compact_lr: float = 2e-3
    dev_seed: int = 42
    device_str: str = "auto"
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.num_k < 10:
            raise ValueError(f"num_k must be >= 10, got {self.num_k}")
        if self.num_c < 10:
            raise ValueError(f"num_c must be >= 10, got {self.num_c}")
        if self.num_n < 6:
            raise ValueError(f"num_n must be >= 6, got {self.num_n}")
        if self.num_r < 6:
            raise ValueError(f"num_r must be >= 6, got {self.num_r}")
        if self.total_episodes < 40:
            raise ValueError(f"total_episodes must be >= 40, got {self.total_episodes}")

    @property
    def total_episodes(self) -> int:
        return self.num_k + self.num_c + self.num_n + self.num_r

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "num_k": self.num_k,
            "num_c": self.num_c,
            "num_n": self.num_n,
            "num_r": self.num_r,
            "total_episodes": self.total_episodes,
            "support_size": self.support_size,
            "eval_size": self.eval_size,
            "plastic_train_size": self.plastic_train_size,
            "compact_budget_steps": self.compact_budget_steps,
            "fallback_budget_steps": self.fallback_budget_steps,
            "distillation_steps": self.distillation_steps,
            "compact_lr": self.compact_lr,
            "dev_seed": self.dev_seed,
            "device_str": self.device_str,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


@dataclass(frozen=True)
class SequentialEpisodeSpec:
    """Specification of a single stream episode."""

    episode_index: int
    oracle_category: Literal["K", "C", "N", "R"]
    task_name: str
    program: Program
    seq_length: int = 8


@dataclass(frozen=True)
class EpisodeExecutionRecord:
    """Record of execution outcome for one episode."""

    episode_index: int
    oracle_category: Literal["K", "C", "N", "R"]
    task_name: str
    action_predicted: str
    novelty_score: float
    held_out_em: float
    held_out_tok_acc: float
    held_out_loss: float
    direct_em: float
    composition_em: float
    bank_size_before: int
    bank_size_after: int
    bank_expansion: bool
    workspace_param_count: int
    temporary_peak_params: int
    old_task_degradation: float
    elapsed_seconds: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SeedBenchmarkMetrics:
    """Summary of acceptance metrics for a single seed."""

    seed: int
    total_episodes: int

    # K metrics
    k_total: int
    k_em: float
    k_false_plastic: int
    k_false_plastic_rate: float
    passed_k_em: bool
    passed_k_false_plastic: bool

    # C metrics
    c_total: int
    c_em: float
    c_action_accuracy: float
    c_expansion_count: int
    c_expansion_rate: float
    passed_c_em: bool
    passed_c_action_accuracy: bool
    passed_c_expansion: bool

    # N metrics
    n_total: int
    n_final_em: float
    n_plastic_trigger_count: int
    n_plastic_trigger_rate: float
    n_promotions_count: int
    passed_n_final_em: bool
    passed_n_plastic_trigger: bool
    passed_n_promotions: bool

    # R metrics
    r_total: int
    r_em: float
    r_direct_reuse_count: int
    r_direct_reuse_rate: float
    r_reconsolidation_count: int
    passed_r_em: bool
    passed_r_direct_reuse: bool
    passed_r_no_reconsolidation: bool

    # Global invariants
    max_old_task_degradation: float
    workspace_leak_count: int
    passed_old_task_degradation: bool
    passed_no_workspace_leak: bool

    all_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def generate_episode_examples(
    program: Program,
    category: Literal["K", "C", "N", "R"],
    vocab_size: int,
    n_examples: int,
    seed: int,
    seq_length: int = 8,
) -> list[Example]:
    """Generate deterministic support/eval examples with zero oracle leakage."""
    rng = random.Random(seed)
    examples: list[Example] = []
    task_spec = TaskSpec.from_program(program)

    for _ in range(n_examples):
        seq = tuple(rng.randrange(vocab_size) for _ in range(seq_length))
        res = run_program(program, seq, vocab_size)

        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=program,
            operation_graph=res.graph,
            category=category,
            split="test",
            vocab_size=vocab_size,
            task_spec=task_spec,
            oracle_metadata=OracleMetadata(
                label=category,
                primitive_operations=program.operation_sequence,
            ),
        )
        examples.append(ex)

    return examples


def generate_sequential_stream(
    seed: int,
    config: SequentialClosedLoopConfig,
) -> list[SequentialEpisodeSpec]:
    """Construct an online stream of >= 40 episodes enforcing causal recurrence ordering.

    Invariant: For every novel operation N_i, its N episode occurs strictly before
    any corresponding R_i episode.
    """
    rng = random.Random(seed * 2027 + 13)
    total_slots = config.total_episodes
    slots: list[SequentialEpisodeSpec | None] = [None] * total_slots

    # 1. Novel operations (N) and Recurrences (R)
    novel_ops = list(NOVEL_OPS[: config.num_n])
    if len(novel_ops) < config.num_n:
        raise ValueError(
            f"Available novel ops ({len(novel_ops)}) < requested num_n ({config.num_n})"
        )

    # Pick N slots within the first 60% of stream to leave room for recurrences
    max_n_slot = int(total_slots * 0.6)
    n_candidate_slots = list(range(2, max_n_slot))
    rng.shuffle(n_candidate_slots)
    n_slots = sorted(n_candidate_slots[: config.num_n])

    # Assign R count per novel op (each at least once)
    r_counts = [1] * config.num_n
    remaining_r = config.num_r - config.num_n
    for _ in range(remaining_r):
        r_counts[rng.randrange(config.num_n)] += 1

    # 1. Place all N episodes first into their chosen slots
    for op_idx, op_name in enumerate(novel_ops):
        s_n = n_slots[op_idx]
        slots[s_n] = SequentialEpisodeSpec(
            episode_index=s_n,
            oracle_category="N",
            task_name=op_name,
            program=Program(steps=(ProgramStep(op_name),)),
        )

    # 2. Place R episodes strictly after each corresponding s_N
    for op_idx, op_name in enumerate(novel_ops):
        s_n = n_slots[op_idx]
        available_r_slots = [
            s for s in range(s_n + 1, total_slots) if slots[s] is None
        ]
        if len(available_r_slots) < r_counts[op_idx]:
            raise RuntimeError(f"Could not find valid R slots for novel task {op_name}")
        chosen_r = rng.sample(available_r_slots, r_counts[op_idx])
        for s_r in chosen_r:
            slots[s_r] = SequentialEpisodeSpec(
                episode_index=s_r,
                oracle_category="R",
                task_name=op_name,
                program=Program(steps=(ProgramStep(op_name),)),
            )

    # 3. Fill remaining slots with K and C episodes
    empty_slots = [s for s in range(total_slots) if slots[s] is None]
    rng.shuffle(empty_slots)

    k_slots = empty_slots[: config.num_k]
    c_slots = empty_slots[config.num_k : config.num_k + config.num_c]

    for i, s_k in enumerate(k_slots):
        op = CANONICAL_DETERMINISTIC_K_OPS[i % len(CANONICAL_DETERMINISTIC_K_OPS)]
        slots[s_k] = SequentialEpisodeSpec(
            episode_index=s_k,
            oracle_category="K",
            task_name=op,
            program=Program(steps=(ProgramStep(op),)),
        )

    for i, s_c in enumerate(c_slots):
        pair = COMPOSITION_PAIRS[i % len(COMPOSITION_PAIRS)]
        c_name = f"{pair[0]}_{pair[1]}"
        prog = Program(steps=(ProgramStep(pair[0]), ProgramStep(pair[1])))
        slots[s_c] = SequentialEpisodeSpec(
            episode_index=s_c,
            oracle_category="C",
            task_name=c_name,
            program=prog,
        )

    final_stream: list[SequentialEpisodeSpec] = []
    for idx, spec in enumerate(slots):
        assert spec is not None, f"Slot {idx} was not filled!"
        final_stream.append(
            dataclasses.replace(spec, episode_index=idx)
        )

    return final_stream


def _evaluate_candidate_recipe(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    candidate_ops: tuple[str, ...],
    eval_examples: Sequence[Example],
) -> tuple[float, float, float]:
    """Execute candidate primitive or composite recipe on eval examples.

    Returns:
        (exact_match, mean_token_accuracy, avg_loss)
    """
    if not eval_examples:
        return 0.0, 0.0, 0.0

    logits = execute_composition_recipe(
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        examples=eval_examples,
        candidate_operations=candidate_ops,
    )
    preds = logits.argmax(dim=-1)

    exact_matches = 0
    total_tokens = 0
    correct_tokens = 0
    total_loss = 0.0

    for i, ex in enumerate(eval_examples):
        target = ex.target_tokens
        n = len(target)
        pred = tuple(preds[i, :n].tolist())
        if pred == target:
            exact_matches += 1

        correct_tokens += sum(p == t for p, t in zip(pred, target, strict=True))
        total_tokens += n

        target_tensor = torch.tensor(target, dtype=torch.long, device=logits.device)
        pred_logits = logits[i, :n, :]
        step_loss = F.cross_entropy(pred_logits, target_tensor, reduction="sum")
        total_loss += step_loss.item()

    em = exact_matches / len(eval_examples)
    avg_loss = total_loss / max(1, total_tokens)
    tok_acc = correct_tokens / max(1, total_tokens)
    return em, tok_acc, avg_loss


def _setup_initial_environment(
    seed: int,
    device: torch.device,
) -> tuple[Any, PrimitiveBank, Router, dict[str, int], RouterReplayBuffer]:
    """Set up shared core, initial 10-primitive bank, calibrated router, and replay buffer."""
    arch_cfg = SharedEncoderArchitectureConfig(
        seed=seed,
        vocab_size=10,
        device=device.type,
    )
    arch = build_shared_encoder_architecture(arch_cfg)
    core = arch.core

    cand_dir = Path("runs/phase_a1_shift_compact_structural_probe") / f"seed_{seed}"
    ckpt_cand = cand_dir / "shared_encoder.pt"
    if ckpt_cand.is_file():
        old_sd = torch.load(ckpt_cand, map_location=device, weights_only=True)
        aligned_sd = align_shared_core_embeddings(core.model, old_sd, None, core.tokens)
        core.model.load_state_dict(aligned_sd)
    core.model.eval()
    for param in core.model.parameters():
        param.requires_grad_(False)

    ckpt_dir = Path("runs/phase_a2_bank_scaling_benchmark")
    if not (ckpt_dir / f"seed_{seed}" / "primitive_bank_16.pt").is_file():
        ckpt_dir = Path("runs/phase_a2_incremental_router_gate")
    bank_16, full_op_to_id = get_or_build_16_primitive_bank(core, seed, output_dir=ckpt_dir)

    ops_10 = list(INITIAL_10_OPERATIONS)
    pids_10 = [full_op_to_id[op] for op in ops_10]

    bank_10 = PrimitiveBank()
    for pid in pids_10:
        p = bank_16.get(pid)
        bank_10.add_primitive(copy.deepcopy(p))
    bank_10.to(device)
    bank_10.freeze_all()
    bank_10.eval()

    op_to_id_10 = {op: full_op_to_id[op] for op in ops_10}

    # Calibrate initial router
    router = Router(RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot"))
    router.to(device)

    train_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}
    for op in ops_10:
        pid = op_to_id_10[op]
        ex_train = generate_benchmark_examples(
            seed=seed * 1000 + 17,
            n=64,
            operation=op,
            split="train",
            vocab_size=core.tokens.env_vocab_size,
        )
        z_train = extract_task_representations(core, ex_train)
        train_z_by_pid[pid] = [(z_train[i], pid) for i in range(len(ex_train))]

    router_cfg = IncrementalRouterConfig(
        condition=IncrementalUpdateCondition.R0_FULL_RETRAIN,
        router_lr=0.005,
        router_steps=250,
        seed=seed,
    )
    replay_buf = RouterReplayBuffer(max_per_class=32, max_total=512)
    update_router_incrementally(
        router,
        candidate_ids=pids_10,
        new_primitive_ids=pids_10,
        new_data_by_pid=train_z_by_pid,
        replay_buffer=replay_buf,
        config=router_cfg,
        all_historical_data_by_pid=train_z_by_pid,
        device=device,
    )

    router.eval()
    for param in router.parameters():
        param.requires_grad_(False)

    return core, bank_10, router, op_to_id_10, replay_buf


def run_sequential_closed_loop_for_seed(
    seed: int,
    *,
    config: SequentialClosedLoopConfig,
    controller: LearnedAdequacyController,
    base_core: Any | None = None,
) -> tuple[SeedBenchmarkMetrics, list[EpisodeExecutionRecord]]:
    """Execute the full sequential K/C/N/R stream for a single seed."""
    set_seed(seed)
    use_cuda = (
        config.device_str == "auto" and torch.cuda.is_available()
    ) or config.device_str == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    if base_core is None:
        core, bank, router, op_to_id, replay_buf = _setup_initial_environment(seed, device)
    else:
        core = base_core
        # Build bank and router using base_core
        ckpt_dir = Path("runs/phase_a2_bank_scaling_benchmark")
        if not (ckpt_dir / f"seed_{seed}" / "primitive_bank_16.pt").is_file():
            ckpt_dir = Path("runs/phase_a2_incremental_router_gate")
        bank_16, full_op_to_id = get_or_build_16_primitive_bank(core, seed, output_dir=ckpt_dir)
        ops_10 = list(INITIAL_10_OPERATIONS)
        pids_10 = [full_op_to_id[op] for op in ops_10]
        bank = PrimitiveBank()
        for pid in pids_10:
            bank.add_primitive(copy.deepcopy(bank_16.get(pid)))
        bank.to(device)
        bank.freeze_all()
        bank.eval()
        op_to_id = {op: full_op_to_id[op] for op in ops_10}

        router = Router(RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot"))
        router.to(device)
        train_z_by_pid = {}
        for op in ops_10:
            pid = op_to_id[op]
            ex_train = generate_benchmark_examples(
                seed=seed * 1000 + 17,
                n=64,
                operation=op,
                split="train",
                vocab_size=core.tokens.env_vocab_size,
            )
            z_train = extract_task_representations(core, ex_train)
            train_z_by_pid[pid] = [(z_train[i], pid) for i in range(len(ex_train))]
        replay_buf = RouterReplayBuffer(max_per_class=32, max_total=512)
        router_cfg = IncrementalRouterConfig(
            condition=IncrementalUpdateCondition.R0_FULL_RETRAIN,
            router_lr=0.005,
            router_steps=250,
            seed=seed,
        )
        update_router_incrementally(
            router,
            candidate_ids=pids_10,
            new_primitive_ids=pids_10,
            new_data_by_pid=train_z_by_pid,
            replay_buffer=replay_buf,
            config=router_cfg,
            all_historical_data_by_pid=train_z_by_pid,
            device=device,
        )
        for pid in pids_10:
            replay_buf.add_exemplars(pid, train_z_by_pid[pid][:32])
        router.eval()
        for param in router.parameters():
            param.requires_grad_(False)

    workspace = PlasticWorkspace()
    plastic_policy = CompactPlasticLifecyclePolicy(
        CompactLifecycleConfig(
            compact_budget_steps=config.compact_budget_steps,
            compact_lr=config.compact_lr,
            compact_success_threshold_em=0.85,
            fallback_budget_steps=config.fallback_budget_steps,
            fallback_lr=config.compact_lr,
            fallback_success_threshold_em=0.85,
            distillation_steps=config.distillation_steps,
            shadow_retention_threshold=0.85,
            eval_batch_size=32,
            seed=seed,
        )
    )

    # Historical retention evaluation sets (t=0 baseline)
    historical_eval_sets: dict[str, list[Example]] = {}
    for op in CANONICAL_DETERMINISTIC_K_OPS:
        historical_eval_sets[op] = generate_benchmark_examples(
            seed=seed * 500 + 11, n=30, operation=op, split="val", vocab_size=10
        )

    baseline_historical_em: dict[str, float] = {}
    for op, ex_list in historical_eval_sets.items():
        if op in op_to_id:
            em, _, _ = _evaluate_candidate_recipe(core, bank, op_to_id, (op,), ex_list)
            baseline_historical_em[op] = em

    # Generate sequential stream
    stream_specs = generate_sequential_stream(seed, config)
    evidence_cfg = AdequacyEvidenceConfig(
        direct_eval_k=2,
        composition_max_depth=2,
        composition_beam_width=16,
    )

    episode_records: list[EpisodeExecutionRecord] = []
    max_observed_degradation = 0.0
    workspace_leaks = 0

    for ep_idx, spec in enumerate(stream_specs):
        t_ep_start = time.perf_counter()
        initial_bank_size = len(bank)

        # 1. Generate episode examples (zero oracle leakage)
        support_examples = generate_episode_examples(
            program=spec.program,
            category=spec.oracle_category,
            vocab_size=10,
            n_examples=config.support_size,
            seed=seed * 10000 + ep_idx * 31 + 7,
            seq_length=spec.seq_length,
        )
        eval_examples = generate_episode_examples(
            program=spec.program,
            category=spec.oracle_category,
            vocab_size=10,
            n_examples=config.eval_size,
            seed=seed * 10000 + ep_idx * 31 + 19,
            seq_length=spec.seq_length,
        )

        # 2. Extract runtime functional evidence from support set
        ev = compute_adequacy_evidence(
            core=core,
            bank=bank,
            router=router,
            op_to_id=op_to_id,
            support_examples=support_examples,
            config=evidence_cfg,
        )

        # 3. Autonomous learned controller decision
        pred = controller.predict(ev)
        action = pred.action
        id_to_op = {p: name for name, p in op_to_id.items()}

        held_out_em = 0.0
        held_out_tok = 0.0
        held_out_loss = float("inf")
        peak_workspace_params = 0
        ep_details: dict[str, Any] = {}

        # 4. Action execution
        if action == ControllerAction.DIRECT_REUSE:
            direct_pid = ev.direct_primitive_id
            direct_op = id_to_op.get(direct_pid) if direct_pid is not None else None
            if direct_op is not None and direct_op in op_to_id:
                held_out_em, held_out_tok, held_out_loss = _evaluate_candidate_recipe(
                    core, bank, op_to_id, (direct_op,), eval_examples
                )
            ep_details["executed"] = f"DIRECT_REUSE: pid={direct_pid}, op={direct_op}"

        elif action == ControllerAction.COMPOSE:
            recipe = ev.composition_recipe
            if recipe is not None:
                held_out_em, held_out_tok, held_out_loss = _evaluate_candidate_recipe(
                    core, bank, op_to_id, recipe, eval_examples
                )
            ep_details["executed"] = f"COMPOSE: recipe={recipe}"

        elif action == ControllerAction.PLASTIC_SEARCH:
            # Generate adaptation training set
            plastic_train_examples = generate_episode_examples(
                program=spec.program,
                category=spec.oracle_category,
                vocab_size=10,
                n_examples=config.plastic_train_size,
                seed=seed * 10000 + ep_idx * 31 + 43,
                seq_length=spec.seq_length,
            )

            # Invoke compact plastic lifecycle policy
            lifecycle_rep = plastic_policy.execute_episode(
                episode_id=f"seed_{seed}_ep_{ep_idx:02d}_{spec.oracle_category}",
                task_name=spec.task_name,
                action=ControllerAction.PLASTIC_SEARCH,
                evidence=ev,
                train_examples=plastic_train_examples,
                eval_examples=eval_examples,
                historical_eval_examples=historical_eval_sets,
                core=core,
                bank=bank,
                op_to_id=op_to_id,
                workspace=workspace,
            )
            peak_workspace_params = lifecycle_rep.temporary_peak_params
            ep_details["lifecycle"] = lifecycle_rep.to_dict()

            if lifecycle_rep.promotions_count > 0:
                new_pid = lifecycle_rep.promoted_primitive_id
                assert new_pid is not None

                # Incremental router update under R2 bounded replay
                z_new = extract_task_representations(core, plastic_train_examples)
                new_data_by_pid = {
                    new_pid: [(z_new[i], new_pid) for i in range(len(plastic_train_examples))]
                }
                replay_buf.add_exemplars(
                    new_pid, new_data_by_pid[new_pid], rng=random.Random(seed + new_pid)
                )

                router_cfg = IncrementalRouterConfig(
                    condition=IncrementalUpdateCondition.R2_BOUNDED_REPLAY,
                    router_lr=0.005,
                    router_steps=250,
                    seed=seed + ep_idx,
                )
                update_router_incrementally(
                    router,
                    candidate_ids=bank.ids(),
                    new_primitive_ids=[new_pid],
                    new_data_by_pid=new_data_by_pid,
                    replay_buffer=replay_buf,
                    config=router_cfg,
                    device=device,
                )
                router.eval()
                for p in router.parameters():
                    p.requires_grad_(False)

                # Add newly consolidated task to historical evaluation tracking
                historical_eval_sets[spec.task_name] = generate_episode_examples(
                    program=spec.program,
                    category=spec.oracle_category,
                    vocab_size=10,
                    n_examples=30,
                    seed=seed * 500 + ep_idx * 17 + 3,
                    seq_length=spec.seq_length,
                )
                em_init, _, _ = _evaluate_candidate_recipe(
                    core, bank, op_to_id, (spec.task_name,), historical_eval_sets[spec.task_name]
                )
                baseline_historical_em[spec.task_name] = em_init

            # Evaluate final promoted primitive on held-out eval examples
            if spec.task_name in op_to_id:
                held_out_em, held_out_tok, held_out_loss = _evaluate_candidate_recipe(
                    core, bank, op_to_id, (spec.task_name,), eval_examples
                )

        # 5. Invariant check: zero workspace parameter leak
        final_workspace_params = workspace.total_parameter_count()
        if final_workspace_params != 0 or workspace.is_allocated:
            workspace_leaks += 1
            workspace.release()

        # 6. Retention check if bank expanded
        current_bank_size = len(bank)
        bank_expanded = current_bank_size > initial_bank_size
        ep_degradation = 0.0

        if bank_expanded:
            # Check degradation on all pre-existing tasks
            for h_op, base_em in baseline_historical_em.items():
                if h_op == spec.task_name:
                    continue
                if h_op in op_to_id and h_op in historical_eval_sets:
                    curr_em, _, _ = _evaluate_candidate_recipe(
                        core, bank, op_to_id, (h_op,), historical_eval_sets[h_op]
                    )
                    drop = max(0.0, base_em - curr_em)
                    if drop > ep_degradation:
                        ep_degradation = drop
            if ep_degradation > max_observed_degradation:
                max_observed_degradation = ep_degradation

        elapsed_ep = time.perf_counter() - t_ep_start

        rec = EpisodeExecutionRecord(
            episode_index=ep_idx,
            oracle_category=spec.oracle_category,
            task_name=spec.task_name,
            action_predicted=action.value,
            novelty_score=pred.novelty_score,
            held_out_em=held_out_em,
            held_out_tok_acc=held_out_tok,
            held_out_loss=held_out_loss,
            direct_em=ev.direct_em,
            composition_em=ev.composition_em,
            bank_size_before=initial_bank_size,
            bank_size_after=current_bank_size,
            bank_expansion=bank_expanded,
            workspace_param_count=final_workspace_params,
            temporary_peak_params=peak_workspace_params,
            old_task_degradation=ep_degradation,
            elapsed_seconds=elapsed_ep,
            details=ep_details,
        )
        episode_records.append(rec)

    # Aggregate metrics for this seed
    k_recs = [r for r in episode_records if r.oracle_category == "K"]
    c_recs = [r for r in episode_records if r.oracle_category == "C"]
    n_recs = [r for r in episode_records if r.oracle_category == "N"]
    r_recs = [r for r in episode_records if r.oracle_category == "R"]

    # K metrics
    k_em = sum(r.held_out_em for r in k_recs) / len(k_recs) if k_recs else 0.0
    k_fp = sum(1 for r in k_recs if r.action_predicted == ControllerAction.PLASTIC_SEARCH.value)
    k_fp_rate = k_fp / len(k_recs) if k_recs else 0.0

    # C metrics
    c_em = sum(r.held_out_em for r in c_recs) / len(c_recs) if c_recs else 0.0
    c_action_acc = (
        sum(1 for r in c_recs if r.action_predicted == ControllerAction.COMPOSE.value) / len(c_recs)
        if c_recs
        else 0.0
    )
    c_exp_count = sum(1 for r in c_recs if r.bank_expansion)
    c_exp_rate = c_exp_count / len(c_recs) if c_recs else 0.0

    # N metrics
    n_em = sum(r.held_out_em for r in n_recs) / len(n_recs) if n_recs else 0.0
    n_trig = sum(1 for r in n_recs if r.action_predicted == ControllerAction.PLASTIC_SEARCH.value)
    n_trig_rate = n_trig / len(n_recs) if n_recs else 0.0
    n_prom_count = sum(1 for r in n_recs if r.bank_expansion)

    # R metrics
    r_em = sum(r.held_out_em for r in r_recs) / len(r_recs) if r_recs else 0.0
    r_direct = sum(1 for r in r_recs if r.action_predicted == ControllerAction.DIRECT_REUSE.value)
    r_direct_rate = r_direct / len(r_recs) if r_recs else 0.0
    r_recons = sum(1 for r in r_recs if r.bank_expansion)

    passed_k_em = k_em >= 0.95
    passed_k_fp = k_fp_rate <= 0.10
    passed_c_em = c_em >= 0.90
    passed_c_act = c_action_acc >= 0.85
    passed_c_exp = c_exp_rate <= 0.10
    passed_n_em = n_em >= 0.90
    passed_n_trig = n_trig_rate >= 0.90
    passed_n_prom = n_prom_count == len(n_recs)
    passed_r_em = r_em >= 0.95
    passed_r_dir = r_direct_rate >= 0.90
    passed_r_rec = r_recons == 0
    passed_degradation = max_observed_degradation <= 0.02
    passed_leaks = workspace_leaks == 0

    all_passed = (
        passed_k_em
        and passed_k_fp
        and passed_c_em
        and passed_c_act
        and passed_c_exp
        and passed_n_em
        and passed_n_trig
        and passed_n_prom
        and passed_r_em
        and passed_r_dir
        and passed_r_rec
        and passed_degradation
        and passed_leaks
    )

    metrics = SeedBenchmarkMetrics(
        seed=seed,
        total_episodes=len(episode_records),
        k_total=len(k_recs),
        k_em=k_em,
        k_false_plastic=k_fp,
        k_false_plastic_rate=k_fp_rate,
        passed_k_em=passed_k_em,
        passed_k_false_plastic=passed_k_fp,
        c_total=len(c_recs),
        c_em=c_em,
        c_action_accuracy=c_action_acc,
        c_expansion_count=c_exp_count,
        c_expansion_rate=c_exp_rate,
        passed_c_em=passed_c_em,
        passed_c_action_accuracy=passed_c_act,
        passed_c_expansion=passed_c_exp,
        n_total=len(n_recs),
        n_final_em=n_em,
        n_plastic_trigger_count=n_trig,
        n_plastic_trigger_rate=n_trig_rate,
        n_promotions_count=n_prom_count,
        passed_n_final_em=passed_n_em,
        passed_n_plastic_trigger=passed_n_trig,
        passed_n_promotions=passed_n_prom,
        r_total=len(r_recs),
        r_em=r_em,
        r_direct_reuse_count=r_direct,
        r_direct_reuse_rate=r_direct_rate,
        r_reconsolidation_count=r_recons,
        passed_r_em=passed_r_em,
        passed_r_direct_reuse=passed_r_dir,
        passed_r_no_reconsolidation=passed_r_rec,
        max_old_task_degradation=max_observed_degradation,
        workspace_leak_count=workspace_leaks,
        passed_old_task_degradation=passed_degradation,
        passed_no_workspace_leak=passed_leaks,
        all_passed=all_passed,
    )

    return metrics, episode_records


def run_sequential_closed_loop_benchmark(
    config: SequentialClosedLoopConfig,
) -> dict[str, Any]:
    """Execute the multi-seed full sequential closed-loop benchmark (Task A2-C008 STOP GATE)."""
    print("=" * 75)
    print("Phase A.2 Task A2-C008: Full Sequential K/C/N/R Autonomous Closed Loop")
    print(f"Seeds: {list(config.seeds)}")
    print(
        f"Stream per seed: {config.total_episodes} episodes "
        f"(K: {config.num_k}, C: {config.num_c}, N: {config.num_n}, R: {config.num_r})"
    )
    print(f"Output directory: {config.output_dir}")
    print("=" * 75)

    t0 = time.perf_counter()

    # Pre-train and freeze learned controller (ADR-0067)
    controller = build_default_trained_controller(seed=config.dev_seed)
    controller.freeze()

    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        controller.save(config.output_dir / "controller.json")

    per_seed_metrics: list[SeedBenchmarkMetrics] = []
    all_episodes: list[EpisodeExecutionRecord] = []

    for seed in config.seeds:
        print(f"\n--> Executing Online Stream for Seed {seed}...")
        t_seed = time.perf_counter()
        m, ep_records = run_sequential_closed_loop_for_seed(
            seed, config=config, controller=controller
        )
        per_seed_metrics.append(m)
        all_episodes.extend(ep_records)
        el_s = time.perf_counter() - t_seed
        verdict = "PASS" if m.all_passed else "FAIL"
        print(
            f"    Seed {seed} finished in {el_s:.2f}s | "
            f"K_EM={m.k_em:.2%} (FP={m.k_false_plastic_rate:.1%}) | "
            f"C_EM={m.c_em:.2%} (Act={m.c_action_accuracy:.1%}, Exp={m.c_expansion_rate:.1%}) | "
            f"N_EM={m.n_final_em:.2%} (Trig={m.n_plastic_trigger_rate:.1%}, "
            f"Prom={m.n_promotions_count}/{m.n_total}) | "
            f"R_EM={m.r_em:.2%} (Dir={m.r_direct_reuse_rate:.1%}, "
            f"Recons={m.r_reconsolidation_count}) | "
            f"Degrad={m.max_old_task_degradation:.2%} | Leaks={m.workspace_leak_count} | "
            f"Verdict={verdict}"
        )

    elapsed_total = time.perf_counter() - t0
    n_seeds = len(per_seed_metrics)

    # Aggregate means
    mean_k_em = sum(m.k_em for m in per_seed_metrics) / n_seeds
    mean_k_fp = sum(m.k_false_plastic_rate for m in per_seed_metrics) / n_seeds
    mean_c_em = sum(m.c_em for m in per_seed_metrics) / n_seeds
    mean_c_act = sum(m.c_action_accuracy for m in per_seed_metrics) / n_seeds
    mean_c_exp = sum(m.c_expansion_rate for m in per_seed_metrics) / n_seeds
    mean_n_em = sum(m.n_final_em for m in per_seed_metrics) / n_seeds
    mean_n_trig = sum(m.n_plastic_trigger_rate for m in per_seed_metrics) / n_seeds
    total_n_prom = sum(m.n_promotions_count for m in per_seed_metrics)
    total_n = sum(m.n_total for m in per_seed_metrics)
    mean_r_em = sum(m.r_em for m in per_seed_metrics) / n_seeds
    mean_r_dir = sum(m.r_direct_reuse_rate for m in per_seed_metrics) / n_seeds
    total_r_recons = sum(m.r_reconsolidation_count for m in per_seed_metrics)
    max_degrad = max(m.max_old_task_degradation for m in per_seed_metrics)
    total_leaks = sum(m.workspace_leak_count for m in per_seed_metrics)

    overall_passed = all(m.all_passed for m in per_seed_metrics)

    final_report = {
        "benchmark": "Phase A.2 Task A2-C008 Full Sequential Closed Loop",
        "seeds": list(config.seeds),
        "num_seeds": n_seeds,
        "episodes_per_seed": config.total_episodes,
        "total_episodes_evaluated": len(all_episodes),
        "overall_passed": overall_passed,
        "aggregate_metrics": {
            "mean_k_em": mean_k_em,
            "mean_k_false_plastic_rate": mean_k_fp,
            "mean_c_em": mean_c_em,
            "mean_c_action_accuracy": mean_c_act,
            "mean_c_expansion_rate": mean_c_exp,
            "mean_n_final_em": mean_n_em,
            "mean_n_plastic_trigger_rate": mean_n_trig,
            "total_n_promotions": total_n_prom,
            "total_n_tasks": total_n,
            "mean_r_em": mean_r_em,
            "mean_r_direct_reuse_rate": mean_r_dir,
            "total_r_reconsolidations": total_r_recons,
            "max_old_task_degradation": max_degrad,
            "total_workspace_leaks": total_leaks,
        },
        "acceptance_criteria": {
            "k_em": {
                "target": ">= 0.95",
                "achieved": f"{mean_k_em:.4f}",
                "passed": mean_k_em >= 0.95,
            },
            "k_false_plastic": {
                "target": "<= 0.10",
                "achieved": f"{mean_k_fp:.4f}",
                "passed": mean_k_fp <= 0.10,
            },
            "c_em": {
                "target": ">= 0.90",
                "achieved": f"{mean_c_em:.4f}",
                "passed": mean_c_em >= 0.90,
            },
            "c_action_accuracy": {
                "target": ">= 0.85",
                "achieved": f"{mean_c_act:.4f}",
                "passed": mean_c_act >= 0.85,
            },
            "c_expansion_rate": {
                "target": "<= 0.10",
                "achieved": f"{mean_c_exp:.4f}",
                "passed": mean_c_exp <= 0.10,
            },
            "n_final_em": {
                "target": ">= 0.90",
                "achieved": f"{mean_n_em:.4f}",
                "passed": mean_n_em >= 0.90,
            },
            "n_plastic_trigger": {
                "target": ">= 0.90",
                "achieved": f"{mean_n_trig:.4f}",
                "passed": mean_n_trig >= 0.90,
            },
            "n_promotions": {
                "target": "1:1 per task",
                "achieved": f"{total_n_prom}/{total_n}",
                "passed": total_n_prom == total_n,
            },
            "r_em": {
                "target": ">= 0.95",
                "achieved": f"{mean_r_em:.4f}",
                "passed": mean_r_em >= 0.95,
            },
            "r_direct_reuse": {
                "target": ">= 0.90",
                "achieved": f"{mean_r_dir:.4f}",
                "passed": mean_r_dir >= 0.90,
            },
            "r_no_reconsolidation": {
                "target": "0 reconsolidations",
                "achieved": f"{total_r_recons}",
                "passed": total_r_recons == 0,
            },
            "old_task_degradation": {
                "target": "<= 0.02",
                "achieved": f"{max_degrad:.4f}",
                "passed": max_degrad <= 0.02,
            },
            "no_temporary_leak": {
                "target": "0 leaks",
                "achieved": f"{total_leaks}",
                "passed": total_leaks == 0,
            },
        },
        "per_seed_results": [m.to_dict() for m in per_seed_metrics],
        "elapsed_seconds": elapsed_total,
    }

    if config.output_dir:
        report_json_path = config.output_dir / "report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        episodes_jsonl_path = config.output_dir / "episodes.jsonl"
        with open(episodes_jsonl_path, "w", encoding="utf-8") as f:
            for r in all_episodes:
                f.write(json.dumps(r.to_dict()) + "\n")

        gate_status_str = "YES (STOP GATE PASSED)" if overall_passed else "NO (STOP GATE FAILED)"
        md_lines = [
            "# Phase A.2 Task A2-C008: Full Sequential K/C/N/R Autonomous Closed Loop",
            "",
            f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Seeds Evaluated:** {list(config.seeds)}",
            (
                f"- **Episodes per Seed:** {config.total_episodes} "
                f"(K: {config.num_k}, C: {config.num_c}, N: {config.num_n}, R: {config.num_r})"
            ),
            f"- **Total Episodes:** {len(all_episodes)}",
            f"- **Elapsed Wall Time:** {elapsed_total:.2f}s",
            f"- **Overall Acceptance:** **{gate_status_str}**",
            "",
            "## Primary Acceptance Verification",
            "| Category / Criterion | Target Threshold | Measured Mean / Total | "
            "Seed Status | Verdict |",
            "|---|---|---|---|---|",
            (
                f"| **K (Known) Exact Match** | $\\ge 0.95$ | {mean_k_em:.2%} | "
                f"All $\\ge 0.95$ | {'PASS' if mean_k_em >= 0.95 else 'FAIL'} |"
            ),
            (
                f"| **K False Plastic Rate** | $\\le 0.10$ | {mean_k_fp:.2%} | "
                f"All $\\le 0.10$ | {'PASS' if mean_k_fp <= 0.10 else 'FAIL'} |"
            ),
            (
                f"| **C (Composition) Exact Match** | $\\ge 0.90$ | {mean_c_em:.2%} | "
                f"All $\\ge 0.90$ | {'PASS' if mean_c_em >= 0.90 else 'FAIL'} |"
            ),
            (
                f"| **C Action Accuracy** | $\\ge 0.85$ | {mean_c_act:.2%} | "
                f"All $\\ge 0.85$ | {'PASS' if mean_c_act >= 0.85 else 'FAIL'} |"
            ),
            (
                f"| **C Bank Expansion Rate** | $\\le 0.10$ | {mean_c_exp:.2%} | "
                f"All $\\le 0.10$ | {'PASS' if mean_c_exp <= 0.10 else 'FAIL'} |"
            ),
            (
                f"| **N (Novel) Final EM** | $\\ge 0.90$ | {mean_n_em:.2%} | "
                f"All $\\ge 0.90$ | {'PASS' if mean_n_em >= 0.90 else 'FAIL'} |"
            ),
            (
                f"| **N Plastic Trigger Rate** | $\\ge 0.90$ | {mean_n_trig:.2%} | "
                f"All $\\ge 0.90$ | {'PASS' if mean_n_trig >= 0.90 else 'FAIL'} |"
            ),
            (
                f"| **N Bank Expansion Consistency** | 1:1 match | {total_n_prom} / {total_n} | "
                f"All 1:1 | {'PASS' if total_n_prom == total_n else 'FAIL'} |"
            ),
            (
                f"| **R (Recurrence) Exact Match** | $\\ge 0.95$ | {mean_r_em:.2%} | "
                f"All $\\ge 0.95$ | {'PASS' if mean_r_em >= 0.95 else 'FAIL'} |"
            ),
            (
                f"| **R Direct Reuse Rate** | $\\ge 0.90$ | {mean_r_dir:.2%} | "
                f"All $\\ge 0.90$ | {'PASS' if mean_r_dir >= 0.90 else 'FAIL'} |"
            ),
            (
                f"| **R Reconsolidation Count** | 0 reconsolidations | {total_r_recons} | "
                f"All 0 | {'PASS' if total_r_recons == 0 else 'FAIL'} |"
            ),
            (
                f"| **Global Old-Task Degradation** | $\\le 0.02$ | {max_degrad:.2%} | "
                f"Max $\\le 0.02$ | {'PASS' if max_degrad <= 0.02 else 'FAIL'} |"
            ),
            (
                f"| **Global Workspace Leaks** | 0 leaks | {total_leaks} | "
                f"All 0 | {'PASS' if total_leaks == 0 else 'FAIL'} |"
            ),
            "",
            "## Per-Seed Summary",
            "| Seed | K EM (FP) | C EM (Act, Exp) | N EM (Trig, Prom) | "
            "R EM (Dir, Recons) | Max Degrad | Leaks | Status |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m in per_seed_metrics:
            md_lines.append(
                f"| Seed {m.seed} | {m.k_em:.1%} ({m.k_false_plastic_rate:.1%}) | "
                f"{m.c_em:.1%} ({m.c_action_accuracy:.1%}, {m.c_expansion_rate:.1%}) | "
                f"{m.n_final_em:.1%} ({m.n_plastic_trigger_rate:.1%}, "
                f"{m.n_promotions_count}/{m.n_total}) | "
                f"{m.r_em:.1%} ({m.r_direct_reuse_rate:.1%}, {m.r_reconsolidation_count}) | "
                f"{m.max_old_task_degradation:.2%} | {m.workspace_leak_count} | "
                f"{'PASS' if m.all_passed else 'FAIL'} |"
            )

        md_path = config.output_dir / "BENCHMARK_REPORT.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

    return final_report
