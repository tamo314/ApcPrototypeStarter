"""T32-R: executed-transition logging, shared router features, and prefix-replay branching.

One execution path is used for evaluation, diagnosis and acquisition:
``policy.action() -> env.step() -> policy.after_step()`` and the record is written
immediately afterwards, including the terminal step.  The router and candidate
inputs stored in each record are computed by the same functions that the
selector uses at execution time, so learning never reconstructs them.

Success is only the unbroken ``success_hold`` (default 20) step streak of the
task's own ``success`` signal.  Nothing is back-filled: values that the executor
did not produce (for example IK flags in base mode) are stored as ``None``.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .primitive_learning import SCHEMA_V5, LearnedSelector, features
from .primitive_policy import GraspRecoveryGuardSelector, PrimitivePolicy
from .primitive_tree import CART
from .primitives import CONTINUE, HOLD, NAMES20
from .runner import RunConfig, array, make_env, scalar

ROUTER_FEATURE_NAMES = (
    "raw_is_4", "grasped", "cube_lift_z", "cube_rel_goal_z", "goal_z", "dist_xy_to_goal",
    "dist_z_to_goal", "cube_z", "hand_dist_to_cube_xy", "hand_dist_to_cube_z",
    "gripper_target", "raw_not_5_or_7",
)
MODULES = ("base", "grasp_recovery", "transit", "place")
MODULES_EXT = MODULES + ("candidate_A",)
ROUTER_V2_EXTRA = ("stall_steps_div100", "goal_minus_cube_root_x", "goal_minus_cube_root_y",
                   "last_step_ik_rejected")
TASK_ENVS = {"pick": "APC-FetchPickCubeFar-v1", "place": "APC-FetchPlaceCubeFar-v1",
             "true_place": "APC-FetchTruePlaceFar-v1"}
OVERRIDE_REASONS = {0: "none", 1: "target_timeout", 2: "ik_or_joint_or_table_rejected",
                    3: "base_path_rejected"}


def router_features(state: dict, base_proposed_id: int) -> np.ndarray:
    """The 12 router inputs, from the pre-action state and the Base proposal.

    Identical arithmetic to ``UnifiedRouterSelector._extract_router_features``;
    it is the only extractor used for both router training and execution here.
    """
    cube = np.asarray(state["cube_position"], dtype=np.float32)
    goal = np.asarray(state["goal_position"], dtype=np.float32)
    hand = np.asarray(state["measured_hand_position"], dtype=np.float32)
    lift = float(cube[2] - float(state["cube_initial_z"]))
    return np.array([
        float(base_proposed_id == 4), float(bool(state["grasped"])), lift,
        float(cube[2] - goal[2]), goal[2], float(np.linalg.norm(cube[:2] - goal[:2])),
        float(abs(cube[2] - goal[2])), cube[2], float(np.linalg.norm(hand[:2] - cube[:2])),
        float(abs(hand[2] - (cube[2] + 0.012))), float(state["gripper_target_m"]),
        float(base_proposed_id not in (5, 7)),
    ], dtype=np.float32)


class StallTracker:
    """Observation-history feature: steps held without >=3 mm horizontal progress.

    Same rule as the detector's far-transit stagnation counter, evaluated on the
    pre-action state of every control step (resets while not grasped).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.min_dist = np.inf
        self.count = 0

    def update(self, grasped: bool, dist: float) -> int:
        if not grasped:
            self.reset()
        elif dist < self.min_dist - 0.003:
            self.min_dist = dist
            self.count = 0
        else:
            self.count += 1
        return self.count


def router_features_v2(state: dict, base_proposed_id: int, stall: int) -> np.ndarray:
    """v1 features + stall history, goal direction in the robot frame, last rejection."""
    root = np.asarray(state["root_rotation"], dtype=np.float64)
    delta = root.T @ (np.asarray(state["goal_position"], np.float64) - np.asarray(state["cube_position"], np.float64))
    extra = [stall / 100.0, float(delta[0]), float(delta[1]),
             float(state.get("last_override_reason_code", 0) == 2)]
    return np.concatenate([router_features(state, base_proposed_id), np.asarray(extra, np.float32)])


def router_x2_from_rows(rows: list[dict]) -> list[list[float]]:
    """Offline recomputation of router v2 inputs from one episode's logged rows.

    Uses only quantities the online selector saw: the logged v1 inputs, the
    pre-action distance/grasp sequence, the schema-v5 goal-cube root-frame delta
    and the previous step's override code.
    """
    tracker = StallTracker()
    out, previous = [], 0
    for r in sorted(rows, key=lambda r: r["control_step"]):
        rc = r["reward_components"]
        stall = tracker.update(bool(rc["grasped_before"]), float(rc["cube_goal_dist_xy_before_m"]))
        x = list(r["router_x"]) + [stall / 100.0, r["candidate_x"][3], r["candidate_x"][4],
                                   float(previous == 2)]
        out.append(x)
        previous = r["override_reason_code"]
    return out


def candidate_features(state: dict) -> np.ndarray:
    """Candidate/Base input (schema v5), identical to ``LearnedSelector``."""
    return features(state, schema=SCHEMA_V5, primitive_count=20)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Bank:
    """Paths of one bank version. Hash is over role names and file hashes."""

    base: Path
    router: Path
    transit: Path | None = None
    place: Path | None = None
    candidate_A: Path | None = None
    candidates: dict = field(default_factory=dict)  # further candidate_* roles
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_roles(cls, root: Path, roles: dict) -> "Bank":
        fixed = {k: Path(root) / v for k, v in roles.items()
                 if k in ("base", "router", "transit", "place", "candidate_A")}
        more = {k: Path(root) / v for k, v in roles.items() if k.startswith("candidate_") and k != "candidate_A"}
        return cls(**fixed, candidates=more)

    def candidate_roles(self) -> dict[str, Path]:
        out = {"candidate_A": self.candidate_A} if self.candidate_A is not None else {}
        return {**out, **{k: Path(v) for k, v in self.candidates.items()}}

    def files(self) -> dict[str, Path]:
        items = {"base": self.base, "router": self.router, "transit": self.transit,
                 "place": self.place, "candidate_A": self.candidate_A, **self.candidates}
        return {k: Path(v) for k, v in items.items() if v is not None}

    def manifest(self) -> dict:
        files = {k: {"path": str(v), "sha256": sha256_file(v), "bytes": v.stat().st_size}
                 for k, v in self.files().items()}
        digest = hashlib.sha256(json.dumps({k: v["sha256"] for k, v in sorted(files.items())},
                                           sort_keys=True).encode()).hexdigest()
        return {"bank_hash": digest, "files": files, **self.extra}

    @classmethod
    def initial(cls, bundle: Path = Path("dist_autonomous_bundle_v1")) -> "Bank":
        return cls(base=bundle / "base_selector.pt", router=bundle / "unified_router.pt",
                   transit=bundle / "transit_candidate.pt", place=bundle / "place_candidate.pt")


def load_router(path: Path) -> CART:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(ckpt.get("class_names", MODULES))
    tree = CART(ckpt["n_nodes"], len(classes))
    tree.load_state_dict(ckpt["state_dict"])
    tree.eval()
    tree.feature_schema = ckpt.get("feature_schema", "router_v1")
    tree.class_names = classes
    return tree


def save_router(path: Path, tree: CART, *, feature_schema="router_v1", class_names=MODULES, **meta) -> None:
    names = list(ROUTER_FEATURE_NAMES) + (list(ROUTER_V2_EXTRA) if feature_schema == "router_v2" else [])
    torch.save({"schema": "learned_unified_router_cart_v1", "feature_schema": feature_schema,
                "feature_names": names, "class_names": list(class_names), "state_dict": tree.state_dict(),
                "n_nodes": int(tree.feature.shape[0]), **meta}, path)


class RoutedSelector:
    """Unified-router execution with a per-step decision record.

    Default behaviour reproduces ``UnifiedRouterSelector`` step for step (place
    latch, recovery safety layer before the router).  Diagnostic hooks:

    * ``force(step, state) -> module | None``: select that module's selector after
      the recovery safety layer; the forced module still goes through the executor
      and its IK/joint/table checks.
    * ``script``: a queue of primitive IDs executed verbatim (diagnostic reference).
    * ``place_gate(step, state, features) -> bool | None``: consulted only when the
      router proposes Place; False keeps Base control for this step (veto/delay).
    * ``place_release(step, state) -> bool``: allow leaving the place latch.
    """

    def __init__(self, base, recovery, modules: dict, router: CART, *, force=None,
                 place_gate=None, place_release=None):
        self.base_selector = base
        self.recovery = recovery
        self.modules = dict(modules)
        self.router = router
        self.force = force
        self.place_gate = place_gate
        self.place_release = place_release
        self.script: list[int] = []
        self.metadata = dict(base.metadata, learned=True, selector="routed_selector_r1",
                             router_nodes=int(router.feature.shape[0]))
        self.reset()

    @property
    def primitive_count(self):
        return self.base_selector.primitive_count

    def reset(self):
        for sel in [self.base_selector, self.recovery, *self.modules.values()]:
            if sel is not None and hasattr(sel, "reset"):
                sel.reset()
        self.stall = StallTracker()
        self.last_step = None
        self.latched_place = False
        self.active_module = "base"
        self.last_scores = []
        self.decision: dict = {}
        self.script = []

    def _run(self, module, step, observation):
        selector = self.modules[module]
        action = int(selector.select(step, observation))
        self.last_scores = getattr(selector, "last_scores", [])
        return action

    def select(self, step, observation):
        raw = int(self.base_selector.select(step, observation))
        self.last_scores = getattr(self.base_selector, "last_scores", [])
        cube, goal = np.asarray(observation["cube_position"]), np.asarray(observation["goal_position"])
        stall = self.stall.update(bool(observation["grasped"]), float(np.linalg.norm(cube[:2] - goal[:2])))
        x = router_features(observation, raw)
        x2 = router_features_v2(observation, raw, stall)
        with torch.no_grad():
            xin = x2 if getattr(self.router, "feature_schema", "router_v1") == "router_v2" else x
            router_class = int(self.router(torch.from_numpy(xin)).argmax())
        d = dict(base_proposed_id=raw, router_class=router_class, router_x=x.tolist(),
                 router_x2=x2.tolist(), forced=False, scripted=False, latch_released=False,
                 committed=False)
        self.decision = d
        contiguous = self.last_step is not None and step == self.last_step + 1
        self.last_step = step
        if self.script:
            d.update(scripted=True, selected_module="script")
            self.active_module = "script"
            d["proposed_id"] = int(self.script.pop(0))
            return d["proposed_id"]
        if self.latched_place and self.place_release is not None and self.place_release(step, observation):
            self.latched_place = False
            d["latch_released"] = True
        if self.latched_place and self.modules.get("place") is not None:
            return self._finish(d, "place", self._run("place", step, observation))
        grasped = bool(observation.get("grasped", False))
        if not grasped and self.recovery is not None:
            act = int(self.recovery.select(step, observation))
            if self.recovery.last_guard_triggered:
                self.last_scores = getattr(self.recovery, "last_scores", [])
                return self._finish(d, "grasp_recovery", act)
        # Option commitment: a started option runs its holds (recovery layer above
        # still pre-empts when the grasp is lost).
        option = self.modules.get(self.active_module)
        if (self.active_module.startswith("candidate_") and option is not None and contiguous
                and getattr(option, "queue", None)):
            d["committed"] = True
            return self._finish(d, self.active_module, self._run(self.active_module, step, observation))
        forced = self.force(step, observation) if self.force is not None else None
        if forced is not None:
            d["forced"] = True
            if forced == "base":
                return self._finish(d, "base", raw)
            if forced == "place":
                self.latched_place = True
            return self._finish(d, forced, self._run(forced, step, observation))
        pred = router_class
        if self.place_gate is not None and pred == 3:
            # Veto-type gate: only when the router proposes Place, decide "enter now"
            # or "keep the existing control" (Base) for this step.
            gate = self.place_gate(step, observation, x)
            d["place_gate"] = None if gate is None else bool(gate)
            if gate is False:
                pred = 0
        if pred == 3 and self.modules.get("place") is not None:
            self.latched_place = True
            return self._finish(d, "place", self._run("place", step, observation))
        if pred == 2 and self.modules.get("transit") is not None:
            return self._finish(d, "transit", self._run("transit", step, observation))
        name = self.router.class_names[pred] if pred < len(getattr(self.router, "class_names", MODULES)) else None
        if pred >= 4 and name in self.modules and self.modules[name] is not None:
            return self._finish(d, name, self._run(name, step, observation))
        if pred == 1 and self.recovery is not None:
            act = int(self.recovery.select(step, observation))
            self.last_scores = getattr(self.recovery, "last_scores", [])
            return self._finish(d, "grasp_recovery", act)
        return self._finish(d, "base", raw)

    def _finish(self, d, module, action):
        d.update(selected_module=module, proposed_id=int(action))
        self.active_module = module
        return int(action)

    # Mutable selector state for prefix bookkeeping (not used for physics restore).
    def state(self):
        rec = self.recovery
        return dict(latched_place=self.latched_place,
                    recovery=None if rec is None else dict(closing_steps=rec.closing_steps,
                                                           recovery_mode=rec.recovery_mode))


# Option interface: an existing ID followed by executor-settling CONTINUE holds.
HOLDS = {**{i: 8 for i in range(6)}, **{i: 8 for i in range(10, 16)}, 16: 25, 17: 25, 18: 25,
         19: 25, HOLD: 8}


def option_ids(option) -> list[int]:
    primitive, repeats = (option, 1) if isinstance(option, int) else option
    return op(primitive, HOLDS[primitive]) * int(repeats)


class OptionSelector:
    """Candidate that picks an option (ID, repeats) at decision points.

    The CART sees the schema-v5 state at the first step of an option; the
    remaining steps of the option are its fixed CONTINUE holds.  A gap in the
    step sequence (the router selected another module) drops the queue.
    """

    def __init__(self, checkpoint: Path):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck.get("model_kind") != "option_cart":
            raise ValueError("Not an option candidate checkpoint")
        self.options = [tuple(o) for o in ck["options"]]
        self.tree = CART(ck["tree_nodes"], len(self.options))
        self.tree.load_state_dict(ck["state_dict"])
        self.mean, self.std = ck["mean"], ck["std"]
        self.metadata = dict(selector="option_cart_v1", learned=True, tree_nodes=ck["tree_nodes"],
                             checkpoint_sha256=sha256_file(checkpoint))
        self.primitive_count = 20
        self.reset()

    def reset(self):
        self.queue: list[int] = []
        self.last_step = None
        self.last_scores = []
        self.last_option = None

    def select(self, step, observation):
        if self.last_step is None or step != self.last_step + 1:
            self.queue = []
        self.last_step = step
        if not self.queue:
            x = (candidate_features(observation) - self.mean) / self.std
            with torch.no_grad():
                scores = self.tree(torch.from_numpy(x.astype(np.float32)))
            self.last_scores = scores.tolist()
            self.last_option = self.options[int(scores.argmax())]
            self.queue = option_ids(self.last_option)
        return int(self.queue.pop(0))


def save_option_candidate(path: Path, tree: CART, mean, std, options, *, source: dict) -> dict:
    torch.save({"schema": SCHEMA_V5, "model_kind": "option_cart", "options": [list(o) for o in options],
                "holds": {str(k): v for k, v in HOLDS.items()}, "state_dict": tree.state_dict(),
                "tree_nodes": int(tree.feature.shape[0]), "mean": mean, "std": std, "source": source}, path)
    return dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size,
                tree_nodes=int(tree.feature.shape[0]))


def load_module(path: Path, meta: dict, freq: float):
    """LearnedSelector for per-step checkpoints, OptionSelector for option CARTs."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("model_kind") == "option_cart":
        return OptionSelector(Path(path))
    return LearnedSelector(Path(path), meta, freq, strict_task=False)


def make_task_env(task: str, out: Path, seed: int = 0):
    config = RunConfig(env_id=TASK_ENVS[task], robot_uids="fetch", policy="external", episodes=1,
                       seed=seed, max_steps=100000, env_max_steps=100000)
    out.mkdir(parents=True, exist_ok=True)
    return make_env(config, out)


def build_policy(env, out: Path, bank: Bank, *, recovery=True, force=None, place_gate=None,
                 place_release=None, transit_selector=None):
    meta = env.unwrapped.experiment_metadata()
    freq = float(env.unwrapped.sim_config.control_freq)

    def learned(path):
        return load_module(Path(path), meta, freq)

    base = learned(bank.base)
    modules = {"transit": transit_selector if transit_selector is not None else
               (learned(bank.transit) if bank.transit is not None else None),
               "place": learned(bank.place) if bank.place is not None else None,
               **{name: learned(path) for name, path in bank.candidate_roles().items()}}
    guard = GraspRecoveryGuardSelector(base) if recovery else None
    selector = RoutedSelector(base, guard, modules, load_router(bank.router), force=force,
                              place_gate=place_gate, place_release=place_release)
    return PrimitivePolicy(env, out, selector=selector, allow_rotation=True, allow_base=True)


def _vec(x):
    return np.asarray(x, dtype=np.float64)


def _task_success(env) -> tuple[bool, dict]:
    info = env.unwrapped.evaluate()
    keep = {}
    for key in ("success", "is_grasped", "is_obj_placed_surface", "is_released", "is_obj_static",
                "is_robot_static", "cube_goal_dist_xy_m", "is_obj_placed"):
        if key in info:
            keep[key] = scalar(info[key])
    return bool(keep.get("success", False)), keep


class EpisodeLog:
    """Streams transition records of one episode to a JSONL file (may be None)."""

    def __init__(self, path: Path | None, *, run_id: str, episode_id: str, seed: int, task: str,
                 bank_hash: str, event_id: str | None = None, source_event_hash: str | None = None,
                 store_states: bool = True):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8") if path is not None else None
        self.common = dict(run_id=run_id, episode_id=episode_id, environment_seed=seed, task=task,
                           event_id=event_id, source_event_hash=source_event_hash,
                           active_bank_hash=bank_hash)
        self.store_states = store_states

    def write(self, record: dict) -> None:
        if self.handle is not None:
            self.handle.write(json.dumps(dict(self.common, **record), allow_nan=False) + "\n")

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    return value


def transition_record(step: int, report: dict, decision: dict, reward: float, success: bool,
                      eval_info: dict, consecutive: int, terminated: bool, truncated: bool,
                      elapsed_seconds: float, store_states: bool) -> dict:
    s0, s1 = report["pre_action_state"], report["post_action_state"]
    base_mode = bool(report.get("ik_skipped_base_mode", False))
    goal = _vec(s1["goal_position"])
    d0 = float(np.linalg.norm(_vec(s0["cube_position"])[:2] - goal[:2]))
    d1 = float(np.linalg.norm(_vec(s1["cube_position"])[:2] - goal[:2]))
    base0, base1 = _vec(s0["base_pose"]), _vec(s1["base_pose"])
    yaw = float(np.arctan2(np.sin(base1[2] - base0[2]), np.cos(base1[2] - base0[2])))
    record = dict(
        control_step=step,
        base_proposed_id=decision.get("base_proposed_id"),
        router_class=decision.get("router_class"),
        selected_module=decision.get("selected_module"),
        forced=decision.get("forced", False), scripted=decision.get("scripted", False),
        latch_released=decision.get("latch_released", False),
        place_gate=decision.get("place_gate"),
        proposed_id=int(report["proposed_id"]), executed_id=int(report["executed_id"]),
        submitted_action=report["submitted_action"],
        override_reason_code=int(report["override_reason_code"]),
        override_reason=OVERRIDE_REASONS[int(report["override_reason_code"])],
        interruption_reason_code=int(report["interruption_reason_code"]),
        mode_before=int(report["mode_before_code"]), mode_after=int(report["mode_after_code"]),
        # IK flags exist only when the hand executor ran; base mode records None.
        ik_success=None if base_mode else report.get("ik_success"),
        ik_within_limits=None if base_mode else report.get("ik_within_limits"),
        ik_table_clear=None if base_mode else report.get("ik_table_clear"),
        ik_position_error_m=None if base_mode else report.get("ik_position_error_m"),
        base_step_clear=report.get("base_step_clear") if base_mode else None,
        target_updated=(report["attempted_target_position"] != report["target_after_update"]),
        attempted_target_position=report["attempted_target_position"],
        target_after_update=report["target_after_update"],
        router_x=decision.get("router_x"), router_x2=decision.get("router_x2"),
        committed=decision.get("committed", False),
        teacher_id=decision.get("teacher_id"),
        candidate_x=candidate_features(s0).tolist(),
        actual_displacements=dict(
            hand_m=(_vec(s1["measured_hand_position"]) - _vec(s0["measured_hand_position"])).tolist(),
            cube_m=(_vec(s1["cube_position"]) - _vec(s0["cube_position"])).tolist(),
            base_xy_m=(base1[:2] - base0[:2]).tolist(), base_yaw_rad=yaw,
            cube_goal_xy_progress_m=d0 - d1),
        reward_components=dict(reward=reward, task_success=success, **eval_info,
                               cube_goal_dist_xy_before_m=d0, cube_goal_dist_xy_after_m=d1,
                               grasped_before=bool(s0["grasped"]), grasped_after=bool(s1["grasped"]),
                               cube_lift_after_m=float(s1["cube_position"][2] - s1["cube_initial_z"]),
                               table_contact_force_n=report.get("table_contact_force_norm_sum_n")),
        consecutive_success=consecutive, terminated=terminated, truncated=truncated,
        elapsed_control_steps=step + 1, elapsed_seconds=elapsed_seconds,
    )
    if store_states:
        record["s_t"] = s0
        record["s_tp1"] = s1
    return _clean(record)


@dataclass
class EpisodeResult:
    seed: int
    steps: int
    success: bool
    first_success_step: int | None
    max_consecutive: int
    wall_seconds: float
    executed_counts: dict
    module_counts: dict
    rejection_counts: dict
    final: dict
    events: list
    stopped_early: bool = False

    def as_dict(self):
        return dict(self.__dict__)


def run_episode(env, policy, *, seed: int, max_steps: int = 1200, log: EpisodeLog | None = None,
                success_hold: int = 20, detector=None, on_step: Callable | None = None,
                stop_after: int | None = None, reset: bool = True,
                initial_script: list | None = None) -> EpisodeResult:
    """Run (or continue) one episode on the shared execution path.

    Prefixes are reproduced by re-running the same policy from reset (verified
    deterministic on PhysX CPU; every replayed step is a counted env step).
    ``on_step(step, record, policy)`` may set scripts or return ``"stop"``.  ``stop_after`` ends the call after that many steps
    (without marking truncation) for branch prefixes.
    """
    start = time.monotonic()
    if reset:
        env.reset(seed=seed)
        policy.reset()
        if detector is not None:
            detector.reset()
    selector = policy.selector
    if initial_script:
        selector.script = list(initial_script)
    consecutive = max_consecutive = 0
    first_success = None
    executed, modules, rejections = {}, {}, {}
    events = []
    step = policy.step
    stopped = False
    record = {}
    while step < max_steps:
        action = policy.action()
        _, reward, _, _, _ = env.step(action)
        report = policy.after_step()
        success, eval_info = _task_success(env)
        consecutive = consecutive + 1 if success else 0
        max_consecutive = max(max_consecutive, consecutive)
        if success and first_success is None:
            first_success = step
        terminated = consecutive >= success_hold
        truncated = (not terminated) and step + 1 >= max_steps
        record = transition_record(step, report, getattr(selector, "decision", {}),
                                   float(scalar(reward)), success, eval_info, consecutive,
                                   terminated, truncated, time.monotonic() - start,
                                   log.store_states if log is not None else False)
        ex = record["executed_id"]
        executed[ex] = executed.get(ex, 0) + 1
        mod = record["selected_module"]
        modules[mod] = modules.get(mod, 0) + 1
        if record["override_reason_code"]:
            key = record["override_reason"]
            if record["override_reason_code"] == 2:
                key += ":" + ("ik_not_converged" if not record["ik_success"] else
                              "joint_limit" if not record["ik_within_limits"] else "table_clearance")
            rejections[key] = rejections.get(key, 0) + 1
        if detector is not None:
            diag = detector.update(report["post_action_state"], ex, record["override_reason_code"])
            record["detector_status"] = diag.status
            if diag.trigger_adaptation and not events:
                events.append(dict(step=step, status=diag.status, reason=diag.reason,
                                   streak_steps=diag.streak_steps))
        if log is not None:
            log.write(record)
        step += 1
        if on_step is not None and on_step(step - 1, record, policy) == "stop":
            stopped = True
            break
        if terminated or truncated:
            break
        if stop_after is not None and step >= stop_after:
            stopped = True
            break
    s1 = policy._state()
    goal = np.asarray(s1["goal_position"])
    final = dict(cube_position=s1["cube_position"], goal_position=s1["goal_position"],
                 cube_goal_dist_xy_m=float(np.linalg.norm(np.asarray(s1["cube_position"])[:2] - goal[:2])),
                 grasped=bool(s1["grasped"]), base_pose=s1["base_pose"],
                 hand_position=s1["measured_hand_position"])
    return EpisodeResult(seed=seed, steps=step, success=max_consecutive >= success_hold,
                         first_success_step=first_success, max_consecutive=max_consecutive,
                         wall_seconds=time.monotonic() - start, executed_counts=executed,
                         module_counts=modules, rejection_counts=rejections, final=_clean(final),
                         events=events, stopped_early=stopped)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_transitions(path: Path) -> dict:
    """Success/cost summary recomputed from the transition log itself."""
    rows = read_jsonl(path)
    episodes: dict[str, dict] = {}
    for r in rows:
        e = episodes.setdefault(r["episode_id"], dict(seed=r["environment_seed"], task=r["task"],
                                                      steps=0, success=False, max_consecutive=0,
                                                      wall_seconds=0.0, rejections=0,
                                                      last_step=-1, terminal_recorded=False))
        e["steps"] += 1
        e["max_consecutive"] = max(e["max_consecutive"], r["consecutive_success"])
        e["success"] = e["success"] or r["terminated"]
        e["wall_seconds"] = max(e["wall_seconds"], r["elapsed_seconds"])
        e["rejections"] += int(r["override_reason_code"] != 0)
        e["last_step"] = max(e["last_step"], r["control_step"])
        e["terminal_recorded"] = e["terminal_recorded"] or r["terminated"] or r["truncated"]
    return dict(records=len(rows), episodes=episodes,
                env_steps=sum(e["steps"] for e in episodes.values()),
                successes=sum(e["success"] for e in episodes.values()),
                wall_seconds=sum(e["wall_seconds"] for e in episodes.values()))


def fit_cart(x, y, output_dim, *, weights=None, max_depth=6, min_leaf=2, normalize=True):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if normalize:
        mean = x.mean(0)
        std = x.std(0)
        std[std < 1e-4] = 1.0
    else:
        mean, std = np.zeros(x.shape[1], np.float32), np.ones(x.shape[1], np.float32)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    tree = CART.fit((x - mean) / std, y, w, output_dim, max_depth=max_depth, min_leaf=min_leaf)
    return tree, mean.astype(np.float32), std.astype(np.float32)


def save_candidate(path: Path, tree: CART, mean, std, *, task_meta: dict, source: dict) -> dict:
    ckpt = {"schema": SCHEMA_V5, "primitive_names": list(NAMES20), "task": task_meta,
            "control_freq": 20.0, "state_dict": tree.state_dict(), "mean": mean, "std": std,
            "source_steps_sha256": source.get("sha256", ""), "source": source,
            "model_kind": "cart", "tree_nodes": int(tree.feature.shape[0]), "updates": 0,
            "allow_parameterized_goal_tasks": True}
    torch.save(ckpt, path)
    return dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size,
                tree_nodes=int(tree.feature.shape[0]))


def op(primitive: int, hold: int = 1) -> list[int]:
    """One primitive followed by ``hold - 1`` CONTINUE steps (target tracking)."""
    return [int(primitive)] + [CONTINUE] * (hold - 1)


__all__ = ["ROUTER_FEATURE_NAMES", "MODULES", "Bank", "RoutedSelector", "router_features",
           "candidate_features", "build_policy", "make_task_env", "run_episode", "EpisodeLog",
           "summarize_transitions", "read_jsonl", "fit_cart", "save_candidate", "save_router",
           "load_router", "op", "CONTINUE", "HOLD"]
