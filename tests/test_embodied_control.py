"""Learning, composition, causal controls, safe installation and reproducibility."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from apc.embodied.bundle import load_bundle, save_bundle
from apc.embodied.cli import main
from apc.embodied.control import (
    EmbodiedPrimitiveCall,
    LinearMotionPrimitive,
    MotionBank,
    MotionCompositionLibrary,
    MotionRecipe,
    run_episode,
    task_fingerprint,
    try_library,
)
from apc.embodied.learning import consolidate_after_failure, fit_from_demonstrations
from apc.embodied.viewer import write_replay
from apc.embodied.world import EmbodiedEnv, SeedSplit, WorldConfig, make_task


@pytest.fixture(scope="module")
def fitted():
    config = WorldConfig()
    return tuple(fit_from_demonstrations(config, tuple(range(10, 18)), spatial=spatial, name=name)
                 for spatial, name in ((False, "planar"), (True, "spatial")))


def setup_bank(fitted):
    bank, library = MotionBank("numpy"), MotionCompositionLibrary()
    for fit in fitted:
        bank.add(fit.primitive)
        library.add(MotionRecipe(f"repeat_{fit.primitive.name}", (fit.primitive.name,)))
    return bank, library


def test_call_arguments_are_immutable_and_validated():
    target = [1, 2, 3]
    call = EmbodiedPrimitiveCall("move", {"target": target})
    target[0] = 99
    assert call.arguments["target"] == (1, 2, 3)
    with pytest.raises(TypeError):
        call.arguments["target"] = (0, 0, 0)
    for arguments in ({}, {"target": [1, 2]}, {"target": [1, 2, 3], "task_id": 1}):
        with pytest.raises(ValueError):
            EmbodiedPrimitiveCall("move", arguments)


def test_demonstrations_fit_new_weights_and_need_excitation(fitted):
    planar, spatial = fitted
    assert planar.report["feature_rank"] == 5
    assert spatial.report["feature_rank"] == 7
    assert spatial.report["train_mse"] < 1e-12
    assert spatial.report["persistent_parameters_touched"] == 0
    assert spatial.primitive.fingerprint() != planar.primitive.fingerprint()
    with pytest.raises(RuntimeError, match="excite"):
        fit_from_demonstrations(WorldConfig(), [10, 11], spatial=True, name="bad")


@pytest.mark.parametrize("family", ["planar", "spatial", "route", "detour"])
@pytest.mark.parametrize("seed", [1000, 1003])
def test_reusable_controller_on_unseen_goals_and_compositions(fitted, family, seed):
    bank, library = setup_bank(fitted)
    before = bank.fingerprint()
    results = try_library(WorldConfig(), make_task(seed, family), bank, library)
    assert len(results) == 1 and results[0].success
    assert results[0].collisions == 0
    assert results[0].primitive_calls["planar"] == 0
    assert results[0].primitive_calls["spatial"] == results[0].control_steps
    assert results[0].max_active_parameters == 21
    assert bank.resident_parameters == 42
    assert bank.fingerprint() == before


def test_old_skill_fails_on_altitude_then_candidate_installs(fitted):
    config = WorldConfig()
    bank, library = setup_bank(fitted[:1])
    novel = make_task(1000, "spatial")
    failure = try_library(config, novel, bank, library)[-1]
    assert not failure.success and failure.reason == "option_timeout"
    before = bank.fingerprint()
    shadow = [make_task(1001, f) for f in ("planar", "spatial", "route", "detour")]
    result = consolidate_after_failure(config, failure, fitted[1].primitive, bank, library, shadow)
    assert result["installed"] and result["stable_weights_unchanged"]
    assert bank.fingerprint()["planar"] == before["planar"]
    assert try_library(config, novel, bank, library)[-1].success


def test_shadow_rejection_does_not_change_bank(fitted):
    config = WorldConfig()
    bank, library = setup_bank(fitted[:1])
    task = make_task(1000, "spatial")
    failure = try_library(config, task, bank, library)[-1]
    before = bank.fingerprint(), library.records()
    bad = LinearMotionPrimitive("bad", np.zeros((3, 7)))
    report = consolidate_after_failure(config, failure, bad, bank, library, [task])
    assert report["status"] == "REJECTED" and not report["installed"]
    assert before == (bank.fingerprint(), library.records())
    with pytest.raises(ValueError):
        consolidate_after_failure(config, failure, bad, bank, library, [])


def test_solved_tasks_cannot_trigger_adaptation(fitted):
    bank, library = setup_bank(fitted[:1])
    task = make_task(1000, "planar")
    success = try_library(WorldConfig(), task, bank, library)[-1]
    with pytest.raises(ValueError, match="failure"):
        consolidate_after_failure(
            WorldConfig(), success, fitted[1].primitive, bank, library, [task])


def test_negative_controls_cannot_change_true_goal(fitted):
    bank, library = setup_bank(fitted)
    task = make_task(1000, "spatial")
    calls = library.recipes()[0].bind(task)
    correct = run_episode(WorldConfig(), task, bank, calls)
    none = run_episode(WorldConfig(), task, bank, calls, control="none")
    wrong = run_episode(WorldConfig(), task, bank, calls, control="wrong_arguments")
    assert correct.success and not none.success and not wrong.success
    assert sum(none.primitive_calls.values()) == 0
    assert none.max_active_parameters == 0
    assert none.task == correct.task == wrong.task


def test_mixed_recipe_has_separate_weight_free_storage(fitted):
    bank, _ = setup_bank(fitted)
    task = make_task(1000, "route")
    recipe = MotionRecipe("mixed", ("spatial", "spatial", "spatial"))
    result = run_episode(WorldConfig(), task, bank, recipe.bind(task))
    assert result.success and bank.resident_parameters == 42
    with pytest.raises(ValueError):
        MotionRecipe("mismatch", ("spatial", "planar")).bind(task)
    with pytest.raises(ValueError):
        bank.add(fitted[0].primitive)


def test_goal_translation_and_rotation_equivariance(fitted):
    policy = fitted[1].primitive
    env = EmbodiedEnv()
    task = make_task(1000, "spatial")
    obs, _ = env.reset(task)
    action = policy.act(obs, EmbodiedPrimitiveCall(policy.name, {"target": task.goals[0]}))
    assert np.isfinite(action).all() and np.linalg.norm(action[:3]) > 0
    assert np.all(action[3:] == 0)
    shift = np.array([0.4, -0.2, 0.3])
    shifted = replace(obs, position=tuple(np.array(obs.position)+shift))
    shifted_call = EmbodiedPrimitiveCall(
        policy.name, {"target": np.array(task.goals[0])+shift})
    assert np.allclose(policy.act(shifted, shifted_call), action, atol=1e-12)
    theta = 0.8
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    angle = 2*np.arctan2(obs.orientation[3], obs.orientation[0])+theta
    rotated = replace(obs, position=tuple(rot @ obs.position),
                      velocity=tuple(rot @ obs.velocity),
                      orientation=(np.cos(angle/2), 0, 0, np.sin(angle/2)))
    rotated_call = EmbodiedPrimitiveCall(policy.name, {"target": rot @ task.goals[0]})
    assert np.allclose(policy.act(rotated, rotated_call), action, atol=1e-12)
    # A copied weight array cannot mutate persistent policy state.
    before = policy.fingerprint()
    policy.weights()[:] = 0
    assert policy.fingerprint() == before


def test_bundle_roundtrip_and_corruption(tmp_path, fitted):
    bank, library = setup_bank(fitted)
    config, splits = WorldConfig(), SeedSplit()
    task = make_task(1000)
    path = tmp_path / "bank.json"
    save_bundle(path, config, splits, bank, library, [task_fingerprint(task)])
    c, s, b, lib, exposure = load_bundle(path, backend="numpy")
    assert s == splits and task_fingerprint(task) in exposure
    a = try_library(config, task, bank, library, record_trace=True)[-1]
    fresh = try_library(c, task, b, lib, record_trace=True)[-1]
    assert a.trace == fresh.trace and b.fingerprint() == bank.fingerprint()
    with pytest.raises(FileExistsError):
        save_bundle(path, config, splits, bank, library, [task_fingerprint(task)])
    data = json.loads(path.read_text())
    data["primitives"][0]["weights"][0][0] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(path, backend="numpy")


def test_bundle_unknown_schema_and_unknown_operation_fail(tmp_path, fitted):
    bank, library = setup_bank(fitted)
    path = tmp_path / "bank.json"
    save_bundle(path, WorldConfig(), SeedSplit(), bank, library, ["0"*64])
    original = json.loads(path.read_text())
    data = dict(original, schema="unknown")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_bundle(path, backend="numpy")
    original["recipes"][0]["operations"] = ["unknown"]
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="unknown primitive"):
        load_bundle(path, backend="numpy")


def test_build_evaluate_are_separate_and_never_overwrite(tmp_path, monkeypatch):
    config = Path(__file__).resolve().parents[1] / "configs/embodied/reference.json"
    build_dir, eval_dir = tmp_path / "build", tmp_path / "evaluate"
    assert main(["build", "--backend", "numpy", "--config", str(config),
                 "--output", str(build_dir)]) == 0
    report = json.loads((build_dir / "report.json").read_text())
    assert report["test_evaluation"] == "NOT_EXECUTED"
    assert report["fresh_load"]["identical_replay"]
    import apc.embodied.cli as cli
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must never train")
    monkeypatch.setattr(cli, "fit_from_demonstrations", forbidden)
    monkeypatch.setattr(cli, "expert_policy", forbidden)
    assert main(["evaluate", "--backend", "numpy", "--bundle", str(build_dir / "bank.json"),
                 "--output", str(eval_dir)]) == 0
    evaluation = json.loads((eval_dir / "report.json").read_text())
    assert evaluation["training_calls"] == 0 and evaluation["weights_unchanged"]
    for family in ("planar", "spatial", "route", "detour"):
        assert evaluation["families"][family]["correct"]["success_rate"] == 1.0
    with pytest.raises(SystemExit):
        main(["build", "--output", str(build_dir)])


def test_missing_bundle_never_trains(tmp_path):
    assert main(["evaluate", "--backend", "numpy", "--bundle", str(tmp_path / "missing.json"),
                 "--output", str(tmp_path / "failed")]) == 2
    assert (tmp_path / "failed/error.json").exists()


def test_offline_viewer_escapes_script_injection(tmp_path):
    path = tmp_path / "replay.html"
    write_replay(path, WorldConfig().to_dict(), [{"label": "</script><script>alert(1)</script>"}])
    text = path.read_text()
    assert "https://" not in text and "__DATA__" not in text
    assert "</script><script>alert" not in text
    assert "\\u003c/script" in text
