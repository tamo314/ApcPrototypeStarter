"""One real-simulator path: target operations, teacher data, train, reload, rollout."""
import json
import os
from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_primitive_to_saved_rollout_and_selector(tmp_path):
    from apc_maniskill.primitive_learning import MLPSelector, train
    from apc_maniskill.primitive_policy import (BaseReadyPickSelector, BaseReadyRecoverPickSelector, BaseReadyAxisRetryPickSelector,
                                                RotationProbeSelector, BaseDemoSelector, BaseRecoverPickSelector,
                                                BaseThenPickSelector, PickPlaceSelector,
                                                PitchGuardSelector, PrimitivePolicy)
    from apc_maniskill.runner import RunConfig, collect

    def run(name, episodes, steps, factory, *, seed=1400, env_id="APC-FetchPickCube-v1"):
        out = tmp_path / name
        collect(RunConfig(env_id=env_id, robot_uids="fetch", policy="external",
                          episodes=episodes, max_steps=steps, env_max_steps=steps, seed=seed),
                out, policy_factory=factory)
        assert json.loads((out / "manifest.json").read_text())["status"] == "completed"
        rows = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
        for episode in range(episodes):
            local = [row for row in rows if row["episode"] == episode]
            with np.load(out / f"episode_{episode:04d}.npz", allow_pickle=False) as data:
                assert data["actions"].shape == (len(local), 13)
                assert data["observations"].shape[0] == len(local) + 1
                for t, row in enumerate(local):
                    np.testing.assert_allclose(data["actions"][t],
                                               row["info"]["diagnostic"]["submitted_action"])
        return out, rows

    _, manual = run("manual", 1, 80, lambda env, output: PrimitivePolicy(env, output))
    first = manual[0]["info"]["diagnostic"]
    np.testing.assert_allclose(np.asarray(first["target_after_update"])
                               - first["pre_action_state"]["measured_hand_position"],
                               [0, 0, .01], atol=1e-5)
    for row in manual:
        diagnostic = row["info"]["diagnostic"]
        chosen = diagnostic["executed_id"]
        if chosen in (6, 7, 8):
            np.testing.assert_allclose(diagnostic["target_after_update"],
                                       diagnostic["pre_action_state"]["hand_target_position"])
        if chosen == 9:
            np.testing.assert_allclose(diagnostic["target_after_update"],
                                       diagnostic["pre_action_state"]["measured_hand_position"])
        if chosen == 7:
            assert diagnostic["gripper_target_after_update_m"] == -.01
        if chosen == 6:
            assert diagnostic["gripper_target_after_update_m"] == .05

    _, base = run("base", 1, 180, lambda env, output: PrimitivePolicy(
        env, output, selector=BaseDemoSelector(), allow_rotation=True, allow_base=True))
    decisions = [row["info"]["diagnostic"] for row in base]
    assert [decisions[t]["executed_id"] for t in (30, 65, 100, 135)] == [16, 18, 17, 19]
    assert decisions[30]["interruption_reason_code"] == 1
    assert decisions[170]["interruption_reason_code"] == 2
    assert all(not decisions[t]["post_action_state"]["hand_target_valid"] for t in range(30, 170))
    assert decisions[170]["post_action_state"]["hand_target_valid"]
    assert all(decisions[t]["gripper_target_after_update_m"] == -.01 for t in range(30, 160))
    assert all(decisions[t]["gripper_target_after_update_m"] == .05 for t in range(160, 180))
    assert decisions[64]["post_action_state"]["base_pose"][0] > .015
    assert decisions[169]["post_action_state"]["base_pose"][0] < .005
    assert all(np.isfinite(d["table_contact_force_norm_sum_n"]) for d in decisions)
    np.testing.assert_allclose(decisions[170]["base_target_after"],
                               decisions[170]["pre_action_state"]["base_pose"])

    _, chained = run("base_then_pick", 1, 120, lambda env, output: PrimitivePolicy(
        env, output, selector=BaseThenPickSelector(), allow_rotation=True, allow_base=True))
    chain = [row["info"]["diagnostic"] for row in chained]
    assert chain[0]["executed_id"] == 16
    assert chain[35]["interruption_reason_code"] == 2
    assert chain[35]["mode_after_code"] == 0
    assert all(row["override_reason_code"] != 3 for row in chain)
    assert all(row["table_contact_force_norm_sum_n"] == 0 for row in chain)

    _, recovered = run("base_recovery", 1, 250, lambda env, output: PrimitivePolicy(
        env, output, selector=BaseRecoverPickSelector(), allow_rotation=True, allow_base=True),
        seed=1901)
    recovery = [row["info"]["diagnostic"] for row in recovered]
    retreat = [i for i, row in enumerate(recovery) if row["executed_id"] == 17]
    assert len(retreat) == 1
    assert recovery[retreat[0]]["pre_action_state"]["last_override_reason_code"] == 2
    assert recovery[retreat[0]]["interruption_reason_code"] == 1
    assert recovery[retreat[0] + 35]["interruption_reason_code"] == 2
    assert recovery[retreat[0] + 35]["post_action_state"]["base_pose"][0] < .005
    assert all(row["table_contact_force_norm_sum_n"] == 0 for row in recovery)

    far_out, far_rows = run("far_approach", 2, 550, lambda env, output: PrimitivePolicy(
        env, output, selector=BaseReadyPickSelector(), allow_rotation=True, allow_base=True),
        seed=1901, env_id="APC-FetchPickCubeFar-v1")
    assert json.loads((far_out / "manifest.json").read_text())["task"]["start_back_m"] == .20
    for episode in range(2):
        far = [row["info"]["diagnostic"] for row in far_rows if row["episode"] == episode]
        assert sum(row["executed_id"] == 16 for row in far) == 11
        assert any(row["interruption_reason_code"] == 2 for row in far)
        assert far[-1]["post_action_state"]["base_pose"][0] > .19
        assert all(row["table_contact_force_norm_sum_n"] == 0 for row in far)

    far_train = tmp_path / "far_train"
    from apc_maniskill.primitive_learning import SCHEMA_V4
    train(far_out, far_train, updates=20, feature_schema=SCHEMA_V4)

    def far_model_factory(env, output):
        selector = MLPSelector(far_train / "selector.pt", env.unwrapped.experiment_metadata(),
                               float(env.unwrapped.sim_config.control_freq))
        return PrimitivePolicy(env, output, selector=selector, allow_rotation=True, allow_base=True)

    _, far_model = run("far_model", 1, 30, far_model_factory, seed=1901,
                       env_id="APC-FetchPickCubeFar-v1")
    assert all(len(row["info"]["diagnostic"]["selector_logits"]) == 20 for row in far_model)

    def far_query_factory(env, output):
        selector = MLPSelector(far_train / "selector.pt", env.unwrapped.experiment_metadata(),
                               float(env.unwrapped.sim_config.control_freq))
        return PrimitivePolicy(env, output, selector=selector, allow_rotation=True,
                               allow_base=True, query_teacher=True)

    _, far_query = run("far_query", 1, 30, far_query_factory, seed=1901,
                       env_id="APC-FetchPickCubeFar-v1")
    assert [row["info"]["diagnostic"]["submitted_action"] for row in far_query] == [
        row["info"]["diagnostic"]["submitted_action"] for row in far_model]
    assert all(row["info"]["diagnostic"]["teacher_label_valid"] for row in far_query)

    # New geometry features and a learned tree must survive train/save/load and
    # real hand/base execution. A diagnostic teacher must not change its actions.
    from apc_maniskill.primitive_learning import LearnedSelector, SCHEMA_V6
    tree_train = tmp_path / "tree_train"
    train(far_out, tree_train, feature_schema=SCHEMA_V6, model_kind="cart", partition_grasp=True)
    tree_training = json.loads((tree_train / "training.json").read_text())
    assert tree_training["tree_nodes"] > 1
    assert tree_training["updates"] == 0
    assert tree_training["partition_grasp"]

    def tree_factory(query):
        def factory(env, output):
            selector = LearnedSelector(tree_train / "selector.pt", env.unwrapped.experiment_metadata(),
                                       float(env.unwrapped.sim_config.control_freq))
            return PrimitivePolicy(env, output, selector=selector, allow_rotation=True,
                                   allow_base=True, query_teacher=query)
        return factory

    tree_out, tree_rows = run("tree", 1, 240, tree_factory(False), seed=1903,
                              env_id="APC-FetchPickCubeFar-v1")
    _, tree_query = run("tree_query", 1, 240, tree_factory(True), seed=1903,
                        env_id="APC-FetchPickCubeFar-v1")
    assert json.loads((tree_out / "manifest.json").read_text())["policy_details"]["selector"] == "cart_v1"
    assert all(len(row["info"]["diagnostic"]["selector_logits"]) == 20 for row in tree_rows)
    assert [row["info"]["diagnostic"]["submitted_action"] for row in tree_rows] == [
        row["info"]["diagnostic"]["submitted_action"] for row in tree_query]


    _, far_recovery = run("far_recovery", 1, 530, lambda env, output: PrimitivePolicy(
        env, output, selector=BaseReadyRecoverPickSelector(), allow_rotation=True, allow_base=True),
        seed=2802, env_id="APC-FetchPickCubeFar-v1")
    assert sum(r["info"]["diagnostic"]["proposed_id"] == 17 for r in far_recovery) == 1
    assert far_recovery[-1]["info"]["diagnostic"]["post_action_state"]["base_pose"][0] < .19

    _, axis_retry = run("axis_retry", 1, 490, lambda env, output: PrimitivePolicy(
        env, output, selector=BaseReadyAxisRetryPickSelector(), allow_rotation=True, allow_base=True),
        seed=2802, env_id="APC-FetchPickCubeFar-v1")
    assert any(r["info"]["diagnostic"]["pre_action_state"]["last_override_reason_code"] == 2
               and r["info"]["diagnostic"]["proposed_id"] in (3, 5) for r in axis_retry)

    # Probe the live first rejection; kinematic queries must leave the replay
    # unchanged. The existing axis-retry comparison differs only after it.
    probe_class = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                    / "scripts/probe_primitive_reachability.py"))["ReachabilityProbe"]
    reach_out, reach_rows = run("reachability", 1, 475, probe_class,
                                seed=2802, env_id="APC-FetchPickCubeFar-v1")
    reach = json.loads((reach_out / "reachability_000.json").read_text())
    stop = reach["step"] + 1
    assert [r["info"]["diagnostic"]["submitted_action"] for r in reach_rows[:stop]] == [
        r["info"]["diagnostic"]["submitted_action"] for r in axis_retry[:stop]]
    assert reach["candidates"] and all(np.isfinite(c["residual"]) for c in reach["candidates"])
    assert all(c["success"] and c["within_limits"] and c["path_clearance_m"] >= .002
               for c in reach["candidates"] if c["feasible"])

    _, forward_rows = run("forward_switch", 1, 240, lambda env, output: PrimitivePolicy(
        env, output, selector=BaseReadyPickSelector(base_switch_x_m=.215),
        allow_rotation=True, allow_base=True), seed=2802, env_id="APC-FetchPickCubeFar-v1")
    assert any(r["info"]["diagnostic"]["interruption_reason_code"] == 2
               and r["info"]["diagnostic"]["pre_action_state"]["base_pose"][0] >= .215
               for r in forward_rows)

    _, reseed_rows = run("ik_reseed", 1, 675, lambda env, output: PrimitivePolicy(
        env, output, selector=BaseReadyPickSelector(desired_pitch_deg=0, base_switch_x_m=.215),
        allow_rotation=True, allow_base=True, ik_reset_seed=True),
        seed=2803, env_id="APC-FetchPickCubeFar-v1")
    recovered = [r["info"]["diagnostic"] for r in reseed_rows
                 if r["info"]["diagnostic"].get("ik_reset_seed_used")]
    assert recovered
    assert all(d["ik_success"] and d["ik_within_limits"] and d["ik_table_clear"]
               and d["ik_reset_seed_attempts"] == 5 and d["override_reason_code"] == 0
               for d in recovered)
    assert all(d["ik_position_error_m"] < .001 for d in recovered)

    adapt_out, adapt_rows = run("translation_backoff_pitch_schedule", 1, 800,
        lambda env, output: PrimitivePolicy(env, output,
            selector=BaseReadyPickSelector(desired_pitch_deg=0, descend_pitch_deg=5,
                                          base_switch_x_m=.215, grasp_height_m=.02),
            allow_rotation=True, allow_base=True, translation_backoff=True),
        seed=2802, env_id="APC-FetchPickCubeFar-v1")
    adapted = [r["info"]["diagnostic"] for r in adapt_rows]
    recovered = [d for d in adapted if d["translation_backoff_used"]]
    assert recovered
    for d in recovered:
        initial = d["translation_backoff_initial_solver"]
        assert not (initial["ik_success"] and initial["ik_within_limits"] and initial["ik_table_clear"])
        assert d["override_reason_code"] == 0 and d["ik_success"] and d["ik_table_clear"]
        hand = np.asarray(d["pre_action_state"]["measured_hand_position"])
        np.testing.assert_allclose(np.asarray(d["translation_backoff_target"]) - hand,
                                   .5 * (np.asarray(d["attempted_target_position"]) - hand), atol=1e-7)
        np.testing.assert_allclose(d["target_after_update"], d["translation_backoff_target"])
    phases = [d["pitch_schedule_descending"] for d in adapted]
    assert any(phases) and not phases[0] and phases == sorted(phases)
    assert all(d["pitch_schedule_desired_deg"] == (5 if d["pitch_schedule_descending"] else 0)
               for d in adapted)
    assert json.loads((adapt_out / "manifest.json").read_text())["policy_details"]["grasp_height_m"] == .02

    stable_out, stable_rows = run("stable_pre_rotate_recover", 1, 700,
        lambda env, output: PrimitivePolicy(env, output,
            selector=BaseReadyPickSelector(desired_pitch_deg=10, descend_pitch_deg=10,
                                          base_switch_x_m=.215, grasp_height_m=.02,
                                          recover_grasp=True, pre_rotate=True),
            allow_rotation=True, allow_base=True, translation_backoff=True),
        seed=2801, env_id="APC-FetchPickCubeFar-v1")
    stable = [r["info"]["diagnostic"] for r in stable_rows]
    assert any(d["post_action_state"]["grasped"] for d in stable)
    assert all(d["table_contact_force_norm_sum_n"] == 0 for d in stable)
    assert json.loads((stable_out / "manifest.json").read_text())["policy_details"]["recover_grasp"]
    assert json.loads((stable_out / "manifest.json").read_text())["policy_details"]["pre_rotate"]

    def probe_factory(env, output):
        learned_selector = LearnedSelector(tree_train / "selector.pt", env.unwrapped.experiment_metadata(),
                                          float(env.unwrapped.sim_config.control_freq))
        return PrimitivePolicy(env, output, selector=RotationProbeSelector(learned_selector, {20: 12}),
                               allow_rotation=True, allow_base=True, query_teacher=True)

    probe_out, probe = run("rotation_probe", 1, 30, probe_factory,
                           seed=1903, env_id="APC-FetchPickCubeFar-v1")
    assert not json.loads((probe_out / "manifest.json").read_text())["policy_details"]["learned"]
    assert probe[20]["info"]["diagnostic"]["probe_applied"]
    assert probe[20]["info"]["diagnostic"]["proposed_id"] == 12
    assert all(r["info"]["diagnostic"]["teacher_label_valid"] for r in probe)

    teacher, _ = run("teacher", 2, 80, lambda env, output: PrimitivePolicy(
        env, output, selector=PickPlaceSelector(use_rotation=True), allow_rotation=True))
    learned = tmp_path / "train"
    train(teacher, learned, updates=20)
    training = json.loads((learned / "training.json").read_text())
    assert training["training_samples"] == training["validation_samples"] == 80
    assert training["model_parameters"] > 0

    def learned_factory(env, output):
        selector = MLPSelector(learned / "selector.pt", env.unwrapped.experiment_metadata(),
                               float(env.unwrapped.sim_config.control_freq))
        return PrimitivePolicy(env, output, selector=selector, allow_rotation=True,
                               query_teacher=True)

    _, learner = run("learner", 1, 30, learned_factory)
    assert all(len(row["info"]["diagnostic"]["selector_logits"]) == 16 for row in learner)
    assert all(row["info"]["diagnostic"]["teacher_label_valid"] for row in learner)
    assert all(not row["info"]["diagnostic"]["teacher_ik_feasibility_checked"] for row in learner)

    def guard_factory(env, output):
        base_selector = MLPSelector(learned / "selector.pt", env.unwrapped.experiment_metadata(),
                                    float(env.unwrapped.sim_config.control_freq))
        return PrimitivePolicy(env, output, selector=PitchGuardSelector(base_selector),
                               allow_rotation=True)

    guard_out, guard_rows = run("pitch_guard", 1, 30, guard_factory)
    guard = [row["info"]["diagnostic"] for row in guard_rows]
    assert json.loads((guard_out / "manifest.json").read_text())["policy_details"]["hybrid"]
    assert guard[0]["pitch_guard_triggered"]
    assert guard[0]["proposed_id"] == 13
    assert all(0 <= row["raw_model_id"] < 16 for row in guard)
