"""One real-simulator path: target operations, teacher data, train, reload, rollout."""
import json
import os

import numpy as np
import pytest


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_primitive_to_saved_rollout_and_selector(tmp_path):
    from apc_maniskill.primitive_learning import MLPSelector, train
    from apc_maniskill.primitive_policy import BaseDemoSelector, PickPlaceSelector, PrimitivePolicy
    from apc_maniskill.runner import RunConfig, collect

    def run(name, episodes, steps, factory):
        out = tmp_path / name
        collect(RunConfig(env_id="APC-FetchPickCube-v1", robot_uids="fetch", policy="external",
                          episodes=episodes, max_steps=steps, env_max_steps=steps, seed=1400),
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
