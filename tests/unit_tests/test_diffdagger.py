import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from rlinf.algorithms.diffdagger import (
    DiffDAggerQueryGate,
    EmpiricalCDF,
    combine_loss_masks,
    flow_matching_noisy_action_and_target,
    flow_matching_reconstruction_mse,
    intervention_keep_mask,
    load_calibration_scores,
    merge_expert_interventions,
)
from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult


def test_load_calibration_scores_json_and_jsonl(tmp_path):
    json_path = tmp_path / "scores.json"
    json_path.write_text(json.dumps({"scores": [0.1, 0.2, 0.3]}))
    np.testing.assert_allclose(load_calibration_scores(json_path), [0.1, 0.2, 0.3])

    jsonl_path = tmp_path / "scores.jsonl"
    jsonl_path.write_text('{"score": 0.4}\n{"score": 0.5}\n')
    np.testing.assert_allclose(load_calibration_scores(jsonl_path), [0.4, 0.5])


def test_load_calibration_scores_rejects_non_finite(tmp_path):
    path = tmp_path / "scores.npy"
    np.save(path, np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="finite"):
        load_calibration_scores(path)


def test_empirical_cdf_and_original_diffdagger_quantile_indexing():
    cdf = EmpiricalCDF([0.4, 0.1, 0.3, 0.2])
    assert cdf.quantile(0.5) == pytest.approx(0.3)
    np.testing.assert_allclose(
        cdf.probability(np.array([0.05, 0.3, 0.9])), [0, 0.75, 1]
    )


def test_query_gate_has_independent_patience_and_episode_reset():
    gate = DiffDAggerQueryGate(
        [0.1, 0.2, 0.3, 0.4], alpha=0.5, patience=2, patience_window=2
    )

    first = gate.decide(torch.tensor([0.35, 0.1]))
    assert first.query_mask.tolist() == [False, False]
    second = gate.decide(torch.tensor([0.36, 0.35]))
    assert second.query_mask.tolist() == [True, False]
    third = gate.decide(
        torch.tensor([0.37, 0.36]), reset_mask=torch.tensor([[True], [False]])
    )
    assert third.query_mask.tolist() == [False, True]


def test_flow_matching_target_and_per_sample_reconstruction_loss():
    actions = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    noise = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
    x_t, target = flow_matching_noisy_action_and_target(
        actions, noise, torch.tensor([0.25, 0.5])
    )
    torch.testing.assert_close(x_t[0], torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(x_t[1], torch.tensor([[5.0, 6.0]]))
    torch.testing.assert_close(target, noise - actions)

    prediction = target.clone()
    prediction[1, 0, 0] += 2.0
    losses = flow_matching_reconstruction_mse(
        prediction, target, action_chunk=1, action_dim=2
    )
    torch.testing.assert_close(losses, torch.tensor([0.0, 2.0]))


def test_merge_expert_interventions_replaces_only_queried_rows_and_labels():
    student_actions = torch.zeros(3, 2, 2)
    expert_actions = torch.full_like(student_actions, 9.0)
    student_result = {
        "prev_logprobs": torch.zeros(3, 2, 2),
        "prev_values": torch.zeros(3, 1),
        "forward_inputs": {
            "action": torch.zeros(3, 4),
            "model_action": torch.ones(3, 4),
            "chains": torch.arange(3)[:, None],
        },
    }
    expert_result = {
        "forward_inputs": {
            "action": torch.full((3, 4), 7.0),
            "model_action": torch.full((3, 4), 8.0),
        }
    }

    actions, result = merge_expert_interventions(
        student_actions,
        student_result,
        expert_actions,
        expert_result,
        torch.tensor([False, True, False]),
        num_action_chunks=2,
    )
    torch.testing.assert_close(actions[0], student_actions[0])
    torch.testing.assert_close(actions[1], expert_actions[1])
    torch.testing.assert_close(
        result["forward_inputs"]["model_action"][1], torch.full((4,), 8.0)
    )
    torch.testing.assert_close(
        result["forward_inputs"]["chains"], student_result["forward_inputs"]["chains"]
    )
    assert result["intervene_flags"].tolist() == [
        [False, False],
        [True, True],
        [False, False],
    ]


def test_interventions_are_excluded_from_chunk_level_ppo_mask():
    flags = torch.tensor(
        [
            [[False, False, False, False], [True, False, False, False]],
            [[False, False, False, False], [False, False, False, False]],
        ]
    )
    keep = intervention_keep_mask(flags, reward_type="chunk_level")
    assert keep.shape == (2, 2, 1)
    assert keep.tolist() == [[[True], [False]], [[True], [True]]]

    base = torch.tensor([[[True], [True]], [[False], [True]]])
    assert combine_loss_masks(base, keep).tolist() == [
        [[True], [False]],
        [[False], [True]],
    ]


def test_diffdagger_diagnostics_survive_trajectory_conversion():
    collector = EmbodiedRolloutResult(max_episode_length=10)
    collector.append_step_result(
        ChunkStepResult(
            actions=torch.zeros(2, 4),
            diffdagger_scores=torch.tensor([0.2, 0.7]),
            diffdagger_cdf_values=torch.tensor([0.1, 0.99]),
            diffdagger_thresholds=torch.tensor([0.6, 0.6]),
        )
    )
    trajectory = collector.to_trajectory()
    assert trajectory.diffdagger_scores.shape == (1, 2)
    torch.testing.assert_close(
        trajectory.diffdagger_scores[0], torch.tensor([0.2, 0.7])
    )
    torch.testing.assert_close(
        trajectory.diffdagger_cdf_values[0], torch.tensor([0.1, 0.99])
    )

    trajectory.intervene_flags = torch.tensor([[[False], [True]]])
    filtered = trajectory.extract_intervene_traj()
    assert len(filtered) == 1
    torch.testing.assert_close(filtered[0].diffdagger_scores, torch.tensor([[0.7]]))


@pytest.mark.parametrize(
    "config_name",
    [
        "maniskill_ppo_openpi_pi05_flow_sde",
        "maniskill_ppo_openpi_pi05_flow_sde_diffdagger",
        "maniskill_openpi_pi05_diffdagger_calibration",
    ],
)
def test_flow_sde_configs_compose_with_required_invariants(config_name, monkeypatch):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    monkeypatch.setenv("PI05_MODEL_PATH", "/tmp/pi05")
    monkeypatch.setenv("DIFFDAGGER_EXPERT_MODEL_PATH", "/tmp/expert")
    monkeypatch.setenv("DIFFDAGGER_CALIBRATION_PATH", "/tmp/scores.jsonl")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)

    assert cfg.actor.model.openpi.noise_method == "flow_sde"
    assert cfg.actor.model.openpi.joint_logprob is False
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.algorithm.entropy_bonus == 0.0
