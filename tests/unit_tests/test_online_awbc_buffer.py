from __future__ import annotations

import pytest
import torch

from rlinf.algorithms.online_awbc_buffer import (
    OnlineChunk,
    adaptive_update_steps,
    aggregate_microbatch_loss,
    persistent_flux_weight,
    quality_online_chunks,
    sample_active_batch,
    weighted_online_expert_loss,
)


def _chunk(index: int, *, weight: float = 1.0, valid: bool = True) -> OnlineChunk:
    return OnlineChunk(
        episode_id=index // 2,
        frame_index=(index % 2) * 10,
        next_frame_index=(index % 2 + 1) * 10,
        phi=0.1,
        phi_next=0.2,
        valid=valid,
        episode_length_chunks=4,
        controller="policy",
        weight=weight,
    )


def test_persistent_flux_weight_rejects_invalid_and_negative_progress():
    assert persistent_flux_weight(
        phi=0.4, phi_next=0.2, valid=True, episode_length_chunks=4, mean_episode_length_chunks=4
    ) == 0.0
    assert persistent_flux_weight(
        phi=None, phi_next=None, valid=False, episode_length_chunks=4, mean_episode_length_chunks=4
    ) == 0.0


def test_persistent_flux_weight_retains_episode_length_scale_without_batch_normalization():
    short = persistent_flux_weight(
        phi=0.0, phi_next=0.1, valid=True, episode_length_chunks=4, mean_episode_length_chunks=8
    )
    long = persistent_flux_weight(
        phi=0.0, phi_next=0.1, valid=True, episode_length_chunks=16, mean_episode_length_chunks=8
    )
    assert long == pytest.approx(4 * short)


def test_quality_buffer_filters_invalid_and_small_weight_without_touching_raw_objects():
    raw = (_chunk(0, weight=0.11), _chunk(1, weight=0.09), _chunk(2, weight=1.0, valid=False))
    selected = quality_online_chunks(raw, minimum_weight=0.1)
    assert selected == (raw[0],)
    assert len(raw) == 3


def test_k_one_active_batch_contains_one_expert_and_one_online():
    active = sample_active_batch(
        [_chunk(0)], {0: 20}, generator=torch.Generator().manual_seed(7)
    )
    assert len(active.online) == len(active.expert) == 1
    assert active.size == 2
    assert active.expert[0].next_frame_index - active.expert[0].frame_index == 10


def test_online_selection_is_without_replacement_and_expert_anchor_is_hierarchical():
    chunks = [_chunk(index) for index in range(6)]
    active = sample_active_batch(
        chunks, {0: 15, 1: 21}, generator=torch.Generator().manual_seed(11), max_online_chunks=4
    )
    assert len({(item.episode_id, item.frame_index) for item in active.online}) == 4
    assert all(anchor.episode_id in {0, 1} for anchor in active.expert)
    assert all(anchor.next_frame_index <= {0: 15, 1: 21}[anchor.episode_id] for anchor in active.expert)


def test_weighted_loss_preserves_per_online_chunk_contribution():
    expert = torch.tensor([2.0], requires_grad=True)
    online = torch.tensor([3.0, 5.0], requires_grad=True)
    loss, numerator, denominator = weighted_online_expert_loss(expert, online, torch.tensor([1.0, 0.25]))
    assert loss.item() == pytest.approx((2.0 + 3.0 + 1.25) / 2.25)
    loss.backward()
    assert online.grad[0].item() == pytest.approx(1 / 2.25)
    assert online.grad[1].item() == pytest.approx(0.25 / 2.25)
    assert numerator.item() == pytest.approx(6.25)
    assert denominator.item() == pytest.approx(2.25)


def test_microbatch_accumulation_matches_full_formula():
    full, _, _ = weighted_online_expert_loss(
        torch.tensor([2.0, 4.0]), torch.tensor([3.0, 5.0]), torch.tensor([1.0, 0.25])
    )
    _, n0, d0 = weighted_online_expert_loss(
        torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([1.0])
    )
    _, n1, d1 = weighted_online_expert_loss(
        torch.tensor([4.0]), torch.tensor([5.0]), torch.tensor([0.25])
    )
    torch.testing.assert_close(aggregate_microbatch_loss([n0, n1], [d0, d1]), full)


def test_adaptive_steps_limits_reuse_and_reaches_cap():
    assert adaptive_update_steps(online_count=1, active_online_count=1) == 4
    assert adaptive_update_steps(online_count=1000, active_online_count=32) == 50
