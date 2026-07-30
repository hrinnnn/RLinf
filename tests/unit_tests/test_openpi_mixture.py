from rlinf.data.openpi_mixture import SourceBalancedBatchSampler


def test_source_balanced_batches_have_equal_source_counts():
    sampler = SourceBalancedBatchSampler([3, 7], batch_size=4, seed=7)
    batches = list(sampler)
    assert len(batches) == 4
    for batch in batches:
        assert len(batch) == 4
        assert sum(index < 3 for index in batch) == 2
        assert sum(index >= 3 for index in batch) == 2


def test_source_balanced_sampler_is_reproducible_per_epoch():
    sampler = SourceBalancedBatchSampler([5, 5], batch_size=4, seed=11)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)
