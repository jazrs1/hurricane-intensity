"""Storm-capped batch sampler, storm-deduplicated kNN and category metrics (no networks, no data)."""
from collections import Counter

import numpy as np
import pytest
import torch

from hurricane_intensity.analog import StormCappedBatchSampler, category, knn_predict, metrics_by_category


def test_sampler_caps_storms_and_covers_everything():
    storms = np.repeat([f"S{i}" for i in range(40)], 10)  # 40 storms x 10 windows
    s = StormCappedBatchSampler(storms, batch_size=32, max_per_storm=4, generator=torch.Generator().manual_seed(0))
    batches = list(s)
    seen = [i for b in batches for i in b]
    assert sorted(seen) == list(range(400)) and s.dropped == 0
    for b in batches:
        assert len(b) <= 32 and max(Counter(storms[b]).values()) <= 4
    assert all(len(b) == 32 for b in batches[:-1])


def test_sampler_drops_tiny_leftovers():
    storms = np.array(["A"] * 20 + ["B"] * 2)  # A can contribute at most 4 per batch
    s = StormCappedBatchSampler(storms, batch_size=8, max_per_storm=4, min_batch=4,
                                generator=torch.Generator().manual_seed(1))
    batches = list(s)
    assert all(Counter(storms[b])["A"] <= 4 for b in batches)
    assert sum(len(b) for b in batches) + s.dropped == 22


def test_knn_one_window_per_storm_and_idw():
    db = np.array([[0.0], [0.1], [1.0], [3.0]])
    labels = np.array([100.0, 999.0, 50.0, 10.0])
    storms = np.array(["A", "A", "B", "C"])  # the window at 0.1 is storm A again and must be skipped
    pred, d = knn_predict(np.array([[0.0]]), db, labels, storms, k=2, eps=0.0 + 1e-9)
    # neighbors: A at 0.0 (w = 1e9) and B at 1.0 (w = 1): essentially A's label
    assert pred[0] == pytest.approx(100.0, abs=1e-5) and d[0].tolist() == pytest.approx([0.0, 1.0])
    pred, _ = knn_predict(np.array([[2.0]]), db, labels, storms, k=2)
    # B at 1.0 and C at 1.0 tie (w = 1 each, A is 1.9 away): mean of 50 and 10
    assert pred[0] == pytest.approx(30.0, rel=1e-5)
    with pytest.raises(ValueError, match="only 3 storms"):
        knn_predict(np.array([[0.0]]), db, labels, storms, k=4)


def test_categories_and_metrics():
    assert category([33, 34, 63, 64, 82, 83, 95, 96, 112, 113, 136, 137]).tolist() == \
        ["TD", "TS", "TS", "C1", "C1", "C2", "C2", "C3", "C3", "C4", "C4", "C5"]
    m = metrics_by_category([110, 130, 150], [100, 120, 140])
    assert m["all"]["bias"] == 10 and m[">=113"]["n"] == 2 and m["C5"]["mae"] == 10 and m["TD"]["n"] == 0
