"""PDAL variants (paper eqs. 1, 3-5) on tiny batches whose values can be checked by hand."""
import math

import pytest
import torch

from hurricane_intensity.pdal import pairwise_distances, pdal, pdal_absolute, pdal_loss, pdal_scale_invariant


def test_two_samples():
    # embeddings 1 apart, labels 2 apart: the two off-diagonal pairs each differ by 1; m^2 = 4
    F = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    C = torch.tensor([[0.0], [2.0]])
    assert pdal(F, C).item() == pytest.approx(2 * 1 ** 2 / 4)       # 0.5
    assert pdal_absolute(F, C).item() == pytest.approx(2 * 1 / 4)   # 0.5
    # dF = (0,1,1,0), dC = (0,2,2,0): parallel, cos = 1
    assert pdal_scale_invariant(F, C).item() == pytest.approx(math.exp(-1))


def test_three_samples():
    # F on a line at 0, 3, 4 (distances 3, 4, 1); C at 0, 1, 3 (distances 1, 3, 2)
    F = torch.tensor([[0.0], [3.0], [4.0]])
    C = torch.tensor([[0.0], [1.0], [3.0]])
    diffs = [3 - 1, 4 - 3, 1 - 2]  # pairs (0,1), (0,2), (1,2), each counted twice
    assert pdal(F, C).item() == pytest.approx(2 * sum(d * d for d in diffs) / 9)   # 12/9
    assert pdal_absolute(F, C).item() == pytest.approx(2 * sum(abs(d) for d in diffs) / 9)  # 8/9
    dot = 2 * (3 * 1 + 4 * 3 + 1 * 2)
    cos = dot / (math.sqrt(2 * (9 + 16 + 1)) * math.sqrt(2 * (1 + 9 + 4)))
    assert pdal_scale_invariant(F, C).item() == pytest.approx(math.exp(-cos))


def test_scale_invariance_and_perfect_alignment():
    F = torch.randn(6, 4)
    C = torch.randn(6, 1)
    assert pdal_scale_invariant(5.0 * F, C).item() == pytest.approx(pdal_scale_invariant(F, C).item(), rel=1e-5)
    assert pdal(C, C).item() == pytest.approx(0.0, abs=1e-6)
    assert pdal_scale_invariant(3.0 * C, C).item() == pytest.approx(math.exp(-1), rel=1e-5)


def test_distances_and_finite_gradient_with_duplicates():
    F = torch.tensor([[0.6, 0.8], [0.6, 0.8], [1.0, 0.0]], requires_grad=True)  # two identical embeddings
    C = torch.tensor([[0.0], [0.5], [1.0]])
    d = pairwise_distances(F)
    assert torch.all(d.diagonal() == 0) and d[0, 1].item() == pytest.approx(0.0, abs=1e-5)
    assert d[0, 2].item() == pytest.approx(math.hypot(0.4, 0.8))
    for variant in ("plain", "absolute", "scale_invariant"):
        F.grad = None
        pdal_loss(F, C, variant).backward()
        assert torch.isfinite(F.grad).all(), variant


def test_unknown_variant():
    with pytest.raises(ValueError, match="unknown PDAL variant"):
        pdal_loss(torch.zeros(2, 2), torch.zeros(2, 1), "triplet")
