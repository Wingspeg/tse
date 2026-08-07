"""Tests for the Rényi mutual information estimators."""

from __future__ import annotations

import math

import torch

from pef.renyi_mutual_info import (
    estimate_pairwise_renyi_mi,
    estimate_three_way_renyi_mi,
    jackknife_corrected_mi,
    preprocess,
)


def _make_correlated(n: int, dim: int, seed: int = 0):
    """Build two Gaussian blobs with controlled correlation."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g, dtype=torch.float64)
    y = 0.7 * x + 0.3 * torch.randn(n, dim, generator=g, dtype=torch.float64)
    return x, y


def _make_independent(n: int, dim: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, dim, generator=g, dtype=torch.float64),
        torch.randn(n, dim, generator=g, dtype=torch.float64),
    )


class TestPreprocess:
    def test_zero_mean_unit_std(self) -> None:
        x = torch.randn(200, 8, dtype=torch.float64) * 5.0 + 3.0
        y = preprocess(x)
        assert torch.allclose(y.mean(0), torch.zeros(8, dtype=torch.float64), atol=1e-6)
        assert torch.allclose(y.std(0), torch.ones(8, dtype=torch.float64), atol=1e-2)

    def test_projection_keeps_only_PROJ_DIM_columns(self) -> None:
        x = torch.randn(50, 64, dtype=torch.float64)
        y = preprocess(x)
        assert y.shape == (50, 16)
        assert y.dtype == torch.float64


class TestPairwiseRenyiMI:
    def test_correlated_pairs_score_higher_than_independent(self) -> None:
        x, y = _make_correlated(500, 16, seed=2)
        x2, y2 = _make_independent(500, 16, seed=3)
        mi_corr = estimate_pairwise_renyi_mi(x, y)
        mi_indep = estimate_pairwise_renyi_mi(x2, y2)
        assert mi_corr > mi_indep
        assert mi_corr > 0.05
        assert mi_indep < 0.02

    def test_returns_non_negative(self) -> None:
        x, y = _make_independent(200, 8)
        assert estimate_pairwise_renyi_mi(x, y) >= 0.0


class TestThreeWayRenyiMI:
    def test_three_way_zero_for_independent(self) -> None:
        a, b = _make_independent(200, 6, seed=4)
        c, _ = _make_independent(200, 6, seed=5)
        mi = estimate_three_way_renyi_mi(a, b, c)
        # All three independent → should be ≈ 0.
        assert mi < 0.05

    def test_three_way_positive_when_correlated(self) -> None:
        g = torch.Generator().manual_seed(6)
        a = torch.randn(300, 8, generator=g, dtype=torch.float64)
        b = 0.6 * a + 0.4 * torch.randn(300, 8, generator=g, dtype=torch.float64)
        c = 0.6 * a + 0.4 * torch.randn(300, 8, generator=g, dtype=torch.float64)
        mi = estimate_three_way_renyi_mi(a, b, c)
        assert mi > 0.05


class TestJackknife:
    def test_returns_finite_non_negative(self) -> None:
        x, y = _make_correlated(120, 8, seed=7)
        v = jackknife_corrected_mi(x, y)
        assert math.isfinite(v)
        assert v >= 0.0

    def test_three_way_jackknife(self) -> None:
        g = torch.Generator().manual_seed(8)
        a = torch.randn(120, 8, generator=g, dtype=torch.float64)
        b = 0.6 * a + 0.4 * torch.randn(120, 8, generator=g, dtype=torch.float64)
        c = 0.6 * a + 0.4 * torch.randn(120, 8, generator=g, dtype=torch.float64)
        v = jackknife_corrected_mi(a, b, c)
        assert math.isfinite(v)
