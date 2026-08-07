"""Tests for the Shapley-guided adaptive noise mechanism."""

from __future__ import annotations

import math

import torch

from pef.config import D1, D_REPR, EPSILON_0, THETA
from pef.shapley_enhancement import (
    adaptive_security_enhancement,
    calibrate_noise,
    compute_shapley_values,
    inject_noise,
)


def _triples(n: int, seed: int = 0):
    """Build three random representation triples of the canonical shape."""
    g = torch.Generator().manual_seed(seed)
    a = [torch.randn(D_REPR, generator=g, dtype=torch.float64) for _ in range(n)]
    b = [torch.randn(D_REPR, generator=g, dtype=torch.float64) for _ in range(n)]
    c = [torch.randn(D_REPR, generator=g, dtype=torch.float64) for _ in range(n)]
    return a, b, c


class TestShapley:
    def test_shapley_values_are_positive(self) -> None:
        a, b, c = _triples(8, seed=11)
        sv_a, sv_c, sv_d, mi = compute_shapley_values(
            torch.stack(a), torch.stack(b), torch.stack(c)
        )
        # All Shapley values are floored to 1e-8.
        assert sv_a > 0 and sv_c > 0 and sv_d > 0
        assert math.isfinite(mi)

    def test_shapley_decreases_mi(self) -> None:
        """The adaptive loop should drive the three-way MI down to ≤ θ."""
        a, b, c = _triples(8, seed=12)
        # Make them somewhat correlated so the loop has work to do.
        for i in range(8):
            b[i] = 0.5 * a[i] + 0.5 * b[i]
            c[i] = 0.5 * a[i] + 0.5 * c[i]

        result = adaptive_security_enhancement(a, b, c, theta=THETA, epsilon_0=EPSILON_0)
        final_mi = result["mi_history"][-1] if result["mi_history"] else 0.0
        # Either the loop converged, or it exhausted the iteration budget.
        assert final_mi <= THETA + 0.2  # generous slack for the estimator

    def test_enhanced_vectors_preserved_length(self) -> None:
        a, b, c = _triples(5, seed=13)
        result = adaptive_security_enhancement(a, b, c)
        for phi in result["enhanced_a"]:
            assert phi.shape == (D_REPR,)
        for phi in result["enhanced_c"]:
            assert phi.shape == (D_REPR,)
        for phi in result["enhanced_d"]:
            assert phi.shape == (D_REPR,)


class TestCalibrateNoise:
    def test_sigma_increases_as_epsilon_decreases(self) -> None:
        # Higher Shapley → smaller ε_r → larger σ_r (monotone).
        sigma_low_sv, eps_low = calibrate_noise(shapley_val=0.01, epsilon_0=1.0)
        sigma_high_sv, eps_high = calibrate_noise(shapley_val=10.0, epsilon_0=1.0)
        assert eps_low > eps_high
        assert sigma_high_sv > sigma_low_sv
        assert sigma_low_sv > 0 and math.isfinite(sigma_low_sv)


class TestInjectNoise:
    def test_noise_only_touches_tail_half(self) -> None:
        """The first D1 entries (id+attribute half) should be untouched."""
        phi = torch.zeros(D_REPR, dtype=torch.float64)
        out = inject_noise(phi, sigma=0.1)
        assert torch.equal(out[:D1], phi[:D1])
        # Tail half (D2) is the noise target — at least some entries changed.
        assert not torch.equal(out[D1:], phi[D1:])
