"""Shared test fixtures.

We force CPU so the test suite is fully reproducible on a laptop without
a CUDA device, and so CI doesn't accidentally gate on a GPU.
"""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _force_cpu_for_tests() -> None:
    """All tests run on CPU. The framework still works; it's just slower."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Per-test seed so order doesn't matter."""
    torch.manual_seed(42)


@pytest.fixture
def small_lattice_params():
    """Default lattice parameters.

    `compute_commitment` and `commit_*` use the module-level ``N_LAT`` /
    ``M_LAT`` constants rather than ``params.n`` / ``params.m``, so the
    fixture has to match the framework defaults (256 / 512) to avoid a
    dimension mismatch. The Sampler / SIS / Merkle code paths still
    exercise fully because the inputs are small and CPU-cheap.
    """
    from pef.lattice_utils import LatticeParams

    return LatticeParams()


@pytest.fixture
def small_repr_gen(small_lattice_params):
    """A small trusted representation generator for unit tests."""
    from pef.representation import TrustedRepresentationGenerator

    return TrustedRepresentationGenerator(small_lattice_params, prg_seed=7)
