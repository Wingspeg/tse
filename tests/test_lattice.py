"""Tests for the lattice primitives (sampling, SIS hash, Merkle tree, sign)."""

from __future__ import annotations

import torch

from pef.config import MERKLE_BLOCK_SIZE, Q
from pef.lattice_utils import (
    DiscreteGaussianSampler,
    MerkleTree,
    commit_algorithm,
    commit_compute,
    commit_data,
    encode_block,
    lattice_sign,
    sis_hash,
    verify_signature,
)


class TestDiscreteGaussianSampler:
    def test_shape_and_dtype(self) -> None:
        s = DiscreteGaussianSampler(sigma=2.0, device="cpu")
        out = s.sample((128,))
        assert out.shape == (128,)
        assert out.dtype == torch.int64
        # Values are reduced mod Q, so they live in [0, Q).
        assert (out >= 0).all()
        assert out.max().item() < 12289

    def test_distribution_first_and_second_moment(self) -> None:
        """With σ small relative to Q, the mod-q wrap-around is rare and
        the empirical mean / std should match the continuous Gaussian."""
        s = DiscreteGaussianSampler(sigma=1.0, device="cpu")
        # σ=1, Q=12289 → wraparound probability is negligible for 10k samples.
        samples = s.sample((10_000,)).float()
        # Map the mod-Q representation back to its centred integer representative
        # in (-Q/2, Q/2] for the moment check.
        centred = torch.where(samples > 12289 / 2, samples - 12289, samples)
        assert abs(centred.mean().item()) < 0.1
        assert abs(centred.std().item() - 1.0) < 0.1


class TestLatticeParams:
    def test_matrices_initialised(self, small_lattice_params) -> None:
        p = small_lattice_params
        assert p.A.shape == (p.n, p.m)
        assert p.A1.shape == (p.n, 64)
        assert p.A2.shape == (p.n, 64)
        assert (p.A >= 0).all() and (p.A < p.q).all()


class TestCommit:
    def test_commit_algorithm_shape(self, small_lattice_params) -> None:
        w = torch.randint(0, Q, (small_lattice_params.m,), dtype=torch.int64)
        c = commit_algorithm(small_lattice_params, w)
        assert c.shape == (small_lattice_params.n,)
        assert (c >= 0).all() and (c < small_lattice_params.q).all()

    def test_commit_compute_uses_split_matrices(self, small_lattice_params) -> None:
        from pef.config import S_LAT

        v = torch.randint(0, Q, (S_LAT,), dtype=torch.int64)
        c = commit_compute(small_lattice_params, v)
        assert c.shape == (small_lattice_params.n,)

    def test_commit_data_runs_merkle(self, small_lattice_params) -> None:
        payload = b"hello world" * 100  # >1 Merkle block
        c = commit_data(small_lattice_params, payload)
        assert c.shape == (small_lattice_params.n,)


class TestMerkleTree:
    def test_root_shape_and_modulus(self, small_lattice_params) -> None:
        tree = MerkleTree(small_lattice_params)
        root = tree.build(b"hello world" * 50)
        assert root.shape == (small_lattice_params.n,)
        assert (root >= 0).all() and (root < small_lattice_params.q).all()

    def test_build_handles_empty_payload(self, small_lattice_params) -> None:
        tree = MerkleTree(small_lattice_params)
        root = tree.build(b"")
        # Empty payload gets a single zero block; root must still be well-formed.
        assert root.shape == (small_lattice_params.n,)

    def test_leaves_count_matches_blocks(self, small_lattice_params) -> None:
        tree = MerkleTree(small_lattice_params)
        payload = b"x" * (MERKLE_BLOCK_SIZE * 3 + 17)
        tree.build(payload)
        # 4 full blocks (3 + a partial that is still one block).
        assert len(tree.leaves) == 4


class TestEncodeBlock:
    def test_round_trip_prefix(self) -> None:
        data = bytes(range(10))
        v = encode_block(data, m=64)
        assert v[:10].tolist() == list(range(10))
        # Padded with zeros
        assert v[10:].tolist() == [0] * 54


class TestSignature:
    def test_sign_then_norm_check_passes(self, small_lattice_params) -> None:
        msg = torch.randint(0, Q, (small_lattice_params.m,), dtype=torch.int64)
        sig = lattice_sign(small_lattice_params, msg)
        assert sig.shape == (small_lattice_params.m,)
        # The verification here only checks norm; it should pass for any
        # short {-1, 0, 1} vector.
        assert verify_signature(small_lattice_params, msg, sig)

    def test_signature_is_deterministic(self, small_lattice_params) -> None:
        msg = torch.randint(0, Q, (small_lattice_params.m,), dtype=torch.int64)
        s1 = lattice_sign(small_lattice_params, msg)
        s2 = lattice_sign(small_lattice_params, msg)
        assert torch.equal(s1, s2)


class TestSISHash:
    def test_hash_shape_and_modulus(self, small_lattice_params) -> None:
        data = torch.randint(0, Q, (small_lattice_params.m,), dtype=torch.int64)
        h = sis_hash(small_lattice_params, data)
        assert h.shape == (small_lattice_params.n,)
        assert (h >= 0).all() and (h < small_lattice_params.q).all()
