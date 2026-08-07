"""Tests for the collaborative folding proof protocol."""

from __future__ import annotations

import torch

from pef.collaborative_folding import (
    CollaborativeFoldingProtocol,
    FoldingProofState,
    compute_commitment,
    fold,
    unfold,
    verify,
)
from pef.config import D_REPR


class TestFoldingProofState:
    def test_initial_state(self) -> None:
        s = FoldingProofState()
        assert s.num_folded == 0
        assert torch.equal(s.get_accumulated(), torch.zeros(D_REPR, dtype=torch.float64))


class TestFoldUnfold:
    def test_round_trip_preserves_accumulator(self, small_lattice_params) -> None:
        s = FoldingProofState()
        phi = torch.randn(D_REPR, dtype=torch.float64)
        com = compute_commitment(small_lattice_params, phi)
        fold(s, weight=0.5, phi=phi, commitment=com)
        assert s.num_folded == 1
        assert torch.allclose(s.get_accumulated(), 0.5 * phi)
        unfold(s, weight=0.5, phi=phi, commitment=com)
        assert s.num_folded == 0
        assert torch.allclose(s.get_accumulated(), torch.zeros(D_REPR, dtype=torch.float64))


class TestVerify:
    def test_verify_passes_after_correct_fold(self, small_lattice_params) -> None:
        s = FoldingProofState()
        phis = [torch.randn(D_REPR, dtype=torch.float64) for _ in range(3)]
        weights = [0.5, 0.3, 0.2]
        coms = [compute_commitment(small_lattice_params, p) for p in phis]
        for w, p, c in zip(weights, phis, coms, strict=False):
            fold(s, w, p, c)
        expected = sum(w * p for w, p in zip(weights, phis, strict=False))
        assert verify(small_lattice_params, s, coms, weights, expected)

    def test_verify_rejects_wrong_result(self, small_lattice_params) -> None:
        s = FoldingProofState()
        phis = [torch.randn(D_REPR, dtype=torch.float64) for _ in range(2)]
        coms = [compute_commitment(small_lattice_params, p) for p in phis]
        fold(s, 0.5, phis[0], coms[0])
        fold(s, 0.5, phis[1], coms[1])
        wrong = torch.zeros(D_REPR, dtype=torch.float64)
        assert not verify(small_lattice_params, s, coms, [0.5, 0.5], wrong)


class TestCollaborativeFoldingProtocol:
    def test_end_to_end_aggregation(self, small_lattice_params) -> None:
        proto = CollaborativeFoldingProtocol(small_lattice_params)
        phis = [torch.randn(D_REPR, dtype=torch.float64) for _ in range(4)]
        weights = [0.25] * 4
        for i, (w, p) in enumerate(zip(weights, phis, strict=False)):
            proto.publish_commitment(i, p)
            proto.fold_participant(i, w, p)
        # Aggregated vector = mean of all phis.
        expected = torch.stack(phis).mean(dim=0)
        assert torch.allclose(proto.get_global_representation(), expected, atol=1e-6)
        # Verification should pass against the protocol's own accumulator.
        assert proto.verify_aggregation()

    def test_remove_participant_rolls_back(self, small_lattice_params) -> None:
        proto = CollaborativeFoldingProtocol(small_lattice_params)
        phis = [torch.randn(D_REPR, dtype=torch.float64) for _ in range(3)]
        for i, p in enumerate(phis):
            proto.publish_commitment(i, p)
            proto.fold_participant(i, 1.0 / 3, p)
        # Snapshot after 3 folds
        before = proto.get_global_representation()
        # Roll back one participant
        proto.remove_participant(2, 1.0 / 3, phis[2])
        after = proto.get_global_representation()
        # Mean over 3 vs mean over 2 should differ.
        assert not torch.allclose(before, after)

    def test_proof_size_is_constant(self, small_lattice_params) -> None:
        """The proof is always the last 32-byte SHA-256 digest, regardless of N."""
        proto_a = CollaborativeFoldingProtocol(small_lattice_params)
        for n in (3, 10, 50):
            for i in range(n):
                phi = torch.randn(D_REPR, dtype=torch.float64)
                proto_a.publish_commitment(i, phi)
                proto_a.fold_participant(i, 1.0 / n, phi)
            # After each batch, the proof size must equal one digest.
            assert proto_a.get_proof_size() == 32 * 8
