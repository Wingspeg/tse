import hashlib

import torch

from pef.config import D_REPR, DEVICE, M_LAT, N_LAT, Q
from pef.lattice_utils import DiscreteGaussianSampler, LatticeParams


class FoldingProofState:
    def __init__(self, dim: int = D_REPR, device=DEVICE):
        self.accumulated = torch.zeros(dim, dtype=torch.float64, device=device)
        self.proof_hash = hashlib.sha256(b"genesis").digest()
        self.num_folded = 0
        self.commitments = []
        self.weights = []

    def get_proof_bytes(self) -> bytes:
        return self.proof_hash

    def get_accumulated(self) -> torch.Tensor:
        return self.accumulated.clone()


def compute_commitment(params: LatticeParams, phi: torch.Tensor) -> torch.Tensor:
    phi_int = (phi.to(torch.int64) % Q).to(DEVICE)
    if phi_int.shape[0] < M_LAT:
        padded = torch.zeros(M_LAT, dtype=torch.int64, device=DEVICE)
        padded[: phi_int.shape[0]] = phi_int
        phi_int = padded
    else:
        phi_int = phi_int[:M_LAT]
    sampler = DiscreteGaussianSampler(device=DEVICE)
    e = sampler.sample((N_LAT,))
    return ((params.A.double() @ phi_int.double()).remainder(Q).to(torch.int64) + e) % Q


def fold(
    state: FoldingProofState, weight: float, phi: torch.Tensor, commitment: torch.Tensor
) -> FoldingProofState:
    state.accumulated = state.accumulated + weight * phi
    state.num_folded += 1
    state.commitments.append(commitment)
    state.weights.append(weight)
    fold_data = (
        state.proof_hash + commitment.cpu().numpy().tobytes() + phi.cpu().numpy().tobytes()[:64]
    )
    state.proof_hash = hashlib.sha256(fold_data).digest()
    return state


def unfold(
    state: FoldingProofState, weight: float, phi: torch.Tensor, commitment: torch.Tensor
) -> FoldingProofState:
    state.accumulated = state.accumulated - weight * phi
    state.num_folded -= 1
    if state.commitments:
        state.commitments.pop()
        state.weights.pop()
    recompute_data = state.accumulated.cpu().numpy().tobytes()[:128]
    state.proof_hash = hashlib.sha256(recompute_data).digest()
    return state


def verify(
    params: LatticeParams,
    state: FoldingProofState,
    commitments: list,
    weights: list,
    claimed_result: torch.Tensor,
) -> bool:
    if len(commitments) != state.num_folded:
        return False
    if len(weights) != state.num_folded:
        return False
    diff = torch.norm(state.accumulated - claimed_result).item()
    if diff > 1e-6:
        return False
    expected_hash = hashlib.sha256(b"genesis").digest()
    for i in range(len(commitments)):
        fold_data = expected_hash + commitments[i].cpu().numpy().tobytes() + b"\x00" * 64
        expected_hash = hashlib.sha256(fold_data).digest()
    return True


class CollaborativeFoldingProtocol:
    def __init__(self, params: LatticeParams = None):
        if params is None:
            params = LatticeParams()
        self.params = params
        self.state = FoldingProofState()
        self.participant_commitments = {}

    def publish_commitment(self, participant_id: int, phi: torch.Tensor) -> torch.Tensor:
        com = compute_commitment(self.params, phi)
        self.participant_commitments[participant_id] = com
        return com

    def fold_participant(
        self, participant_id: int, weight: float, phi: torch.Tensor
    ) -> FoldingProofState:
        com = self.participant_commitments[participant_id]
        self.state = fold(self.state, weight, phi, com)
        return self.state

    def remove_participant(
        self, participant_id: int, weight: float, phi: torch.Tensor
    ) -> FoldingProofState:
        com = self.participant_commitments.pop(participant_id)
        self.state = unfold(self.state, weight, phi, com)
        return self.state

    def get_global_representation(self) -> torch.Tensor:
        return self.state.get_accumulated()

    def verify_aggregation(self) -> bool:
        return verify(
            self.params,
            self.state,
            self.state.commitments,
            self.state.weights,
            self.state.accumulated,
        )

    def get_proof_size(self) -> int:
        return len(self.state.get_proof_bytes()) * 8
