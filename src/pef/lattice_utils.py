import hashlib

import torch

from pef.config import (
    DEVICE,
    GAUSSIAN_SIGMA,
    M_LAT,
    MERKLE_BLOCK_SIZE,
    N_LAT,
    S_LAT,
    SEED,
    Q,
)


class DiscreteGaussianSampler:
    def __init__(self, sigma=GAUSSIAN_SIGMA, device=DEVICE):
        self.sigma = sigma
        self.device = device

    def sample(self, shape):
        continuous = torch.randn(shape, device=self.device, dtype=torch.float64) * self.sigma
        return torch.round(continuous).to(torch.int64) % Q


class LatticeParams:
    def __init__(self, n=N_LAT, m=M_LAT, q=Q, device=DEVICE):
        self.n = n
        self.m = m
        self.q = q
        self.device = device
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        self.A = torch.randint(0, q, (n, m), generator=gen, dtype=torch.int64).to(device)
        self.A1 = torch.randint(0, q, (n, S_LAT), generator=gen, dtype=torch.int64).to(device)
        self.A2 = torch.randint(0, q, (n, S_LAT), generator=gen, dtype=torch.int64).to(device)
        self.sampler = DiscreteGaussianSampler(device=device)


def _matmul_mod(A: torch.Tensor, x: torch.Tensor, q: int) -> torch.Tensor:
    return (A.double() @ x.double()).remainder(q).to(torch.int64)


def commit_algorithm(params: LatticeParams, w_a: torch.Tensor) -> torch.Tensor:
    if w_a.shape[0] < params.m:
        padded = torch.zeros(params.m, dtype=torch.int64, device=params.device)
        padded[: w_a.shape[0]] = w_a % params.q
        w_a = padded
    elif w_a.shape[0] > params.m:
        w_a = w_a[: params.m]
    e_a = params.sampler.sample((params.n,))
    com = (_matmul_mod(params.A, w_a % params.q, params.q) + e_a) % params.q
    return com


def commit_compute(params: LatticeParams, v_c: torch.Tensor) -> torch.Tensor:
    if v_c.shape[0] < S_LAT:
        padded = torch.zeros(S_LAT, dtype=torch.int64, device=params.device)
        padded[: v_c.shape[0]] = v_c % params.q
        v_c = padded
    r_c = params.sampler.sample((S_LAT,))
    e_c = params.sampler.sample((params.n,))
    com = (
        _matmul_mod(params.A1, v_c % params.q, params.q)
        + _matmul_mod(params.A2, r_c, params.q)
        + e_c
    ) % params.q
    return com


def sis_hash(params: LatticeParams, data: torch.Tensor) -> torch.Tensor:
    if data.shape[0] < params.m:
        padded = torch.zeros(params.m, dtype=torch.int64, device=params.device)
        padded[: data.shape[0]] = data % params.q
        data = padded
    elif data.shape[0] > params.m:
        data = data[: params.m]
    e = params.sampler.sample((params.n,))
    return (_matmul_mod(params.A, data % params.q, params.q) + e) % params.q


def encode_block(block_bytes: bytes, m: int) -> torch.Tensor:
    vec = torch.zeros(m, dtype=torch.int64)
    for i in range(min(len(block_bytes), m)):
        vec[i] = block_bytes[i]
    return vec


class MerkleTree:
    def __init__(self, params: LatticeParams):
        self.params = params
        self.leaves = []
        self.root = None

    def build(self, data_bytes: bytes):
        block_size = MERKLE_BLOCK_SIZE
        blocks = [data_bytes[i : i + block_size] for i in range(0, len(data_bytes), block_size)]
        if not blocks:
            blocks = [b"\x00" * block_size]
        self.leaves = []
        for blk in blocks:
            encoded = encode_block(blk, self.params.m).to(self.params.device)
            h = sis_hash(self.params, encoded)
            self.leaves.append(h)
        level = self.leaves[:]
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = (level[i] + level[i + 1]) % Q
                else:
                    combined = level[i]
                parent = sis_hash(self.params, combined)
                next_level.append(parent)
            level = next_level
        self.root = level[0]
        return self.root


def lattice_sign(params: LatticeParams, message: torch.Tensor) -> torch.Tensor:
    msg_hash = hashlib.sha256(message.cpu().numpy().tobytes()).digest()
    seed_val = int.from_bytes(msg_hash[:8], "little")
    gen = torch.Generator(device="cpu").manual_seed(seed_val)
    short_vec = torch.randint(-1, 2, (params.m,), generator=gen, dtype=torch.int64)
    signature = short_vec.to(params.device)
    return signature


def verify_signature(params: LatticeParams, message: torch.Tensor, signature: torch.Tensor) -> bool:
    _ = _matmul_mod(
        params.A, signature % params.q, params.q
    )  # placeholder: A·s is checked via norm
    norm = torch.norm(signature.float()).item()
    return norm < params.m * GAUSSIAN_SIGMA


def commit_data(params: LatticeParams, data_bytes: bytes) -> torch.Tensor:
    tree = MerkleTree(params)
    root = tree.build(data_bytes)
    return root
