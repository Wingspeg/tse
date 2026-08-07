import hashlib
import math
import struct
import time

import torch

from pef.config import D1, D2, DEVICE, KAPPA, N_LAT, SEED, Q
from pef.lattice_utils import (
    DiscreteGaussianSampler,
    LatticeParams,
    commit_algorithm,
    commit_compute,
    commit_data,
    lattice_sign,
)


class ResourceType:
    ALGO = "algo"
    COMP = "comp"
    DATA = "data"


def apply_identity_coupling(
    phi_a_list, phi_c_list, phi_d_list, betas=(0.85, 0.97, 0.55), seed_offset=0
):
    n = len(phi_a_list)
    gen = torch.Generator(device="cpu").manual_seed(SEED + 4096 + seed_offset)
    latent = torch.randn(n, D2, generator=gen, dtype=torch.float64).to(DEVICE)
    ba, bc, bd = betas

    def couple(phi_list, beta):
        out = []
        for i, phi in enumerate(phi_list):
            v = phi[D1:]
            mu, sd = v.mean(), v.std().clamp(min=1.0)
            vs = (v - mu) / sd
            mixed = beta * latent[i] + math.sqrt(max(1.0 - beta * beta, 0.0)) * vs
            new_v = mixed * sd + mu
            out.append(torch.cat([phi[:D1], new_v]))
        return out

    return couple(phi_a_list, ba), couple(phi_c_list, bc), couple(phi_d_list, bd)


def reed_solomon_encode(data_vec: torch.Tensor, output_dim: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(SEED + 1000)
    G = torch.randint(0, Q, (output_dim, data_vec.shape[0]), generator=gen, dtype=torch.int64)
    G = G.to(DEVICE)
    encoded = (G.double() @ data_vec.double().to(DEVICE)).remainder(Q).to(torch.int64)
    return encoded


def generate_id(resource_fingerprint: bytes, nonce: int = None) -> torch.Tensor:
    if nonce is None:
        nonce = int(time.time() * 1e6) % (2**32)
    t = struct.pack("<d", time.time())
    nonce_bytes = struct.pack("<I", nonce)
    raw = hashlib.sha256(t + resource_fingerprint + nonce_bytes).digest()
    id_vec = torch.zeros(KAPPA, dtype=torch.int64, device=DEVICE)
    for i in range(min(KAPPA, len(raw) * 8)):
        byte_idx = i // 8
        bit_idx = i % 8
        id_vec[i] = (raw[byte_idx] >> bit_idx) & 1
    return id_vec


def extract_algo_attributes(model_params: dict) -> torch.Tensor:
    total_params = sum(p.numel() for p in model_params.values())
    num_layers = len(model_params)
    max_dim = max(max(p.shape) for p in model_params.values())
    attr = torch.tensor(
        [total_params % Q, num_layers % Q, max_dim % Q, 1, 0], dtype=torch.int64, device=DEVICE
    )
    return attr


def extract_comp_attributes(
    gpu_flops: float, memory_gb: float, bandwidth_gbps: float
) -> torch.Tensor:
    attr = torch.tensor(
        [int(gpu_flops * 100) % Q, int(memory_gb * 100) % Q, int(bandwidth_gbps * 100) % Q, 1, 1],
        dtype=torch.int64,
        device=DEVICE,
    )
    return attr


def extract_data_attributes(num_samples: int, feature_dim: int, num_classes: int) -> torch.Tensor:
    attr = torch.tensor(
        [num_samples % Q, feature_dim % Q, num_classes % Q, 2, 0], dtype=torch.int64, device=DEVICE
    )
    return attr


class TrustedRepresentationGenerator:
    def __init__(self, params: LatticeParams = None, prg_seed: int = SEED):
        if params is None:
            params = LatticeParams()
        self.params = params
        self.prg_seed = prg_seed
        gen = torch.Generator(device="cpu").manual_seed(prg_seed + 2000)
        self.B_K = torch.randint(0, Q, (D2, N_LAT), generator=gen, dtype=torch.int64).to(DEVICE)
        self.sampler = DiscreteGaussianSampler(device=DEVICE)

    def generate(
        self, resource, resource_type: str, attributes: torch.Tensor, fingerprint: bytes = None
    ):
        if fingerprint is None:
            fingerprint = hashlib.sha256(str(resource).encode()).digest()
        id_r = generate_id(fingerprint)
        if resource_type == ResourceType.ALGO:
            com_r = self._commit_algo(resource)
        elif resource_type == ResourceType.COMP:
            com_r = self._commit_comp(resource)
        else:
            com_r = self._commit_data(resource)
        msg = torch.cat([id_r, attributes.to(DEVICE), com_r])
        sigma_r = lattice_sign(self.params, msg)
        u_r = self._stage1_encode(id_r, attributes)
        v_r = self._stage2_embed(com_r)
        phi_r = torch.cat([u_r.float(), v_r.float()]).to(torch.float64)
        descriptor = {"id": id_r, "attr": attributes, "com": com_r, "sigma": sigma_r}
        return descriptor, phi_r

    def _commit_algo(self, model_params: dict):
        flat = []
        for p in model_params.values():
            flat.append(p.detach().flatten())
        w_a = torch.cat(flat).to(torch.int64).to(DEVICE) % Q
        return commit_algorithm(self.params, w_a)

    def _commit_comp(self, comp_vec: torch.Tensor):
        return commit_compute(self.params, comp_vec.to(torch.int64).to(DEVICE))

    def _commit_data(self, data_bytes: bytes):
        return commit_data(self.params, data_bytes)

    def _stage1_encode(self, id_r: torch.Tensor, attr_r: torch.Tensor):
        combined = torch.cat([id_r, attr_r.to(DEVICE)])
        u_r = reed_solomon_encode(combined, D1)
        return u_r

    def _stage2_embed(self, com_r: torch.Tensor):
        if com_r.shape[0] < N_LAT:
            padded = torch.zeros(N_LAT, dtype=torch.int64, device=DEVICE)
            padded[: com_r.shape[0]] = com_r
            com_r = padded
        e_prime = self.sampler.sample((D2,))
        v_r = (self.B_K.double() @ com_r[:N_LAT].double()).remainder(Q).to(torch.int64)
        v_r = (v_r + e_prime) % Q
        return v_r
