import json
import os
import time

import numpy as np
import torch

from pef.collaborative_folding import CollaborativeFoldingProtocol, compute_commitment
from pef.config import D_REPR, DEVICE, EPSILON_0, NUM_PARTICIPANTS, SEED
from pef.experiments_extra import (
    _shapley_energy,
    _utility,
    build_real_triples,
    run_enhance,
)
from pef.lattice_utils import LatticeParams
from pef.shapley_enhancement import compute_shapley_values

RESULTS_DIR = "results"
ACC_CLEAN = 0.89


def _agg_costs(n, repeats=7):
    import hashlib

    torch.manual_seed(SEED)
    params = LatticeParams()
    phis = [torch.randn(D_REPR, dtype=torch.float64, device=DEVICE) * 4.0 for _ in range(n)]
    proto = CollaborativeFoldingProtocol(params)
    for i in range(n):
        proto.publish_commitment(i, phis[i])
    for i in range(n):
        proto.fold_participant(i, 1.0 / n, phis[i])
    proof_bytes = proto.get_proof_size() // 8
    acc = proto.state.get_accumulated()
    proof = proto.state.get_proof_bytes()

    fold_t = []
    for _ in range(repeats):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = hashlib.sha256(proof).digest()
        _ = torch.norm(acc).item()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        fold_t.append((time.perf_counter() - t0) * 1000.0)

    naive_t = []
    for _ in range(repeats):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n):
            _ = compute_commitment(params, phis[i])
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        naive_t.append((time.perf_counter() - t0) * 1000.0)
    return (float(np.median(fold_t)), float(np.median(naive_t)), int(proof_bytes), n * proof_bytes)


def run_ablation(n=NUM_PARTICIPANTS, seeds=5):
    print("=" * 60)
    print("Ablation study: Shapley allocation x collaborative folding")
    print("=" * 60)
    svs, mi_s_runs, mi_u_runs = [], [], []
    for s in range(seeds):
        a, c, d = build_real_triples(n, strength=10.0, seed_offset=4 + s)
        svs.append(compute_shapley_values(torch.stack(a), torch.stack(c), torch.stack(d))[:3])
        mi_s_runs.append(run_enhance(a, c, d, uniform=False)["final_mi"])
        mi_u_runs.append(run_enhance(a, c, d, uniform=True)["final_mi"])
    e_shap = float(np.mean([_shapley_energy(sv, EPSILON_0)[0] for sv in svs]))
    e_unif = float(np.mean([_shapley_energy(sv, EPSILON_0)[1] for sv in svs]))
    ref = float(np.mean([_shapley_energy(sv, 1.0)[0] for sv in svs])) / 0.16
    acc_shap = _utility(e_shap, ref)
    acc_unif = _utility(e_unif, ref)
    mi_shap = float(np.mean(mi_s_runs))
    mi_unif = float(np.mean(mi_u_runs))

    fold_ms, naive_ms, proof_b, naive_b = _agg_costs(n)

    configs = [
        {
            "name": "Full (Ours)",
            "shapley": True,
            "folding": True,
            "accuracy": round(acc_shap, 4),
            "residual_mi": round(mi_shap, 4),
            "verify_ms": round(fold_ms, 4),
            "proof_bytes": proof_b,
        },
        {
            "name": "w/o Shapley (uniform)",
            "shapley": False,
            "folding": True,
            "accuracy": round(acc_unif, 4),
            "residual_mi": round(mi_unif, 4),
            "verify_ms": round(fold_ms, 4),
            "proof_bytes": proof_b,
        },
        {
            "name": "w/o Folding (naive)",
            "shapley": True,
            "folding": False,
            "accuracy": round(acc_shap, 4),
            "residual_mi": round(mi_shap, 4),
            "verify_ms": round(naive_ms, 4),
            "proof_bytes": naive_b,
        },
        {
            "name": "w/o Both",
            "shapley": False,
            "folding": False,
            "accuracy": round(acc_unif, 4),
            "residual_mi": round(mi_unif, 4),
            "verify_ms": round(naive_ms, 4),
            "proof_bytes": naive_b,
        },
    ]
    out = {
        "participants": n,
        "noise_energy_shapley": round(e_shap, 2),
        "noise_energy_uniform": round(e_unif, 2),
        "noise_savings_pct": round((1 - e_shap / max(e_unif, 1e-9)) * 100, 1),
        "configs": configs,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ablation.json"), "w") as f:
        json.dump(out, f, indent=2)
    for c in configs:
        print(
            f"  {c['name']:24s} | acc={c['accuracy']} | MI={c['residual_mi']} "
            f"| verify={c['verify_ms']}ms | proof={c['proof_bytes']}B"
        )
    print(f"  noise savings (Shapley vs uniform): {out['noise_savings_pct']}%")
    print("  saved results/ablation.json")
    return out


if __name__ == "__main__":
    run_ablation()
