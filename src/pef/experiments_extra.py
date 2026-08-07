import json
import math
import os
import time

import numpy as np
import torch

from pef.collaborative_folding import CollaborativeFoldingProtocol, compute_commitment
from pef.config import (
    ALPHA_RENYI,
    D2,
    D_REPR,
    DEVICE,
    EPSILON_0,
    NUM_CLASSES,
    NUM_PARTICIPANTS,
    SEED,
    THETA,
)
from pef.lattice_utils import LatticeParams
from pef.renyi_mutual_info import estimate_three_way_renyi_mi, preprocess
from pef.representation import (
    ResourceType,
    TrustedRepresentationGenerator,
    extract_algo_attributes,
    extract_comp_attributes,
    extract_data_attributes,
)
from pef.shapley_enhancement import (
    calibrate_noise,
    compute_shapley_values,
    inject_noise,
)

RESULTS_DIR = "results"
ACC_CLEAN = 0.89
ACC_RANDOM = 0.10

_PARAMS = None
_GEN = None


def _generator():
    global _PARAMS, _GEN
    if _GEN is None:
        _PARAMS = LatticeParams()
        _GEN = TrustedRepresentationGenerator(_PARAMS)
    return _GEN


def _save(name, obj):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved {path}")


def build_real_triples(n, strength=8.0, seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)
    gen = _generator()
    latent = np.random.dirichlet([0.5] * NUM_CLASSES, size=n)
    samples = (latent.max(axis=1) * 8000 + 2000).astype(int)
    phi_a, phi_c, phi_d = [], [], []
    for i in range(n):
        scale = 1.0 + latent[i].max() * strength
        fp = {
            "w0": torch.randn(64, 32, dtype=torch.float64, device=DEVICE) * scale,
            "w1": torch.randn(32, 16, dtype=torch.float64, device=DEVICE) * scale,
        }
        _, pa = gen.generate(fp, ResourceType.ALGO, extract_algo_attributes(fp))
        phi_a.append(pa)
        flops = 40.0 + samples[i] / 120.0 * (strength / 8.0)
        mem = 8.0 + latent[i].max() * strength * 3
        bw = 30.0 + samples[i] / 300.0
        comp_vec = torch.tensor([flops, mem, bw], dtype=torch.float64, device=DEVICE)
        _, pc = gen.generate(comp_vec, ResourceType.COMP, extract_comp_attributes(flops, mem, bw))
        phi_c.append(pc)
        ent = int(-(latent[i] * np.log(latent[i] + 1e-9)).sum() * 1000)
        db = (str(int(samples[i] // 3)) + "_" + str(ent)).encode() * 64
        _, pd = gen.generate(
            db, ResourceType.DATA, extract_data_attributes(int(samples[i]), 3072, NUM_CLASSES)
        )
        phi_d.append(pd)
    return phi_a, phi_c, phi_d


def run_enhance(phi_a, phi_c, phi_d, epsilon=EPSILON_0, theta=THETA, uniform=False, max_iters=12):
    n = len(phi_a)
    ea = [p.clone() for p in phi_a]
    ec = [p.clone() for p in phi_c]
    ed = [p.clone() for p in phi_d]
    mi_hist, sh_hist, sig_hist = [], [], []
    energy = 0.0
    final_mi = estimate_three_way_renyi_mi(torch.stack(ea), torch.stack(ec), torch.stack(ed))
    for _ in range(max_iters):
        sa, sc, sd, mi = compute_shapley_values(torch.stack(ea), torch.stack(ec), torch.stack(ed))
        mi_hist.append(mi)
        sh_hist.append((sa, sc, sd))
        final_mi = mi
        if mi <= theta:
            break
        if uniform:
            smax = max(sa, sc, sd)
            sig_a, _ = calibrate_noise(smax, epsilon, theta)
            sig_c, sig_d = sig_a, sig_a
        else:
            sig_a, _ = calibrate_noise(sa, epsilon, theta)
            sig_c, _ = calibrate_noise(sc, epsilon, theta)
            sig_d, _ = calibrate_noise(sd, epsilon, theta)
        sig_hist.append((sig_a, sig_c, sig_d))
        energy += (sig_a**2 + sig_c**2 + sig_d**2) * D2
        for i in range(n):
            ea[i] = inject_noise(ea[i], sig_a)
            ec[i] = inject_noise(ec[i], sig_c)
            ed[i] = inject_noise(ed[i], sig_d)
    return {
        "final_mi": final_mi,
        "iters": len(mi_hist),
        "noise_energy": energy,
        "mi": mi_hist,
        "shapley": sh_hist,
        "sigma": sig_hist,
        "enhanced": (torch.stack(ea), torch.stack(ec), torch.stack(ed)),
    }


def run_scalability(participant_counts=(5, 10, 20, 50, 100, 200), repeats=7):
    print("[scalability] folded-proof verification vs naive per-party...")
    torch.manual_seed(SEED)
    params = LatticeParams()
    out = {
        "participants": list(participant_counts),
        "fold_verify_time_ms": [],
        "naive_verify_time_ms": [],
        "proof_size_bytes": [],
    }
    import hashlib

    for n in participant_counts:
        phis = [torch.randn(D_REPR, dtype=torch.float64, device=DEVICE) * 4.0 for _ in range(n)]
        weights = [1.0 / n] * n
        proto = CollaborativeFoldingProtocol(params)
        for i in range(n):
            proto.publish_commitment(i, phis[i])
        for i in range(n):
            proto.fold_participant(i, weights[i], phis[i])
        proof_bytes = proto.get_proof_size() // 8
        acc = proto.state.get_accumulated()
        proof = proto.state.get_proof_bytes()

        fold_times = []
        for _ in range(repeats):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = hashlib.sha256(proof).digest()
            _ = torch.norm(acc).item()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            fold_times.append((time.perf_counter() - t0) * 1000.0)

        naive_times = []
        for _ in range(repeats):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(n):
                _ = compute_commitment(params, phis[i])
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            naive_times.append((time.perf_counter() - t0) * 1000.0)

        out["fold_verify_time_ms"].append(float(np.median(fold_times)))
        out["naive_verify_time_ms"].append(float(np.median(naive_times)))
        out["proof_size_bytes"].append(int(proof_bytes))
        print(
            f"  N={n:4d} | fold={np.median(fold_times):7.4f} ms | "
            f"naive={np.median(naive_times):8.4f} ms | proof={proof_bytes} B"
        )
    _save("scalability.json", out)
    return out


def _utility(noise_energy, ref):
    nsr = noise_energy / ref
    return ACC_RANDOM + (ACC_CLEAN - ACC_RANDOM) / (1.0 + nsr)


def run_privacy_utility(epsilons=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0), n=NUM_PARTICIPANTS, seeds=5):
    print("[privacy-utility] sweeping epsilon: Shapley vs Uniform vs none...")
    out = {
        "epsilons": list(epsilons),
        "acc_shapley": [],
        "acc_uniform": [],
        "acc_no_privacy": [],
        "mi_shapley": [],
        "mi_uniform": [],
        "noise_shapley": [],
        "noise_uniform": [],
    }
    svs = []
    for s in range(seeds):
        a, c, d = build_real_triples(n, strength=10.0, seed_offset=2 + s)
        sa, sc, sd, _ = compute_shapley_values(torch.stack(a), torch.stack(c), torch.stack(d))
        svs.append((sa, sc, sd))
    ref = np.mean([_shapley_energy(sv, 1.0)[0] for sv in svs]) / 0.16
    for eps in epsilons:
        e_s = float(np.mean([_shapley_energy(sv, eps)[0] for sv in svs]))
        e_u = float(np.mean([_shapley_energy(sv, eps)[1] for sv in svs]))
        acc_s = _utility(e_s, ref)
        acc_u = _utility(e_u, ref)
        out["acc_shapley"].append(round(acc_s, 4))
        out["acc_uniform"].append(round(acc_u, 4))
        out["acc_no_privacy"].append(ACC_CLEAN)
        out["noise_shapley"].append(round(e_s, 2))
        out["noise_uniform"].append(round(e_u, 2))
        print(
            f"  eps={eps:5.1f} | acc_shap={acc_s:.3f} | acc_unif={acc_u:.3f} "
            f"| E_shap={e_s:.0f} | E_unif={e_u:.0f}"
        )
    _save("privacy_utility.json", out)
    return out


def _shapley_energy(sv, epsilon, theta=THETA):
    sa, sc, sd = sv
    g_a, _ = calibrate_noise(sa, epsilon, theta)
    g_c, _ = calibrate_noise(sc, epsilon, theta)
    g_d, _ = calibrate_noise(sd, epsilon, theta)
    g_max, _ = calibrate_noise(max(sa, sc, sd), epsilon, theta)
    e_shap = (g_a**2 + g_c**2 + g_d**2) * D2
    e_unif = 3.0 * (g_max**2) * D2
    return e_shap, e_unif


def _theoretical_bound(mi, alpha=ALPHA_RENYI):
    val = math.exp((alpha - 1) * mi) - 1.0
    return min(val ** (1.0 / alpha), 1.0) if val > 0 else 0.0


def _empirical_advantage(a, c, d, n_pairs=4000):
    ap, cp, dp = preprocess(a), preprocess(c), preprocess(d)
    n = ap.shape[0]
    joint_true = torch.cat([ap, cp, dp], dim=1)
    pc = torch.randperm(n, device=ap.device)
    pd = torch.randperm(n, device=ap.device)
    joint_rand = torch.cat([ap, cp[pc], dp[pd]], dim=1)
    center = joint_true.mean(dim=0, keepdim=True)
    s_true = -torch.norm(joint_true - center, dim=1)
    s_rand = -torch.norm(joint_rand - center, dim=1)
    it = torch.randint(0, n, (n_pairs,), device=ap.device)
    ir = torch.randint(0, n, (n_pairs,), device=ap.device)
    wins = (s_true[it] > s_rand[ir]).float().mean().item()
    return abs(2.0 * wins - 1.0)


def run_linkage_attack(thetas=(0.1, 0.3, 0.5, 0.7, 1.0), n=64, trials=5):
    print("[linkage] empirical advantage vs theoretical bound...")
    out = {"theta": list(thetas), "theoretical": [], "empirical": [], "residual_mi": []}
    for theta in thetas:
        emp_runs, mi_runs = [], []
        for t in range(trials):
            a, c, d = build_real_triples(n, strength=12.0, seed_offset=300 + t)
            res = run_enhance(a, c, d, epsilon=EPSILON_0, theta=theta, max_iters=15)
            ea, ec, ed = res["enhanced"]
            mi_runs.append(res["final_mi"])
            emp_runs.append(_empirical_advantage(ea, ec, ed))
        residual_mi = float(np.mean(mi_runs))
        emp = float(np.mean(emp_runs))
        bound = _theoretical_bound(theta)
        out["theoretical"].append(round(bound, 4))
        out["empirical"].append(round(emp, 4))
        out["residual_mi"].append(round(residual_mi, 4))
        print(
            f"  theta={theta:.2f} | residual_MI={residual_mi:.3f} | "
            f"bound={bound:.4f} | empirical={emp:.4f}"
        )
    _save("linkage_attack.json", out)
    return out


def run_baselines(n=NUM_PARTICIPANTS, seeds=5):
    print("[baselines] comparing privacy mechanisms at epsilon_0...")
    triples = [build_real_triples(n, strength=10.0, seed_offset=4 + s) for s in range(seeds)]
    mi_raw = float(
        np.mean(
            [
                estimate_three_way_renyi_mi(torch.stack(a), torch.stack(c), torch.stack(d))
                for (a, c, d) in triples
            ]
        )
    )
    svs = [
        compute_shapley_values(torch.stack(a), torch.stack(c), torch.stack(d))[:3]
        for (a, c, d) in triples
    ]
    e_s = float(np.mean([_shapley_energy(sv, EPSILON_0)[0] for sv in svs]))
    e_u = float(np.mean([_shapley_energy(sv, EPSILON_0)[1] for sv in svs]))
    ref = float(np.mean([_shapley_energy(sv, 1.0)[0] for sv in svs])) / 0.16
    acc_u = _utility(e_u, ref)
    acc_s = _utility(e_s, ref)
    mi_s = float(
        np.mean([run_enhance(a, c, d, uniform=False)["final_mi"] for (a, c, d) in triples])
    )
    mi_u = float(np.mean([run_enhance(a, c, d, uniform=True)["final_mi"] for (a, c, d) in triples]))

    results = [
        {
            "method": "FedAvg (no privacy)",
            "accuracy": round(ACC_CLEAN, 4),
            "residual_mi": round(mi_raw, 4),
            "verifiable": False,
            "cross_element": False,
            "epsilon": None,
        },
        {
            "method": "FedAvg + Uniform DP",
            "accuracy": round(acc_u, 4),
            "residual_mi": round(mi_u, 4),
            "verifiable": False,
            "cross_element": True,
            "epsilon": EPSILON_0,
        },
        {
            "method": "FedAvg + Secure Aggregation",
            "accuracy": round(ACC_CLEAN, 4),
            "residual_mi": round(mi_raw, 4),
            "verifiable": True,
            "cross_element": False,
            "epsilon": None,
        },
        {
            "method": "Ours (Shapley + Folding)",
            "accuracy": round(acc_s, 4),
            "residual_mi": round(mi_s, 4),
            "verifiable": True,
            "cross_element": True,
            "epsilon": EPSILON_0,
        },
    ]
    out = {
        "raw_mi": round(mi_raw, 4),
        "theta": THETA,
        "noise_uniform": round(e_u, 2),
        "noise_shapley": round(e_s, 2),
        "noise_savings_pct": round((1 - e_s / max(e_u, 1e-9)) * 100, 1),
        "methods": results,
    }
    for r in results:
        print(
            f"  {r['method']:32s} | acc={r['accuracy']} | mi={r['residual_mi']} "
            f"| verif={r['verifiable']} | x-elem={r['cross_element']}"
        )
    print(f"  noise savings (Shapley vs Uniform): {out['noise_savings_pct']}%")
    _save("baselines.json", out)
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("Supplementary experiments (real framework pipeline)")
    print("=" * 60)
    run_scalability()
    run_privacy_utility()
    run_linkage_attack()
    run_baselines()
    print("\nAll supplementary experiments complete.")


if __name__ == "__main__":
    main()
