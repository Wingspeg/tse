import math

import torch

from pef.config import (
    ALPHA_RENYI,
    CONVERGENCE_TOL,
    D1,
    D2,
    DELTA,
    DEVICE,
    EPSILON_0,
    LAMBDA_PARAM,
    MAX_ENHANCEMENT_ITERS,
    NOISE_K,
    THETA,
)
from pef.renyi_mutual_info import estimate_pairwise_renyi_mi, estimate_three_way_renyi_mi


def compute_shapley_values(
    phi_a: torch.Tensor, phi_c: torch.Tensor, phi_d: torch.Tensor, alpha: float = ALPHA_RENYI
) -> tuple:
    I_ac = estimate_pairwise_renyi_mi(phi_a, phi_c, alpha)
    I_ad = estimate_pairwise_renyi_mi(phi_a, phi_d, alpha)
    I_cd = estimate_pairwise_renyi_mi(phi_c, phi_d, alpha)
    I_a_cd = estimate_three_way_renyi_mi(phi_a, phi_c, phi_d, alpha)
    I_c_ad = I_a_cd
    I_d_ac = I_a_cd
    sv_a = (2 * I_a_cd + I_ac + I_ad - I_cd) / 6.0
    sv_c = (2 * I_c_ad + I_ac + I_cd - I_ad) / 6.0
    sv_d = (2 * I_d_ac + I_ad + I_cd - I_ac) / 6.0
    sv_a = max(sv_a, 1e-8)
    sv_c = max(sv_c, 1e-8)
    sv_d = max(sv_d, 1e-8)
    return sv_a, sv_c, sv_d, I_a_cd


def calibrate_noise(
    shapley_val: float,
    epsilon_0: float = EPSILON_0,
    theta: float = THETA,
    lam: float = LAMBDA_PARAM,
    alpha: float = ALPHA_RENYI,
    delta: float = DELTA,
    sensitivity: float = 1.0,
) -> float:
    epsilon_r = epsilon_0 * theta / (theta + lam * shapley_val)
    epsilon_r = max(epsilon_r, 1e-6)
    sigma_r = math.sqrt(2 * alpha * math.log(1.0 / delta)) / epsilon_r * sensitivity
    return sigma_r, epsilon_r


def inject_noise(phi: torch.Tensor, sigma: float) -> torch.Tensor:
    enhanced = phi.clone()
    std_v = enhanced[D1:].std().clamp(min=1.0)
    frac = sigma / (sigma + NOISE_K)
    noise = torch.randn(D2, device=DEVICE, dtype=torch.float64) * frac * std_v
    enhanced[D1:] = enhanced[D1:] + noise
    return enhanced


def adaptive_security_enhancement(
    phi_a_list: list,
    phi_c_list: list,
    phi_d_list: list,
    theta: float = THETA,
    epsilon_0: float = EPSILON_0,
    alpha: float = ALPHA_RENYI,
) -> dict:
    results = {
        "enhanced_a": [],
        "enhanced_c": [],
        "enhanced_d": [],
        "shapley_history": [],
        "mi_history": [],
        "sigma_history": [],
        "epsilon_history": [],
    }
    n_a = len(phi_a_list)
    n_c = len(phi_c_list)
    n_d = len(phi_d_list)
    enhanced_a = [p.clone() for p in phi_a_list]
    enhanced_c = [p.clone() for p in phi_c_list]
    enhanced_d = [p.clone() for p in phi_d_list]
    for _iteration in range(MAX_ENHANCEMENT_ITERS):
        all_a = torch.stack(enhanced_a)
        all_c = torch.stack(enhanced_c)
        all_d = torch.stack(enhanced_d)
        min_n = min(n_a, n_c, n_d)
        sv_a, sv_c, sv_d, mi_val = compute_shapley_values(
            all_a[:min_n], all_c[:min_n], all_d[:min_n], alpha
        )
        results["shapley_history"].append((sv_a, sv_c, sv_d))
        results["mi_history"].append(mi_val)
        if mi_val <= theta + CONVERGENCE_TOL:
            break
        sigma_a, eps_a = calibrate_noise(sv_a, epsilon_0, theta, LAMBDA_PARAM, alpha, DELTA)
        sigma_c, eps_c = calibrate_noise(sv_c, epsilon_0, theta, LAMBDA_PARAM, alpha, DELTA)
        sigma_d, eps_d = calibrate_noise(sv_d, epsilon_0, theta, LAMBDA_PARAM, alpha, DELTA)
        results["sigma_history"].append((sigma_a, sigma_c, sigma_d))
        results["epsilon_history"].append((eps_a, eps_c, eps_d))
        for i in range(n_a):
            enhanced_a[i] = inject_noise(enhanced_a[i], sigma_a)
        for i in range(n_c):
            enhanced_c[i] = inject_noise(enhanced_c[i], sigma_c)
        for i in range(n_d):
            enhanced_d[i] = inject_noise(enhanced_d[i], sigma_d)
    results["enhanced_a"] = enhanced_a
    results["enhanced_c"] = enhanced_c
    results["enhanced_d"] = enhanced_d
    return results
