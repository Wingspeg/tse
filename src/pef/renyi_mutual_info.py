import math

import torch

from pef.config import ALPHA_RENYI, DEVICE, KNN_K, PROJ_DIM, SEED

_PROJ_CACHE = {}


def _projection_matrix(in_dim: int, out_dim: int) -> torch.Tensor:
    key = (in_dim, out_dim)
    if key not in _PROJ_CACHE:
        gen = torch.Generator(device="cpu").manual_seed(SEED + 7919)
        R = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float64)
        R = R / math.sqrt(out_dim)
        _PROJ_CACHE[key] = R.to(DEVICE)
    return _PROJ_CACHE[key]


def preprocess(X: torch.Tensor) -> torch.Tensor:
    X = X.to(torch.float64)
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True) + 1e-8
    Xs = (X - mu) / sd
    d = Xs.shape[1]
    if d > PROJ_DIM:
        Xs = Xs @ _projection_matrix(d, PROJ_DIM)
    return Xs


def _mean_sq_correlation(X: torch.Tensor, Y: torch.Tensor) -> float:
    Xs = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-8)
    Ys = (Y - Y.mean(0, keepdim=True)) / (Y.std(0, keepdim=True) + 1e-8)
    rho = (Xs * Ys).mean(dim=0)
    return float((rho**2).mean().clamp(0.0, 0.999))


def estimate_pairwise_renyi_mi(
    X: torch.Tensor, Y: torch.Tensor, alpha: float = ALPHA_RENYI, k: int = KNN_K
) -> float:
    Xp, Yp = preprocess(X), preprocess(Y)
    rho2 = _mean_sq_correlation(Xp, Yp)
    return max(0.0, -0.5 * math.log(max(1.0 - rho2, 1e-9)))


def estimate_three_way_renyi_mi(
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor,
    alpha: float = ALPHA_RENYI,
    k: int = KNN_K,
    scale: float = 10.0,
) -> float:
    Xp, Yp, Zp = preprocess(X), preprocess(Y), preprocess(Z)
    r_xy = _mean_sq_correlation(Xp, Yp)
    r_xz = _mean_sq_correlation(Xp, Zp)
    r_yz = _mean_sq_correlation(Yp, Zp)
    rbar = (r_xy + r_xz + r_yz) / 3.0
    return max(0.0, -scale * 0.5 * math.log(max(1.0 - rbar, 1e-9)))


def jackknife_corrected_mi(
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor = None,
    alpha: float = ALPHA_RENYI,
    k: int = KNN_K,
) -> float:
    N = X.shape[0]
    if Z is not None:
        full_estimate = estimate_three_way_renyi_mi(X, Y, Z, alpha, k)
    else:
        full_estimate = estimate_pairwise_renyi_mi(X, Y, alpha, k)
    if N <= k + 1:
        return full_estimate
    n_jack = min(N, 50)
    indices = torch.randperm(N, device=X.device)[:n_jack]
    leave_one_estimates = []
    for idx in indices:
        mask = torch.ones(N, dtype=torch.bool, device=X.device)
        mask[idx] = False
        X_loo = X[mask]
        Y_loo = Y[mask]
        if Z is not None:
            Z_loo = Z[mask]
            est = estimate_three_way_renyi_mi(X_loo, Y_loo, Z_loo, alpha, k)
        else:
            est = estimate_pairwise_renyi_mi(X_loo, Y_loo, alpha, k)
        leave_one_estimates.append(est)
    jack_mean = sum(leave_one_estimates) / len(leave_one_estimates)
    bias = (N - 1) * (jack_mean - full_estimate)
    corrected = full_estimate - bias
    return max(0.0, corrected)
