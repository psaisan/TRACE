from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from .plotting import plot_fig_ab, plot_fig_cd, plot_fig_ef, plot_fig_gh_threat
from .theory import (
    bayes_auc_from_delta2,
    delta2_from_gaussian,
    gaussian_params_for_X,
    gaussian_params_for_Zhat,
    mi_from_delta2_mc,
)
from .utils import learned_auc_logreg, make_orthonormal_basis, mean_channel_corr


def sample_world(
    rng: np.random.Generator,
    d: int = 1000,
    m: int = 12,
    p: int = 8,
    r: int = 200,
    sigma_x: float = 1.0,
    sigma_z: float = 0.7,
    lam: float = 6.0,
    biology_noise: float = 1.0,
) -> dict[str, Any]:
    b0 = rng.normal(size=m)
    b0 /= np.linalg.norm(b0) + 1e-12
    A = rng.normal(size=(d, m)) / np.sqrt(m)
    T = rng.normal(size=(p, m)) / np.sqrt(m)
    Q = make_orthonormal_basis(d, r, rng)
    return {
        "d": d,
        "m": m,
        "p": p,
        "r": r,
        "sigma_x": sigma_x,
        "sigma_z": sigma_z,
        "lam": lam,
        "biology_noise": biology_noise,
        "b0": b0,
        "A": A,
        "T": T,
        "Q": Q,
    }


def sample_XZ_from_world(params: dict[str, Any], rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, m, p, r = params["d"], params["m"], params["p"], params["r"]
    sigma_x, sigma_z, lam = params["sigma_x"], params["sigma_z"], params["lam"]
    bn = params["biology_noise"]
    b0, A, T, Q = params["b0"], params["A"], params["T"], params["Q"]

    Y = rng.integers(0, 2, size=n)
    s = 2 * Y - 1
    xi = rng.normal(size=(n, m)) * bn
    B = s[:, None] * b0[None, :] + xi
    Z_true = B @ T.T + rng.normal(size=(n, p)) * sigma_z
    ncoef = rng.normal(size=(n, r))
    eps = rng.normal(size=(n, d)) * sigma_x
    X = B @ A.T + lam * (ncoef @ Q.T) + eps
    return X, Z_true, Y


def fit_translator_ridge(X_pair: np.ndarray, Z_pair: np.ndarray, alpha: float = 10.0) -> tuple[Ridge, Any]:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_pair, Z_pair)

    def h(X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    return model, h


def compute_core_panel_data(
    seed: int = 7,
    d: int = 1000,
    m: int = 12,
    p: int = 8,
    r: int = 200,
    sigma_x: float = 1.0,
    sigma_z: float = 0.7,
    biology_noise: float = 1.0,
    lam_hard: float = 6.0,
    n_pair_grid: tuple[int, ...] = (200, 500, 1000, 3000, 8000),
    n_label_grid: tuple[int, ...] = (50, 100, 200, 400, 800, 1600),
    lam_grid: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0),
    corrupt_std_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
    phase_n_grid: tuple[int, ...] = (50, 100, 200, 400, 800),
    phase_lam_grid: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0),
    ridge_alpha: float = 10.0,
    C_X: float = 0.03,
    C_Z: float = 1.0,
    repeats: int = 6,
    mi_mc_n: int = 40000,
    n_test: int = 12000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base_world = sample_world(
        rng,
        d=d,
        m=m,
        p=p,
        r=r,
        sigma_x=sigma_x,
        sigma_z=sigma_z,
        lam=lam_hard,
        biology_noise=biology_noise,
    )
    Xte, _, yte = sample_XZ_from_world(base_world, rng, n=n_test)

    muX, SigmaX = gaussian_params_for_X(base_world)
    delta2_X = delta2_from_gaussian(muX, SigmaX)
    miX = mi_from_delta2_mc(delta2_X, n_mc=mi_mc_n, seed=int(rng.integers(1, 10**9)))
    aucX_star = bayes_auc_from_delta2(delta2_X)

    miX_list, miZ_list = [], []
    aucX_star_list, aucZ_star_list = [], []
    for n_pair in n_pair_grid:
        Xp, Zp_true, _ = sample_XZ_from_world(base_world, rng, n=n_pair)
        model_h, _ = fit_translator_ridge(Xp, Zp_true, alpha=ridge_alpha)
        muZ, SigmaZ = gaussian_params_for_Zhat(muX, SigmaX, model_h)
        delta2_Z = delta2_from_gaussian(muZ, SigmaZ)
        miZ = mi_from_delta2_mc(delta2_Z, n_mc=mi_mc_n, seed=int(rng.integers(1, 10**9)))
        aucZ_star = bayes_auc_from_delta2(delta2_Z)
        miX_list.append(miX)
        miZ_list.append(miZ)
        aucX_star_list.append(aucX_star)
        aucZ_star_list.append(aucZ_star)

    miX_bits = np.array(miX_list) / np.log(2.0)
    miZ_bits = np.array(miZ_list) / np.log(2.0)

    n_pair_fixed = int(n_pair_grid[len(n_pair_grid) // 2])
    Xp, Zp_true, _ = sample_XZ_from_world(base_world, rng, n=n_pair_fixed)
    model_h_fixed, h_fixed = fit_translator_ridge(Xp, Zp_true, alpha=ridge_alpha)
    Zte_hat = h_fixed(Xte)

    aucX_n_mean, aucX_n_std, aucZ_n_mean, aucZ_n_std = [], [], [], []
    for n_lab in n_label_grid:
        aucX_rep, aucZ_rep = [], []
        for _ in range(repeats):
            Xtr, _, ytr = sample_XZ_from_world(base_world, rng, n=n_lab)
            Ztr_hat = h_fixed(Xtr)
            aucX_rep.append(learned_auc_logreg(Xtr, ytr, Xte, yte, C=C_X))
            aucZ_rep.append(learned_auc_logreg(Ztr_hat, ytr, Zte_hat, yte, C=C_Z))
        aucX_n_mean.append(np.mean(aucX_rep))
        aucX_n_std.append(np.std(aucX_rep))
        aucZ_n_mean.append(np.mean(aucZ_rep))
        aucZ_n_std.append(np.std(aucZ_rep))

    n_label_fixed = int(n_label_grid[2]) if len(n_label_grid) >= 3 else int(n_label_grid[0])
    aucX_lam_mean, aucX_lam_std, aucZ_lam_mean, aucZ_lam_std = [], [], [], []
    for lam in lam_grid:
        world_lam = dict(base_world)
        world_lam["lam"] = float(lam)
        Xte2, _, yte2 = sample_XZ_from_world(world_lam, rng, n=n_test)
        Zte_hat2 = h_fixed(Xte2)
        aucX_rep, aucZ_rep = [], []
        for _ in range(repeats):
            Xtr2, _, ytr2 = sample_XZ_from_world(world_lam, rng, n=n_label_fixed)
            Ztr_hat2 = h_fixed(Xtr2)
            aucX_rep.append(learned_auc_logreg(Xtr2, ytr2, Xte2, yte2, C=C_X))
            aucZ_rep.append(learned_auc_logreg(Ztr_hat2, ytr2, Zte_hat2, yte2, C=C_Z))
        aucX_lam_mean.append(np.mean(aucX_rep))
        aucX_lam_std.append(np.std(aucX_rep))
        aucZ_lam_mean.append(np.mean(aucZ_rep))
        aucZ_lam_std.append(np.std(aucZ_rep))

    auc_corrupt_mean, auc_corrupt_std = [], []
    for cs in corrupt_std_grid:
        auc_rep = []
        for _ in range(repeats):
            Xtr, _, ytr = sample_XZ_from_world(base_world, rng, n=n_label_fixed)
            Ztr_hat = h_fixed(Xtr)
            Ztr_cor = Ztr_hat + rng.normal(size=Ztr_hat.shape) * cs
            Zte_cor = Zte_hat + rng.normal(size=Zte_hat.shape) * cs
            auc_rep.append(learned_auc_logreg(Ztr_cor, ytr, Zte_cor, yte, C=C_Z))
        auc_corrupt_mean.append(np.mean(auc_rep))
        auc_corrupt_std.append(np.std(auc_rep))

    phase_delta = np.zeros((len(phase_lam_grid), len(phase_n_grid)))
    for i, lam in enumerate(phase_lam_grid):
        world_lam = dict(base_world)
        world_lam["lam"] = float(lam)
        Xte2, _, yte2 = sample_XZ_from_world(world_lam, rng, n=8000)
        Zte2 = h_fixed(Xte2)
        for j, nlab in enumerate(phase_n_grid):
            aucX_rep, aucZ_rep = [], []
            for _ in range(max(3, repeats // 2)):
                Xtr2, _, ytr2 = sample_XZ_from_world(world_lam, rng, n=nlab)
                Ztr2 = h_fixed(Xtr2)
                aucX_rep.append(learned_auc_logreg(Xtr2, ytr2, Xte2, yte2, C=C_X))
                aucZ_rep.append(learned_auc_logreg(Ztr2, ytr2, Zte2, yte2, C=C_Z))
            phase_delta[i, j] = np.mean(aucZ_rep) - np.mean(aucX_rep)

    return {
        "n_pair_grid": np.array(n_pair_grid),
        "n_label_grid": np.array(n_label_grid),
        "lam_grid": np.array(lam_grid),
        "corrupt_std_grid": np.array(corrupt_std_grid),
        "phase_n_grid": np.array(phase_n_grid),
        "phase_lam_grid": np.array(phase_lam_grid),
        "miX_bits": miX_bits,
        "miZ_bits": miZ_bits,
        "aucX_star_list": np.array(aucX_star_list),
        "aucZ_star_list": np.array(aucZ_star_list),
        "aucX_star": float(aucX_star),
        "aucZ_star_mean": float(np.mean(aucZ_star_list)),
        "aucX_n_mean": np.array(aucX_n_mean),
        "aucX_n_std": np.array(aucX_n_std),
        "aucZ_n_mean": np.array(aucZ_n_mean),
        "aucZ_n_std": np.array(aucZ_n_std),
        "aucX_lam_mean": np.array(aucX_lam_mean),
        "aucX_lam_std": np.array(aucX_lam_std),
        "aucZ_lam_mean": np.array(aucZ_lam_mean),
        "aucZ_lam_std": np.array(aucZ_lam_std),
        "auc_corrupt_mean": np.array(auc_corrupt_mean),
        "auc_corrupt_std": np.array(auc_corrupt_std),
        "phase_delta": phase_delta,
        "lam_hard": float(lam_hard),
        "n_label_fixed": int(n_label_fixed),
    }


def sample_XZ_from_world_site(
    params: dict[str, Any],
    rng: np.random.Generator,
    n: int,
    site_probs: tuple[float, float] = (0.5, 0.5),
    pi_site: tuple[float, float] = (0.5, 0.5),
    delta_site_x: float = 0.0,
    gamma_site_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d, m, p, r = params["d"], params["m"], params["p"], params["r"]
    sigma_x, sigma_z, lam = params["sigma_x"], params["sigma_z"], params["lam"]
    bn = params["biology_noise"]
    b0, A, T, Q = params["b0"], params["A"], params["T"], params["Q"]

    if "u_site" not in params:
        u = rng.normal(size=p)
        params["u_site"] = u / (np.linalg.norm(u) + 1e-12)
    u_site = params["u_site"]

    site_probs_arr = np.array(site_probs, dtype=float)
    site_probs_arr /= site_probs_arr.sum()
    S = rng.choice(2, size=n, p=site_probs_arr)
    probs = np.where(S == 0, pi_site[0], pi_site[1])
    Y = (rng.random(size=n) < probs).astype(int)
    sgn = 2 * Y - 1

    xi = rng.normal(size=(n, m)) * bn
    B = sgn[:, None] * b0[None, :] + xi
    Z_true = B @ T.T + rng.normal(size=(n, p)) * sigma_z
    if gamma_site_z != 0.0:
        Z_true = Z_true + gamma_site_z * (2 * S - 1)[:, None] * u_site[None, :]
    ncoef = rng.normal(size=(n, r))
    eps = rng.normal(size=(n, d)) * sigma_x
    X = B @ A.T + lam * (ncoef @ Q.T) + eps
    if delta_site_x != 0.0:
        X = X + delta_site_x * (2 * S - 1)[:, None] * Q[:, [0]].T
    return X, Z_true, Y, S


def compute_site_shift_threat(
    seed: int = 11,
    d: int = 1000,
    m: int = 12,
    p: int = 8,
    r: int = 200,
    sigma_x: float = 1.0,
    sigma_z: float = 0.7,
    biology_noise: float = 1.0,
    lam_hard: float = 6.0,
    n_pair: int = 3000,
    ridge_alpha: float = 10.0,
    n_train: int = 800,
    n_test: int = 12000,
    site_probs: tuple[float, float] = (0.5, 0.5),
    pi_train: tuple[float, float] = (0.9, 0.1),
    pi_test: tuple[float, float] = (0.9, 0.1),
    delta_site_x: float = 6.0,
    gamma_site_z: float = 4.0,
    halluc_eta: float = 4.0,
    C_X: float = 0.03,
    C_Z: float = 1.0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    world = sample_world(
        rng,
        d=d,
        m=m,
        p=p,
        r=r,
        sigma_x=sigma_x,
        sigma_z=sigma_z,
        lam=lam_hard,
        biology_noise=biology_noise,
    )
    Xp, Zp, _, _ = sample_XZ_from_world_site(
        world,
        rng,
        n_pair,
        site_probs=site_probs,
        pi_site=(0.5, 0.5),
        delta_site_x=delta_site_x,
        gamma_site_z=gamma_site_z,
    )
    model_h, h = fit_translator_ridge(Xp, Zp, alpha=ridge_alpha)
    v_site = world["Q"][:, 0]
    u_site = world["u_site"]

    def h_halluc(X: np.ndarray) -> np.ndarray:
        Zhat = h(X)
        scomp = X @ v_site
        scomp = scomp / (np.std(scomp) + 1e-12)
        return Zhat + halluc_eta * scomp[:, None] * u_site[None, :]

    Xtr, Ztr, ytr, Str = sample_XZ_from_world_site(
        world,
        rng,
        n_train,
        site_probs=site_probs,
        pi_site=pi_train,
        delta_site_x=delta_site_x,
        gamma_site_z=gamma_site_z,
    )
    Xte, Zte, yte, Ste = sample_XZ_from_world_site(
        world,
        rng,
        n_test,
        site_probs=site_probs,
        pi_site=pi_test,
        delta_site_x=delta_site_x,
        gamma_site_z=gamma_site_z,
    )
    Ztr_hat, Zte_hat = h(Xtr), h(Xte)
    Ztr_hal, Zte_hal = h_halluc(Xtr), h_halluc(Xte)
    return {
        "aucY_X": learned_auc_logreg(Xtr, ytr, Xte, yte, C=C_X),
        "aucY_Z": learned_auc_logreg(Ztr_hat, ytr, Zte_hat, yte, C=C_Z),
        "aucY_H": learned_auc_logreg(Ztr_hal, ytr, Zte_hal, yte, C=C_Z),
        "aucS_X": learned_auc_logreg(Xtr, Str, Xte, Ste, C=0.2),
        "aucS_Z": learned_auc_logreg(Ztr_hat, Str, Zte_hat, Ste, C=0.2),
        "aucS_H": learned_auc_logreg(Ztr_hal, Str, Zte_hal, Ste, C=0.2),
        "corrZ": mean_channel_corr(Zte_hat, Zte),
        "corrH": mean_channel_corr(Zte_hal, Zte),
    }


def make_all_linear_panels_and_save(outdir: str = "figs", prefix: str = "fig3_", **kwargs: Any) -> dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    data = compute_core_panel_data(**kwargs)

    fig_ab = plot_fig_ab(data)
    fig_ab.savefig(os.path.join(outdir, f"{prefix}ab_dpi_ceiling.pdf"), bbox_inches="tight", pad_inches=0.03)
    fig_ab.savefig(os.path.join(outdir, f"{prefix}ab_dpi_ceiling.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig_ab)

    fig_cd = plot_fig_cd(data)
    fig_cd.savefig(os.path.join(outdir, f"{prefix}cd_learnability_nuisance.pdf"), bbox_inches="tight", pad_inches=0.03)
    fig_cd.savefig(os.path.join(outdir, f"{prefix}cd_learnability_nuisance.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig_cd)

    fig_ef = plot_fig_ef(data)
    fig_ef.savefig(os.path.join(outdir, f"{prefix}ef_fidelity_phase.pdf"), bbox_inches="tight", pad_inches=0.03)
    fig_ef.savefig(os.path.join(outdir, f"{prefix}ef_fidelity_phase.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig_ef)

    out_matched = compute_site_shift_threat(
        seed=kwargs.get("seed", 7) + 101,
        d=kwargs.get("d", 1000),
        m=kwargs.get("m", 12),
        p=kwargs.get("p", 8),
        r=kwargs.get("r", 200),
        sigma_x=kwargs.get("sigma_x", 1.0),
        sigma_z=kwargs.get("sigma_z", 0.7),
        biology_noise=kwargs.get("biology_noise", 1.0),
        lam_hard=kwargs.get("lam_hard", 6.0),
        n_pair=int(kwargs.get("n_pair_grid", (200, 500, 1000, 3000, 8000))[3]),
        ridge_alpha=kwargs.get("ridge_alpha", 10.0),
        n_train=800,
        n_test=12000,
        pi_train=(0.9, 0.1),
        pi_test=(0.9, 0.1),
        delta_site_x=6.0,
        gamma_site_z=4.0,
        halluc_eta=4.0,
        C_X=kwargs.get("C_X", 0.03),
        C_Z=kwargs.get("C_Z", 1.0),
    )
    out_reversed = compute_site_shift_threat(
        seed=kwargs.get("seed", 7) + 202,
        d=kwargs.get("d", 1000),
        m=kwargs.get("m", 12),
        p=kwargs.get("p", 8),
        r=kwargs.get("r", 200),
        sigma_x=kwargs.get("sigma_x", 1.0),
        sigma_z=kwargs.get("sigma_z", 0.7),
        biology_noise=kwargs.get("biology_noise", 1.0),
        lam_hard=kwargs.get("lam_hard", 6.0),
        n_pair=int(kwargs.get("n_pair_grid", (200, 500, 1000, 3000, 8000))[3]),
        ridge_alpha=kwargs.get("ridge_alpha", 10.0),
        n_train=800,
        n_test=12000,
        pi_train=(0.9, 0.1),
        pi_test=(0.1, 0.9),
        delta_site_x=6.0,
        gamma_site_z=4.0,
        halluc_eta=4.0,
        C_X=kwargs.get("C_X", 0.03),
        C_Z=kwargs.get("C_Z", 1.0),
    )

    fig_gh = plot_fig_gh_threat(out_matched, out_reversed)
    fig_gh.savefig(os.path.join(outdir, f"{prefix}gh_site_shortcut_threat.pdf"), bbox_inches="tight", pad_inches=0.03)
    fig_gh.savefig(os.path.join(outdir, f"{prefix}gh_site_shortcut_threat.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig_gh)
    return data
