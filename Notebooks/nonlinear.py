from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.linear_model import Ridge

from .utils import centered_square, learned_auc_logreg, make_orthonormal_basis, safe_standardize


def sample_nonlinear_world(
    rng: np.random.Generator,
    d: int = 3000,
    m: int = 20,
    p: int = 20,
    r: int = 500,
    sigma_x: float = 1.0,
    sigma_z: float = 0.50,
    biology_noise: float = 1.2,
    lam: float = 7.0,
    x_nl_strength: float = 1.15,
    z_nl_strength: float = 0.45,
    x_mix_quad: float = 0.20,
    z_mix_quad: float = 0.08,
    y_nonlinearity: float = 0.85,
    latent_axis_strength: float = 1.0,
    label_temperature: float = 1.6,
) -> dict[str, Any]:
    b0 = rng.normal(size=m)
    b0 /= np.linalg.norm(b0) + 1e-12
    b1 = rng.normal(size=m)
    b1 -= b0 * (b0 @ b1)
    b1 /= np.linalg.norm(b1) + 1e-12
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
        "biology_noise": biology_noise,
        "lam": lam,
        "x_nl_strength": x_nl_strength,
        "z_nl_strength": z_nl_strength,
        "x_mix_quad": x_mix_quad,
        "z_mix_quad": z_mix_quad,
        "y_nonlinearity": y_nonlinearity,
        "latent_axis_strength": latent_axis_strength,
        "label_temperature": label_temperature,
        "b0": b0,
        "b1": b1,
        "A": A,
        "T": T,
        "Q": Q,
    }


def _make_prob_labels_from_B(
    B: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    y_nonlinearity: float,
    label_temperature: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    s1 = B @ b0
    s2 = B @ b1
    score = s1 + y_nonlinearity * np.tanh(1.1 * s2) + 0.20 * centered_square(s2)
    score = score / (np.std(score) + 1e-12)
    score = score / label_temperature
    p = 1.0 / (1.0 + np.exp(-score))
    Y = (rng.random(size=B.shape[0]) < p).astype(int)
    return Y, score


def sample_XZ_from_world_nonlinear(params: dict[str, Any], rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, m, p, r = params["d"], params["m"], params["p"], params["r"]
    sigma_x, sigma_z, lam = params["sigma_x"], params["sigma_z"], params["lam"]
    bn = params["biology_noise"]
    x_nl_strength = params["x_nl_strength"]
    z_nl_strength = params["z_nl_strength"]
    x_mix_quad = params["x_mix_quad"]
    z_mix_quad = params["z_mix_quad"]
    y_nonlinearity = params["y_nonlinearity"]
    latent_axis_strength = params["latent_axis_strength"]
    label_temperature = params["label_temperature"]
    b0, b1, A, T, Q = params["b0"], params["b1"], params["A"], params["T"], params["Q"]

    B = rng.normal(size=(n, m)) * bn
    B += latent_axis_strength * rng.normal(size=(n, 1)) * b0[None, :]
    Y, _ = _make_prob_labels_from_B(B, b0, b1, y_nonlinearity, label_temperature, rng)

    Z_lin = B @ T.T
    Z_true = (
        (1.0 - z_mix_quad) * np.tanh(Z_lin / (1.0 + z_nl_strength))
        + z_mix_quad * 0.35 * (Z_lin ** 2)
        + rng.normal(size=(n, p)) * sigma_z
    )

    X_lin = B @ A.T
    X_nuis = lam * (rng.normal(size=(n, r)) @ Q.T)
    X = (
        (1.0 - x_mix_quad) * np.tanh((X_lin + 0.35 * X_nuis) / (1.0 + x_nl_strength))
        + x_mix_quad * 0.20 * (X_lin ** 2)
        + 0.65 * X_nuis
        + rng.normal(size=(n, d)) * sigma_x
    )
    return X, Z_true, Y


def fit_translator_ridge(X_pair: np.ndarray, Z_pair: np.ndarray, alpha: float = 10.0) -> tuple[Ridge, Any]:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_pair, Z_pair)

    def h(X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    return model, h


def fit_translator_rff_ridge(
    X_pair: np.ndarray,
    Z_pair: np.ndarray,
    alpha: float = 3.0,
    n_features: int = 4000,
    gamma: float | None = None,
    seed: int = 0,
) -> tuple[Ridge, Any]:
    rng = np.random.default_rng(seed)
    Xs, mu, sd = safe_standardize(X_pair)
    d = Xs.shape[1]
    if gamma is None:
        gamma = 1.0 / d
    W = rng.normal(scale=np.sqrt(2.0 * gamma), size=(d, n_features))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    Phi = np.sqrt(2.0 / n_features) * np.cos(Xs @ W + b[None, :])
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Phi, Z_pair)

    def h(X: np.ndarray) -> np.ndarray:
        Xs_new = (X - mu) / sd
        Phi_new = np.sqrt(2.0 / n_features) * np.cos(Xs_new @ W + b[None, :])
        return model.predict(Phi_new)

    return model, h


def compute_learning_curve_auc_vs_n_nonlinear(
    n_label_grid: tuple[int, ...] | list[int] | np.ndarray,
    repeats: int = 10,
    seed: int = 7,
    d: int = 3000,
    m: int = 20,
    p: int = 20,
    r: int = 500,
    sigma_x: float = 1.0,
    sigma_z: float = 0.50,
    biology_noise: float = 1.2,
    lam_hard: float = 7.0,
    x_nl_strength: float = 1.15,
    z_nl_strength: float = 0.45,
    x_mix_quad: float = 0.20,
    z_mix_quad: float = 0.08,
    y_nonlinearity: float = 0.85,
    latent_axis_strength: float = 1.0,
    label_temperature: float = 1.6,
    n_pair: int = 4000,
    ridge_alpha: float = 3.0,
    n_test: int = 12000,
    C_X: float = 0.02,
    C_Z: float = 1.0,
    translator: Literal["ridge", "rff"] = "rff",
    translator_n_features: int = 4000,
    translator_gamma: float | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    world = sample_nonlinear_world(
        rng,
        d=d,
        m=m,
        p=p,
        r=r,
        sigma_x=sigma_x,
        sigma_z=sigma_z,
        biology_noise=biology_noise,
        lam=lam_hard,
        x_nl_strength=x_nl_strength,
        z_nl_strength=z_nl_strength,
        x_mix_quad=x_mix_quad,
        z_mix_quad=z_mix_quad,
        y_nonlinearity=y_nonlinearity,
        latent_axis_strength=latent_axis_strength,
        label_temperature=label_temperature,
    )
    Xp, Zp_true, _ = sample_XZ_from_world_nonlinear(world, rng, n=n_pair)
    if translator == "rff":
        _, h = fit_translator_rff_ridge(
            Xp,
            Zp_true,
            alpha=ridge_alpha,
            n_features=translator_n_features,
            gamma=translator_gamma,
            seed=seed + 111,
        )
    elif translator == "ridge":
        _, h = fit_translator_ridge(Xp, Zp_true, alpha=ridge_alpha)
    else:
        raise ValueError("translator must be 'ridge' or 'rff'")

    Xte, _, yte = sample_XZ_from_world_nonlinear(world, rng, n=n_test)
    Zte_hat = h(Xte)
    n_grid = np.array(list(n_label_grid), dtype=int)
    aucX = np.zeros((repeats, len(n_grid)), dtype=float)
    aucZ = np.zeros((repeats, len(n_grid)), dtype=float)
    for rep in range(repeats):
        rep_rng = np.random.default_rng(seed + 1000 * (rep + 1))
        for j, nlab in enumerate(n_grid):
            Xtr, _, ytr = sample_XZ_from_world_nonlinear(world, rep_rng, n=int(nlab))
            Ztr_hat = h(Xtr)
            aucX[rep, j] = learned_auc_logreg(Xtr, ytr, Xte, yte, C=C_X)
            aucZ[rep, j] = learned_auc_logreg(Ztr_hat, ytr, Zte_hat, yte, C=C_Z)
    return {
        "n": n_grid,
        "aucX": aucX,
        "aucZ": aucZ,
        "knobs": {
            "seed": seed,
            "repeats": repeats,
            "d": d,
            "m": m,
            "p": p,
            "r": r,
            "sigma_x": sigma_x,
            "sigma_z": sigma_z,
            "biology_noise": biology_noise,
            "lam_hard": lam_hard,
            "x_nl_strength": x_nl_strength,
            "z_nl_strength": z_nl_strength,
            "x_mix_quad": x_mix_quad,
            "z_mix_quad": z_mix_quad,
            "y_nonlinearity": y_nonlinearity,
            "latent_axis_strength": latent_axis_strength,
            "label_temperature": label_temperature,
            "n_pair": n_pair,
            "ridge_alpha": ridge_alpha,
            "n_test": n_test,
            "C_X": C_X,
            "C_Z": C_Z,
            "translator": translator,
            "translator_n_features": translator_n_features,
            "translator_gamma": translator_gamma,
        },
    }


def compute_nuisance_sweep_nonlinear(
    lam_grid: tuple[float, ...] | list[float] | np.ndarray,
    n_label: int = 200,
    repeats: int = 8,
    seed: int = 7,
    d: int = 3000,
    m: int = 20,
    p: int = 20,
    r: int = 500,
    sigma_x: float = 1.0,
    sigma_z: float = 0.50,
    biology_noise: float = 1.2,
    lam_train_translator: float = 7.0,
    x_nl_strength: float = 1.15,
    z_nl_strength: float = 0.45,
    x_mix_quad: float = 0.20,
    z_mix_quad: float = 0.08,
    y_nonlinearity: float = 0.85,
    latent_axis_strength: float = 1.0,
    label_temperature: float = 1.6,
    n_pair: int = 4000,
    ridge_alpha: float = 3.0,
    n_test: int = 12000,
    C_X: float = 0.02,
    C_Z: float = 1.0,
    translator: Literal["ridge", "rff"] = "rff",
    translator_n_features: int = 4000,
    translator_gamma: float | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_world = sample_nonlinear_world(
        rng,
        d=d,
        m=m,
        p=p,
        r=r,
        sigma_x=sigma_x,
        sigma_z=sigma_z,
        biology_noise=biology_noise,
        lam=lam_train_translator,
        x_nl_strength=x_nl_strength,
        z_nl_strength=z_nl_strength,
        x_mix_quad=x_mix_quad,
        z_mix_quad=z_mix_quad,
        y_nonlinearity=y_nonlinearity,
        latent_axis_strength=latent_axis_strength,
        label_temperature=label_temperature,
    )
    Xp, Zp_true, _ = sample_XZ_from_world_nonlinear(base_world, rng, n=n_pair)
    if translator == "rff":
        _, h = fit_translator_rff_ridge(
            Xp,
            Zp_true,
            alpha=ridge_alpha,
            n_features=translator_n_features,
            gamma=translator_gamma,
            seed=seed + 222,
        )
    elif translator == "ridge":
        _, h = fit_translator_ridge(Xp, Zp_true, alpha=ridge_alpha)
    else:
        raise ValueError("translator must be 'ridge' or 'rff'")

    aucX_mean, aucX_std, aucZ_mean, aucZ_std = [], [], [], []
    for lam in lam_grid:
        world_lam = dict(base_world)
        world_lam["lam"] = float(lam)
        Xte, _, yte = sample_XZ_from_world_nonlinear(world_lam, rng, n=n_test)
        Zte_hat = h(Xte)
        aucX_rep, aucZ_rep = [], []
        for rep in range(repeats):
            rep_rng = np.random.default_rng(seed + 5000 + rep + int(10 * lam))
            Xtr, _, ytr = sample_XZ_from_world_nonlinear(world_lam, rep_rng, n=n_label)
            Ztr_hat = h(Xtr)
            aucX_rep.append(learned_auc_logreg(Xtr, ytr, Xte, yte, C=C_X))
            aucZ_rep.append(learned_auc_logreg(Ztr_hat, ytr, Zte_hat, yte, C=C_Z))
        aucX_mean.append(np.mean(aucX_rep))
        aucX_std.append(np.std(aucX_rep))
        aucZ_mean.append(np.mean(aucZ_rep))
        aucZ_std.append(np.std(aucZ_rep))
    return {
        "lam": np.array(list(lam_grid), dtype=float),
        "aucX_mean": np.array(aucX_mean),
        "aucX_std": np.array(aucX_std),
        "aucZ_mean": np.array(aucZ_mean),
        "aucZ_std": np.array(aucZ_std),
    }
