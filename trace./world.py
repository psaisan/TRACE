import math
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from .presets import regime_defaults


def make_orthonormal_basis(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, r))
    Q, _ = np.linalg.qr(A)
    return Q[:, :r]


def _norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def delta2_from_gaussian(mu_vec: np.ndarray, Sigma: np.ndarray) -> float:
    return float(mu_vec @ np.linalg.solve(Sigma, mu_vec))


def bayes_auc_from_delta2(delta2: float) -> float:
    return float(_norm_cdf(np.sqrt(2.0 * max(delta2, 0.0))))


def make_biology_aligned_T(
    m: int,
    p: int,
    rng: np.random.Generator,
    jitter: float = 0.05,
) -> np.ndarray:
    T = np.zeros((p, m))
    diag_dim = min(p, m)
    T[:diag_dim, :diag_dim] = np.eye(diag_dim)
    if jitter > 0:
        T = T + jitter * rng.normal(size=(p, m)) / np.sqrt(max(m, 1))
    return T


def sample_world(
    rng: np.random.Generator,
    d: int = 1000,
    m: int = 12,
    p: int = 12,
    r: int = 200,
    sigma_x: float = 1.0,
    sigma_z: float = 0.4,
    lam: float = 8.0,
    biology_noise: float = 1.0,
    T_mode: str = "identity_like",
    T_jitter: float = 0.05,
) -> Dict[str, Any]:
    b0 = rng.normal(size=m)
    b0 /= (np.linalg.norm(b0) + 1e-12)

    A = rng.normal(size=(d, m)) / np.sqrt(m)

    if T_mode == "identity_like":
        T = make_biology_aligned_T(m, p, rng, jitter=T_jitter)
    elif T_mode == "random":
        T = rng.normal(size=(p, m)) / np.sqrt(m)
    else:
        raise ValueError("T_mode must be 'identity_like' or 'random'")

    Q = make_orthonormal_basis(d, r, rng)

    return dict(
        d=d,
        m=m,
        p=p,
        r=r,
        sigma_x=sigma_x,
        sigma_z=sigma_z,
        lam=lam,
        biology_noise=biology_noise,
        b0=b0,
        A=A,
        T=T,
        Q=Q,
        T_mode=T_mode,
        T_jitter=T_jitter,
    )


def sample_XZ_from_world(params: Dict[str, Any], rng: np.random.Generator, n: int):
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


def fit_translator_ridge(X_pair: np.ndarray, Z_pair: np.ndarray, alpha: float = 1.0):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_pair, Z_pair)

    def h(X):
        return model.predict(X)

    return model, h


def gaussian_params_for_X(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    b0, A, Q = params["b0"], params["A"], params["Q"]
    bn, lam, sigma_x = params["biology_noise"], params["lam"], params["sigma_x"]

    muX = A @ b0
    SigmaX = (bn ** 2) * (A @ A.T) + (lam ** 2) * (Q @ Q.T) + (sigma_x ** 2) * np.eye(params["d"])
    return muX, SigmaX


def gaussian_params_for_Zhat(muX: np.ndarray, SigmaX: np.ndarray, translator_model: Ridge):
    W = translator_model.coef_
    muZ = W @ muX
    SigmaZ = W @ SigmaX @ W.T
    return muZ, SigmaZ


def generate_linear_trace_dataset(
    regime: str,
    n_samples: int = 4000,
    seed: int = 7,
    **overrides,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    params = regime_defaults(regime)
    params.update(overrides)

    n_pair = int(params.pop("n_pair"))
    ridge_alpha = float(params.pop("ridge_alpha"))
    C_X = float(params.pop("C_X"))
    C_H = float(params.pop("C_H"))

    world = sample_world(rng, **params)

    Xp, Zp, _ = sample_XZ_from_world(world, rng, n=n_pair)
    translator_model, h = fit_translator_ridge(Xp, Zp, alpha=ridge_alpha)

    X, _, y = sample_XZ_from_world(world, rng, n=n_samples)
    H = h(X)

    muX, SigmaX = gaussian_params_for_X(world)
    aucX_star = bayes_auc_from_delta2(delta2_from_gaussian(muX, SigmaX))

    muH, SigmaH = gaussian_params_for_Zhat(muX, SigmaX, translator_model)
    aucH_star = bayes_auc_from_delta2(delta2_from_gaussian(muH, SigmaH))

    meta = dict(
        regime=regime,
        seed=int(seed),
        n_samples=int(n_samples),
        n_pair=int(n_pair),
        ridge_alpha=float(ridge_alpha),
        C_X=float(C_X),
        C_H=float(C_H),
        aucX_star=float(aucX_star),
        aucH_star=float(aucH_star),
        d=int(world["d"]),
        m=int(world["m"]),
        p=int(world["p"]),
        r=int(world["r"]),
        sigma_x=float(world["sigma_x"]),
        sigma_z=float(world["sigma_z"]),
        lam=float(world["lam"]),
        biology_noise=float(world["biology_noise"]),
        T_mode=str(world["T_mode"]),
        T_jitter=float(world["T_jitter"]),
    )

    return {
        "X": X,
        "H": H,
        "y": y,
        "meta": meta,
        "world": world,
        "translator_model": translator_model,
    }
