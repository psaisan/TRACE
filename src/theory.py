import numpy as np
from sklearn.linear_model import Ridge

from .utils import norm_cdf


def delta2_from_gaussian(mu_vec: np.ndarray, Sigma: np.ndarray) -> float:
    return float(mu_vec @ np.linalg.solve(Sigma, mu_vec))


def bayes_auc_from_delta2(delta2: float) -> float:
    return float(norm_cdf(np.sqrt(2.0 * max(delta2, 0.0))))


def mi_from_delta2_mc(delta2: float, n_mc: int = 40000, seed: int = 0) -> float:
    delta2 = max(float(delta2), 0.0)
    if delta2 == 0.0:
        return 0.0
    rng = np.random.default_rng(seed)
    T = rng.normal(loc=delta2, scale=np.sqrt(delta2), size=n_mc)
    out = np.log(2.0) - np.logaddexp(0.0, -2.0 * T)
    return float(np.mean(out))


def gaussian_params_for_X(params: dict) -> tuple[np.ndarray, np.ndarray]:
    b0, A, Q = params["b0"], params["A"], params["Q"]
    bn, lam, sigma_x = params["biology_noise"], params["lam"], params["sigma_x"]
    d = params["d"]
    muX = A @ b0
    SigmaX = (bn ** 2) * (A @ A.T) + (lam ** 2) * (Q @ Q.T) + (sigma_x ** 2) * np.eye(d)
    return muX, SigmaX


def gaussian_params_for_Zhat(muX: np.ndarray, SigmaX: np.ndarray, translator_model: Ridge) -> tuple[np.ndarray, np.ndarray]:
    W = translator_model.coef_
    muZ = W @ muX
    SigmaZ = W @ SigmaX @ W.T
    return muZ, SigmaZ
