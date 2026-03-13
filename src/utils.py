import math
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def make_orthonormal_basis(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, r))
    Q, _ = np.linalg.qr(A)
    return Q[:, :r]


def learned_auc_logreg(
    train_feat: np.ndarray,
    ytr: np.ndarray,
    test_feat: np.ndarray,
    yte: np.ndarray,
    C: float = 0.05,
    max_iter: int = 4000,
) -> float:
    ytr = np.asarray(ytr).astype(int)
    yte = np.asarray(yte).astype(int)
    if np.unique(ytr).size < 2:
        return 0.5

    sc = StandardScaler(with_mean=True, with_std=True)
    Xtr = sc.fit_transform(train_feat)
    Xte = sc.transform(test_feat)

    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p))


def mean_channel_corr(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    A0 = A - A.mean(axis=0, keepdims=True)
    B0 = B - B.mean(axis=0, keepdims=True)
    num = (A0 * B0).mean(axis=0)
    den = np.sqrt((A0 * A0).mean(axis=0) * (B0 * B0).mean(axis=0) + eps)
    return float(np.mean(num / den))


def norm_cdf(x: float | np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / np.sqrt(2.0)))


def gaussian_kernel1d(sigma_pts: float = 1.4, radius: int | None = None) -> np.ndarray:
    if radius is None:
        radius = int(max(2, np.ceil(3 * sigma_pts)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)
    k /= k.sum()
    return k


def smooth_series_logx(
    x: np.ndarray,
    y: np.ndarray,
    sigma_pts: float = 1.4,
    upsample: int = 700,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx = np.log10(x)
    lx_dense = np.linspace(lx.min(), lx.max(), upsample)
    y_dense = np.interp(lx_dense, lx, y)
    k = gaussian_kernel1d(sigma_pts=sigma_pts)
    pad = len(k) // 2
    y_pad = np.pad(y_dense, (pad, pad), mode="edge")
    y_smooth = np.convolve(y_pad, k, mode="same")[pad:-pad]
    x_smooth = 10 ** lx_dense
    return x_smooth, y_smooth


def compute_band(stat_mat: np.ndarray, band_mode: str = "ci95") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = stat_mat.mean(axis=0)
    if stat_mat.shape[0] > 1:
        std = stat_mat.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    sem = std / np.sqrt(max(1, stat_mat.shape[0]))

    if band_mode == "std":
        band = std
    elif band_mode == "sem":
        band = sem
    elif band_mode == "ci95":
        band = 1.96 * sem
    else:
        raise ValueError("band_mode must be one of: 'std', 'sem', 'ci95'")

    return mean, band, std, sem


def make_dense_n_grid(
    n_min: int = 30,
    n_mid: int = 1000,
    n_max: int = 10000,
    n_low: int = 30,
    n_high: int = 10,
) -> tuple[int, ...]:
    low = np.unique(np.round(np.logspace(np.log10(n_min), np.log10(n_mid), n_low)).astype(int))
    high = np.unique(np.round(np.logspace(np.log10(max(n_mid + 1, 1200)), np.log10(n_max), n_high)).astype(int))
    return tuple(np.unique(np.concatenate([low, high])).tolist())


def centered_square(v: np.ndarray) -> np.ndarray:
    return v**2 - np.mean(v**2)


def safe_standardize(X: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, mu, sd
