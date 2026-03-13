from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import compute_band, smooth_series_logx


def plot_fig_ab(data: dict) -> plt.Figure:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    n_pair = data["n_pair_grid"]
    axA.plot(n_pair, data["miX_bits"], marker="o", label=r"$I(Y;X)$")
    axA.plot(n_pair, data["miZ_bits"], marker="o", label=r"$I(Y;\hat Z)$")
    axA.set_xscale("log")
    axA.set_xlabel(r"paired translator training size $n_{\mathrm{pair}}$")
    axA.set_ylabel("Mutual information (bits)")
    axA.set_title("A) DPI check")
    axA.legend(frameon=False)

    axB.plot(n_pair, data["aucX_star_list"], marker="o", label=r"$\mathrm{AUC}^*(X)$")
    axB.plot(n_pair, data["aucZ_star_list"], marker="o", label=r"$\mathrm{AUC}^*(\hat Z)$")
    axB.set_xscale("log")
    axB.set_xlabel(r"paired translator training size $n_{\mathrm{pair}}$")
    axB.set_ylabel("Bayes AUC ceiling")
    axB.set_ylim(0.5, 1.0)
    axB.set_title("B) Ceiling ordering")
    axB.legend(frameon=False)
    return fig


def plot_fig_cd(data: dict) -> plt.Figure:
    fig, (axC, axD) = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    n_label = data["n_label_grid"]
    axC.errorbar(n_label, data["aucX_n_mean"], yerr=data["aucX_n_std"], marker="o", capsize=3, label=r"$\mathrm{AUC}_n(X)$")
    axC.errorbar(n_label, data["aucZ_n_mean"], yerr=data["aucZ_n_std"], marker="o", capsize=3, label=r"$\mathrm{AUC}_n(\hat Z)$")
    axC.axhline(data["aucX_star"], linestyle="--", linewidth=1, label=r"$\mathrm{AUC}^*(X)$")
    axC.axhline(data["aucZ_star_mean"], linestyle="--", linewidth=1, label=r"$\mathrm{AUC}^*(\hat Z)$ avg")
    axC.set_xscale("log")
    axC.set_xlabel(r"labeled slides $n_{\mathrm{label}}$")
    axC.set_ylabel("Test AUC")
    axC.set_ylim(0.45, 1.0)
    axC.set_title(rf"C) Learnability @ high nuisance ($\lambda={data['lam_hard']}$)")
    axC.legend(frameon=False)

    lam = data["lam_grid"]
    axD.errorbar(lam, data["aucX_lam_mean"], yerr=data["aucX_lam_std"], marker="o", capsize=3, label=r"$\mathrm{AUC}_n(X)$")
    axD.errorbar(lam, data["aucZ_lam_mean"], yerr=data["aucZ_lam_std"], marker="o", capsize=3, label=r"$\mathrm{AUC}_n(\hat Z)$")
    axD.axhline(0.5, linewidth=1)
    axD.set_xlabel(r"nuisance strength $\lambda$")
    axD.set_ylabel("Test AUC")
    axD.set_ylim(0.45, 1.0)
    axD.set_title(rf"D) Nuisance dependence @ fixed $n_{{\mathrm{{label}}}}={data['n_label_fixed']}$")
    axD.legend(frameon=False)
    return fig


def plot_fig_ef(data: dict) -> plt.Figure:
    fig, (axE, axF) = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    cs = data["corrupt_std_grid"]
    axE.errorbar(cs, data["auc_corrupt_mean"], yerr=data["auc_corrupt_std"], marker="o", capsize=3)
    axE.set_xlabel("translation corruption std")
    axE.set_ylabel(r"$\mathrm{AUC}_n(\hat Z_{\mathrm{corrupt}})$")
    axE.set_ylim(0.45, 1.0)
    axE.set_title("E) Fidelity dependence")

    phase_delta = data["phase_delta"]
    ngrid = data["phase_n_grid"]
    lamgrid = data["phase_lam_grid"]
    im = axF.imshow(phase_delta, aspect="auto", origin="lower", interpolation="nearest")
    axF.set_title(r"F) Phase diagram: $\Delta$AUC = AUC($\hat Z$) - AUC(X)")
    axF.set_xlabel(r"label budget $n_{\mathrm{label}}$")
    axF.set_ylabel(r"nuisance strength $\lambda$")
    axF.set_xticks(np.arange(len(ngrid)))
    axF.set_xticklabels([str(int(x)) for x in ngrid])
    axF.set_yticks(np.arange(len(lamgrid)))
    axF.set_yticklabels([str(float(x)) for x in lamgrid])
    divider = make_axes_locatable(axF)
    cax = divider.append_axes("right", size="4.5%", pad=0.10)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\Delta$AUC")
    return fig


def plot_fig_gh_threat(out_matched: dict, out_reversed: dict) -> plt.Figure:
    fig, (axG, axH) = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    reps = ["X", "Zhat", "Zhat+halluc"]
    aucY_mat = [out_matched["aucY_X"], out_matched["aucY_Z"], out_matched["aucY_H"]]
    aucY_rev = [out_reversed["aucY_X"], out_reversed["aucY_Z"], out_reversed["aucY_H"]]
    aucS_mat = [out_matched["aucS_X"], out_matched["aucS_Z"], out_matched["aucS_H"]]
    aucS_rev = [out_reversed["aucS_X"], out_reversed["aucS_Z"], out_reversed["aucS_H"]]
    x = np.arange(len(reps))
    w = 0.38

    axG.bar(x - w / 2, aucY_mat, width=w, label="matched train/test")
    axG.bar(x + w / 2, aucY_rev, width=w, label="reversed confounding at test")
    axG.set_xticks(x)
    axG.set_xticklabels(reps)
    axG.set_ylim(0.45, 1.0)
    axG.set_ylabel("Test AUC on Y")
    axG.set_title("G) Shortcut sensitivity under site-label shift")
    axG.axhline(0.5, linewidth=1)
    axG.legend(frameon=False)

    axH.bar(x - w / 2, aucS_mat, width=w, label="matched train/test")
    axH.bar(x + w / 2, aucS_rev, width=w, label="reversed confounding at test")
    axH.set_xticks(x)
    axH.set_xticklabels(reps)
    axH.set_ylim(0.45, 1.0)
    axH.set_ylabel("Test AUC on site S")
    axH.set_title("H) Site leakage probe")
    axH.axhline(0.5, linewidth=1)
    axH.legend(frameon=False)
    return fig


def plot_auc_learning_curve_smoothed(
    curve: dict,
    auc_ceiling: float | None = None,
    title: str | None = None,
    smooth_sigma_pts: float = 1.5,
    show_raw_points: bool = True,
    show_mean_line: bool = True,
    show_ci: bool = True,
    point_alpha: float = 0.18,
    ci_alpha: float = 0.15,
    figsize: tuple[float, float] = (8.2, 5.2),
) -> plt.Figure:
    n = np.asarray(curve["n"], dtype=float)
    aucX = np.asarray(curve["aucX"], dtype=float)
    aucZ = np.asarray(curve["aucZ"], dtype=float)
    mx = aucX.mean(axis=0)
    mz = aucZ.mean(axis=0)
    if aucX.shape[0] > 1:
        sx = aucX.std(axis=0, ddof=1) / np.sqrt(aucX.shape[0])
        sz = aucZ.std(axis=0, ddof=1) / np.sqrt(aucZ.shape[0])
    else:
        sx = np.zeros_like(mx)
        sz = np.zeros_like(mz)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    if show_raw_points:
        rng = np.random.default_rng(12345)
        for j, nj in enumerate(n):
            jitter_x_X = nj * np.exp(rng.normal(0.0, 0.022, size=aucX.shape[0]))
            jitter_x_Z = nj * np.exp(rng.normal(0.0, 0.022, size=aucZ.shape[0]))
            ax.scatter(jitter_x_Z, aucZ[:, j], s=18, alpha=point_alpha)
            ax.scatter(jitter_x_X, aucX[:, j], s=18, alpha=point_alpha)
    if show_ci:
        ax.fill_between(n, mz - 1.96 * sz, mz + 1.96 * sz, alpha=ci_alpha)
        ax.fill_between(n, mx - 1.96 * sx, mx + 1.96 * sx, alpha=ci_alpha)
    if show_mean_line:
        ax.semilogx(n, mz, linewidth=1.2, alpha=0.70, marker="o", markersize=3.5, label=r"$AUC_n(\hat Z)$ raw mean")
        ax.semilogx(n, mx, linewidth=1.2, alpha=0.70, marker="o", markersize=3.5, label=r"$AUC_n(X)$ raw mean")
    xs_z, ys_z = smooth_series_logx(n, mz, sigma_pts=smooth_sigma_pts, upsample=600)
    xs_x, ys_x = smooth_series_logx(n, mx, sigma_pts=smooth_sigma_pts, upsample=600)
    ax.semilogx(xs_z, ys_z, linewidth=3.0, label=r"$AUC_n(\hat Z)$ smoothed")
    ax.semilogx(xs_x, ys_x, linewidth=3.0, label=r"$AUC_n(X)$ smoothed")
    if auc_ceiling is not None:
        ax.axhline(float(auc_ceiling), linestyle="--", linewidth=2.0, label=r"$AUC^*(X)$ ceiling")
    ax.set_xlabel(r"labeled samples $n$ (log scale)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, fontsize=10)
    if title is not None:
        ax.set_title(title)
    return fig


def plot_auc_learning_curve_smoothed_with_band(
    curve: dict,
    auc_ceiling: float | None = None,
    title: str | None = None,
    smooth_sigma_pts_mean: float = 1.5,
    smooth_sigma_pts_band: float = 1.8,
    band_mode: str = "std",
    show_raw_points: bool = True,
    show_raw_mean: bool = True,
    show_smoothed_mean: bool = True,
    show_band: bool = True,
    point_alpha: float = 0.12,
    raw_mean_alpha: float = 0.45,
    band_alpha: float = 0.18,
    figsize: tuple[float, float] = (8.4, 5.4),
) -> plt.Figure:
    n = np.asarray(curve["n"], dtype=float)
    aucX = np.asarray(curve["aucX"], dtype=float)
    aucZ = np.asarray(curve["aucZ"], dtype=float)
    meanX, bandX, _, _ = compute_band(aucX, band_mode=band_mode)
    meanZ, bandZ, _, _ = compute_band(aucZ, band_mode=band_mode)
    xsX, meanX_s = smooth_series_logx(n, meanX, sigma_pts=smooth_sigma_pts_mean, upsample=700)
    xsZ, meanZ_s = smooth_series_logx(n, meanZ, sigma_pts=smooth_sigma_pts_mean, upsample=700)
    _, bandX_s = smooth_series_logx(n, bandX, sigma_pts=smooth_sigma_pts_band, upsample=700)
    _, bandZ_s = smooth_series_logx(n, bandZ, sigma_pts=smooth_sigma_pts_band, upsample=700)
    bandX_s = np.maximum(bandX_s, 0.0)
    bandZ_s = np.maximum(bandZ_s, 0.0)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    if show_raw_points:
        rng = np.random.default_rng(12345)
        for j, nj in enumerate(n):
            jitter_x_X = nj * np.exp(rng.normal(0.0, 0.020, size=aucX.shape[0]))
            jitter_x_Z = nj * np.exp(rng.normal(0.0, 0.020, size=aucZ.shape[0]))
            ax.scatter(jitter_x_Z, aucZ[:, j], s=16, alpha=point_alpha)
            ax.scatter(jitter_x_X, aucX[:, j], s=16, alpha=point_alpha)
    if show_raw_mean:
        ax.semilogx(n, meanZ, marker="o", markersize=3.2, linewidth=1.1, alpha=raw_mean_alpha, label=r"$AUC_n(\hat Z)$ raw mean")
        ax.semilogx(n, meanX, marker="o", markersize=3.2, linewidth=1.1, alpha=raw_mean_alpha, label=r"$AUC_n(X)$ raw mean")
    if show_band:
        ax.fill_between(xsZ, meanZ_s - bandZ_s, meanZ_s + bandZ_s, alpha=band_alpha, label=rf"$\hat Z$ {band_mode} band")
        ax.fill_between(xsX, meanX_s - bandX_s, meanX_s + bandX_s, alpha=band_alpha, label=rf"$X$ {band_mode} band")
    if show_smoothed_mean:
        ax.semilogx(xsZ, meanZ_s, linewidth=3.0, label=r"$AUC_n(\hat Z)$ smoothed")
        ax.semilogx(xsX, meanX_s, linewidth=3.0, label=r"$AUC_n(X)$ smoothed")
    if auc_ceiling is not None:
        ax.axhline(float(auc_ceiling), linestyle="--", linewidth=2.0, label=r"$AUC^*(X)$ ceiling")
    ax.set_xlabel(r"labeled samples $n$ (log scale)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 1.0)
    if title is not None:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    return fig
