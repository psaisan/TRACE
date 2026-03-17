from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


def _gaussian_kernel1d(sigma_pts: float = 1.4, radius: Optional[int] = None) -> np.ndarray:
    if radius is None:
        radius = int(max(2, np.ceil(3 * sigma_pts)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)
    k /= k.sum()
    return k


def _smooth_series_logx(x, y, sigma_pts: float = 1.4, upsample: int = 600):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lx = np.log10(x)
    lx_dense = np.linspace(lx.min(), lx.max(), upsample)
    y_dense = np.interp(lx_dense, lx, y)

    k = _gaussian_kernel1d(sigma_pts=sigma_pts)
    pad = len(k) // 2
    y_pad = np.pad(y_dense, (pad, pad), mode="edge")
    y_smooth = np.convolve(y_pad, k, mode="same")[pad:-pad]

    x_smooth = 10 ** lx_dense
    return x_smooth, y_smooth


def _compute_band(stat_mat: np.ndarray, band_mode: str = "std"):
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


def plot_learning_curves_on_ax(
    ax,
    result: Dict[str, Any],
    title: Optional[str] = None,
    band_mode: str = "std",
    smooth_sigma_pts_mean: float = 1.4,
    smooth_sigma_pts_band: float = 1.8,
):
    n = np.asarray(result["n"], dtype=float)
    aucX = np.asarray(result["aucX"], dtype=float)
    aucH = np.asarray(result["aucH"], dtype=float)

    meanX, bandX, _, _ = _compute_band(aucX, band_mode=band_mode)
    meanH, bandH, _, _ = _compute_band(aucH, band_mode=band_mode)

    xsX, meanX_s = _smooth_series_logx(n, meanX, sigma_pts=smooth_sigma_pts_mean, upsample=700)
    xsH, meanH_s = _smooth_series_logx(n, meanH, sigma_pts=smooth_sigma_pts_mean, upsample=700)
    _, bandX_s = _smooth_series_logx(n, bandX, sigma_pts=smooth_sigma_pts_band, upsample=700)
    _, bandH_s = _smooth_series_logx(n, bandH, sigma_pts=smooth_sigma_pts_band, upsample=700)

    bandX_s = np.maximum(bandX_s, 0.0)
    bandH_s = np.maximum(bandH_s, 0.0)

    ax.fill_between(xsH, meanH_s - bandH_s, meanH_s + bandH_s, alpha=0.18)
    ax.fill_between(xsX, meanX_s - bandX_s, meanX_s + bandX_s, alpha=0.18)
    ax.semilogx(xsH, meanH_s, linewidth=3.0, label=r"$AUC_n(H)$")
    ax.semilogx(xsX, meanX_s, linewidth=3.0, label=r"$AUC_n(X)$")

    ax.set_xlabel(r"labeled samples $n$")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 1.0)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8)


def plot_arc_on_ax(
    ax,
    result: Dict[str, Any],
    title: Optional[str] = None,
    band_mode: str = "std",
    smooth_sigma_pts_mean: float = 1.4,
    smooth_sigma_pts_band: float = 1.8,
):
    n = np.asarray(result["n"], dtype=float)
    arc = np.asarray(result["arc"], dtype=float)

    meanARC, bandARC, _, _ = _compute_band(arc, band_mode=band_mode)
    xs, mean_s = _smooth_series_logx(n, meanARC, sigma_pts=smooth_sigma_pts_mean, upsample=700)
    _, band_s = _smooth_series_logx(n, bandARC, sigma_pts=smooth_sigma_pts_band, upsample=700)
    band_s = np.maximum(band_s, 0.0)

    ax.fill_between(xs, mean_s - band_s, mean_s + band_s, alpha=0.20)
    ax.semilogx(xs, mean_s, linewidth=3.0, label=r"$ARC(n)$")
    ax.axhline(0.0, linestyle="--", linewidth=1.2)

    ax.set_xlabel(r"labeled samples $n$")
    ax.set_ylabel("ARC")
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8)


def plot_regime_scoreboard(ax, scores: Dict[str, float], title: str = "TRACE regime meter"):
    labels = [
        "sample_efficiency",
        "persistent_advantage",
        "no_advantage",
        "lossy_translation",
    ]
    values = [float(scores.get(k, 0.0)) for k in labels]

    y = np.arange(len(labels))
    ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels([
        "sample efficiency",
        "persistent advantage",
        "no advantage",
        "lossy translation",
    ])
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("match score")
    ax.set_title(title)

    for i, v in enumerate(values):
        ax.text(min(v + 0.02, 0.98), i, f"{v:.2f}", va="center", fontsize=10)


def make_scenario_figure(
    scenario_name: str,
    result: Dict[str, Any],
    meta: Dict[str, Any],
    save_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    plot_learning_curves_on_ax(ax1, result, title=f"{scenario_name}: learning curves")
    plot_arc_on_ax(ax2, result, title=f"{scenario_name}: ARC")
    plot_regime_scoreboard(ax3, result["regime_scores"], title=f"{scenario_name}: regime meter")

    summary = result["summary"]
    fig.suptitle(
        f"{scenario_name} | class={summary['classification']} | "
        f"lowARC={summary['low_label_arc']:.3f}, "
        f"tailARC={summary['tail_arc']:.3f}, "
        f"finalGap={summary['final_gap']:.3f}",
        fontsize=12,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.03)

    return fig


def make_summary_grid(
    all_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
):
    names = list(all_results.keys())
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(len(names), 3, width_ratios=[1.2, 0.95, 1.0])

    for i, name in enumerate(names):
        res = all_results[name]

        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        ax3 = fig.add_subplot(gs[i, 2])

        plot_learning_curves_on_ax(ax1, res, title=f"{name}: learning curves")
        plot_arc_on_ax(ax2, res, title=f"{name}: ARC")
        plot_regime_scoreboard(ax3, res["regime_scores"], title=f"{name}: regime meter")

    fig.suptitle(
        "TRACE reference scenarios: learning curves, ARC, and regime scoreboards",
        fontsize=14
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.03)

    return fig
