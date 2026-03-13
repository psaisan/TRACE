from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_curve_metrics(curve: dict, k_low: int = 6, k_tail: int = 4) -> dict:
    n = np.asarray(curve["n"])
    aucX = np.asarray(curve["aucX"])
    aucZ = np.asarray(curve["aucZ"])
    meanX = aucX.mean(axis=0)
    meanZ = aucZ.mean(axis=0)
    diff = meanZ - meanX
    k_low_eff = min(k_low, len(n))
    k_tail_eff = min(k_tail, len(n))
    low_n_gain = diff[:k_low_eff].mean()
    low_n_gain_max = diff[:k_low_eff].max()
    low_n_gain_min = diff[:k_low_eff].min()
    largest_n_gap = diff[-1]
    tail_gap_mean = diff[-k_tail_eff:].mean()

    crossover_n = np.nan
    crossover_idx = np.nan
    for j in range(1, len(diff)):
        if diff[j - 1] > 0 and diff[j] <= 0:
            crossover_n = n[j]
            crossover_idx = j
            break

    targetX = meanX[-1] * 0.95
    targetZ = meanZ[-1] * 0.95
    idxX = np.where(meanX >= targetX)[0]
    idxZ = np.where(meanZ >= targetZ)[0]
    n95_X = n[idxX[0]] if len(idxX) > 0 else np.nan
    n95_Z = n[idxZ[0]] if len(idxZ) > 0 else np.nan

    return {
        "n_min": int(n[0]),
        "n_max": int(n[-1]),
        "low_n_gain": float(low_n_gain),
        "low_n_gain_max": float(low_n_gain_max),
        "low_n_gain_min": float(low_n_gain_min),
        "largest_n_gap": float(largest_n_gap),
        "tail_gap_mean": float(tail_gap_mean),
        "crossover_n": crossover_n,
        "crossover_idx": crossover_idx,
        "final_aucX": float(meanX[-1]),
        "final_aucZ": float(meanZ[-1]),
        "n95_X": n95_X,
        "n95_Z": n95_Z,
    }


def summarize_many_curves(curves: dict[str, dict], k_low: int = 6, k_tail: int = 4) -> pd.DataFrame:
    rows = []
    for name, curve in curves.items():
        met = summarize_curve_metrics(curve, k_low=k_low, k_tail=k_tail)
        met["condition"] = name
        rows.append(met)
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df[
        [
            "condition",
            "n_min",
            "n_max",
            "low_n_gain",
            "low_n_gain_max",
            "low_n_gain_min",
            "largest_n_gap",
            "tail_gap_mean",
            "crossover_n",
            "final_aucZ",
            "final_aucX",
            "n95_Z",
            "n95_X",
        ]
    ]
    summary_df = summary_df.sort_values("low_n_gain", ascending=False).reset_index(drop=True)
    return summary_df
