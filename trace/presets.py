from typing import Dict, Any


def regime_defaults(regime: str) -> Dict[str, Any]:
    base = dict(
        d=1000,
        m=12,
        p=12,
        r=200,
        sigma_x=1.0,
        sigma_z=0.4,
        lam=8.0,
        biology_noise=1.0,
        T_mode="identity_like",
        T_jitter=0.05,
        n_pair=12000,
        ridge_alpha=1.0,
        C_X=0.03,
        C_H=1.0,
    )

    presets = {
        "sample_efficiency": dict(
            d=700,
            p=12,
            r=120,
            lam=5.5,
            sigma_z=0.35,
            T_mode="identity_like",
            T_jitter=0.04,
            n_pair=12000,
            ridge_alpha=1.0,
            C_X=0.06,
            C_H=1.0,
        ),
        "persistent_advantage": dict(
            d=2600,
            p=8,
            r=650,
            lam=18.0,
            sigma_z=0.22,
            T_mode="identity_like",
            T_jitter=0.02,
            n_pair=30000,
            ridge_alpha=0.5,
            C_X=0.008,
            C_H=1.6,
        ),
        "no_advantage": dict(
            d=250,
            p=12,
            r=40,
            lam=0.3,
            sigma_z=0.45,
            T_mode="identity_like",
            T_jitter=0.03,
            n_pair=4000,
            ridge_alpha=1.2,
            C_X=0.12,
            C_H=0.9,
        ),
        "lossy_translation": dict(
            d=1000,
            p=8,
            r=200,
            lam=5.0,
            sigma_z=1.2,
            T_mode="random",
            T_jitter=0.2,
            n_pair=1500,
            ridge_alpha=2.0,
            C_X=0.04,
            C_H=0.8,
        ),
    }

    if regime not in presets:
        raise ValueError(f"Unknown regime '{regime}'")

    out = dict(base)
    out.update(presets[regime])
    return out
