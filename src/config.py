from dataclasses import dataclass, field
from typing import Literal, Sequence


@dataclass(frozen=True)
class LinearWorldConfig:
    d: int = 1000
    m: int = 12
    p: int = 8
    r: int = 200
    sigma_x: float = 1.0
    sigma_z: float = 0.7
    lam: float = 6.0
    biology_noise: float = 1.0


@dataclass(frozen=True)
class CorePanelConfig:
    seed: int = 7
    world: LinearWorldConfig = field(default_factory=LinearWorldConfig)
    lam_hard: float = 6.0
    n_pair_grid: Sequence[int] = (200, 500, 1000, 3000, 8000)
    n_label_grid: Sequence[int] = (50, 100, 200, 400, 800, 1600)
    lam_grid: Sequence[float] = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)
    corrupt_std_grid: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
    phase_n_grid: Sequence[int] = (50, 100, 200, 400, 800)
    phase_lam_grid: Sequence[float] = (0.0, 2.0, 4.0, 6.0, 8.0)
    ridge_alpha: float = 10.0
    C_X: float = 0.03
    C_Z: float = 1.0
    repeats: int = 6
    mi_mc_n: int = 40000
    n_test: int = 12000


@dataclass(frozen=True)
class ThreatConfig:
    seed: int = 11
    world: LinearWorldConfig = field(default_factory=LinearWorldConfig)
    lam_hard: float = 6.0
    n_pair: int = 3000
    ridge_alpha: float = 10.0
    n_train: int = 800
    n_test: int = 12000
    site_probs: Sequence[float] = (0.5, 0.5)
    pi_train: Sequence[float] = (0.9, 0.1)
    pi_test: Sequence[float] = (0.9, 0.1)
    delta_site_x: float = 6.0
    gamma_site_z: float = 4.0
    halluc_eta: float = 4.0
    C_X: float = 0.03
    C_Z: float = 1.0


@dataclass(frozen=True)
class LearningCurveConfig:
    n_label_grid: Sequence[int]
    repeats: int = 10
    seed: int = 7
    world: LinearWorldConfig = field(default_factory=lambda: LinearWorldConfig(p=16, r=150, lam=8.0, sigma_z=0.4))
    n_pair: int = 12000
    ridge_alpha: float = 1.0
    n_test: int = 12000
    C_X: float = 0.03
    C_Z: float = 1.0


@dataclass(frozen=True)
class NonlinearWorldConfig:
    d: int = 3000
    m: int = 20
    p: int = 20
    r: int = 500
    sigma_x: float = 1.0
    sigma_z: float = 0.50
    lam: float = 7.0
    biology_noise: float = 1.2
    x_nl_strength: float = 1.15
    z_nl_strength: float = 0.45
    x_mix_quad: float = 0.20
    z_mix_quad: float = 0.08
    y_nonlinearity: float = 0.85
    latent_axis_strength: float = 1.0
    label_temperature: float = 1.6


@dataclass(frozen=True)
class NonlinearSweepConfig:
    n_label_grid: Sequence[int]
    repeats: int = 10
    seed: int = 7
    world: NonlinearWorldConfig = field(default_factory=NonlinearWorldConfig)
    n_pair: int = 4000
    ridge_alpha: float = 3.0
    n_test: int = 12000
    C_X: float = 0.02
    C_Z: float = 1.0
    translator: Literal["ridge", "rff"] = "rff"
    translator_n_features: int = 4000
    translator_gamma: float | None = None
