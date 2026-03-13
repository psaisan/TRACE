from pathlib import Path

from translator_toolkit import make_all_linear_panels_and_save


if __name__ == "__main__":
    outdir = Path("figs_linear_smoke")
    make_all_linear_panels_and_save(
        outdir=str(outdir),
        prefix="smoke_",
        seed=7,
        d=150,
        m=6,
        p=4,
        r=20,
        sigma_x=1.0,
        sigma_z=0.7,
        biology_noise=1.0,
        lam_hard=3.0,
        n_pair_grid=(50, 100, 200),
        n_label_grid=(20, 40, 80),
        lam_grid=(0.0, 1.0, 2.0, 3.0),
        corrupt_std_grid=(0.0, 0.5, 1.0),
        phase_n_grid=(20, 40, 80),
        phase_lam_grid=(0.0, 1.5, 3.0),
        ridge_alpha=3.0,
        C_X=0.05,
        C_Z=1.0,
        repeats=3,
        mi_mc_n=2000,
        n_test=1000,
    )
    print(f"Saved linear smoke outputs to {outdir}")
