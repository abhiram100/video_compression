"""
Step 2 – Build the three population matrices.

Inputs:
  <data_dir>/latents.npy  – (T, D) float32

Outputs:
  <output>/raw.npy   – (T,   D)  raw latents
  <output>/diff.npy  – (T-1, D)  z_t - z_{t-1}
  <output>/cond.npy  – (T-1, D)  z_t - Ridge(z_{t-1})
"""

import argparse
import os
import pickle

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def build_diff(latents: np.ndarray) -> np.ndarray:
    """Simple temporal difference: z_t - z_{t-1}."""
    return latents[1:] - latents[:-1]          # (T-1, D)


def build_conditional_residual(latents: np.ndarray,
                                alpha: float = 1.0,
                                pca_dims: int = 256) -> tuple[np.ndarray, dict]:
    """
    Fit Ridge(z_{t-1} -> z_t) and return residuals z_t - W(z_{t-1}).

    Because D is large (~16 k) we first project to `pca_dims` principal
    components, fit Ridge in that space, then project back.  This keeps
    the fit tractable while still being a linear predictor.

    Returns:
        residuals  – (T-1, D) float32
        predictor  – dict with keys 'svd', 'ridge', 'latent_shape' for later reuse
    """
    X = latents[:-1]   # (T-1, D)  predictors
    Y = latents[1:]    # (T-1, D)  targets

    T = X.shape[0]
    actual_dims = min(pca_dims, T - 1, X.shape[1])

    print(f"[build_populations] fitting PCA({actual_dims}) for Ridge predictor …")
    svd = TruncatedSVD(n_components=actual_dims, random_state=0)
    X_low = svd.fit_transform(X)               # (T-1, pca_dims)

    print(f"[build_populations] fitting Ridge(alpha={alpha}) …")
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_low, Y)

    Y_hat = ridge.predict(X_low)               # (T-1, D)
    residuals = Y - Y_hat                      # (T-1, D)

    predictor = {"svd": svd, "ridge": ridge}
    return residuals.astype(np.float32), predictor


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def build_populations(latents_path: str, output_dir: str,
                      ridge_alpha: float = 1.0,
                      pca_dims: int = 256) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    latents = np.load(latents_path).astype(np.float32)  # (T, D)
    print(f"[build_populations] loaded latents {latents.shape}")

    # Population A – Raw
    raw = latents
    raw_path = os.path.join(output_dir, "raw.npy")
    np.save(raw_path, raw)
    print(f"[build_populations] raw    {raw.shape} → {raw_path}")

    # Population B – Linear Difference
    diff = build_diff(latents)
    diff_path = os.path.join(output_dir, "diff.npy")
    np.save(diff_path, diff)
    print(f"[build_populations] diff   {diff.shape} → {diff_path}")

    # Population C – Conditional Residual
    cond, predictor = build_conditional_residual(latents, alpha=ridge_alpha,
                                                 pca_dims=pca_dims)
    cond_path = os.path.join(output_dir, "cond.npy")
    np.save(cond_path, cond)
    print(f"[build_populations] cond   {cond.shape} → {cond_path}")

    # Save predictor (SVD + Ridge) for later evaluation
    predictor_path = os.path.join(output_dir, "predictor.pkl")
    with open(predictor_path, "wb") as f:
        pickle.dump(predictor, f)
    print(f"[build_populations] predictor → {predictor_path}")

    return {"raw": raw_path, "diff": diff_path, "cond": cond_path,
            "predictor": predictor_path}


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Raw / Diff / Conditional-Residual population matrices.")
    p.add_argument("--latents", default="results/latents.npy",
                   help="Path to latents.npy produced by extract_latents.py.")
    p.add_argument("--ridge_alpha", type=float, default=1.0,
                   help="Regularisation strength for Ridge predictor.")
    p.add_argument("--pca_dims", type=int, default=256,
                   help="PCA dims used before fitting Ridge (speed vs accuracy).")
    p.add_argument("--output", default="results/",
                   help="Output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_populations(
        latents_path=args.latents,
        output_dir=args.output,
        ridge_alpha=args.ridge_alpha,
        pca_dims=args.pca_dims,
    )
