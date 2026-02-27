"""
Step 2 – Build population matrices for spectral analysis.

Inputs:
  <data_dir>/latents.npy  – (T, D) float32

Outputs (always):
  <output>/raw.npy          – (T,   D)  raw latents
  <output>/diff.npy         – (T-1, D)  z_t - z_{t-1}
  <output>/cond.npy         – (T-1, D)  z_t - Ridge(z_{t-1})
  <output>/predictor.pkl    – SVD + Ridge objects

Outputs (per GOP size K):
  <output>/gop_diff_{K}.npy  – (T, D)  raw_diff within each GOP (keyframe → inter)
  <output>/gop_cond_{K}.npy  – (T, D)  cond_residual within each GOP
"""

import argparse
import math
import os
import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge


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

    Returns:
        residuals  – (T-1, D) float32
        predictor  – dict with keys 'svd', 'ridge' for later reuse
    """
    X = latents[:-1]   # (T-1, D)
    Y = latents[1:]    # (T-1, D)

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


def build_gop_populations(latents: np.ndarray,
                           predictor: dict,
                           gop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the two residual populations for a GOP of size K.

    Within each GOP the first frame is the keyframe (stored verbatim).
    The remaining (K-1) inter-frames are compared against:
      - gop_diff: z_t - z_{keyframe}  (raw offset from keyframe)
      - gop_cond: z_t - chain_predict(z_{keyframe}, t-gop_start steps)

    Both arrays have shape (N_inter, D) where N_inter = number of inter-frames
    across all GOPs in the sequence.

    Returns:
        gop_diff – (N_inter, D) float32
        gop_cond – (N_inter, D) float32
    """
    T, D = latents.shape
    svd   = predictor["svd"]
    ridge = predictor["ridge"]

    diff_rows = []
    cond_rows = []

    for gop_start in range(0, T, gop_size):
        kf = latents[gop_start]          # (D,)  keyframe

        # chain-predict from the keyframe for each step in the GOP
        prev = kf
        for offset in range(1, gop_size):
            t = gop_start + offset
            if t >= T:
                break
            gt = latents[t]              # (D,)

            # GOP diff: residual from the keyframe
            diff_rows.append(gt - kf)

            # GOP cond: residual from chain-predicted latent
            prev_low = svd.transform(prev[np.newaxis])   # (1, pca_dims)
            pred = ridge.predict(prev_low)[0]            # (D,)
            cond_rows.append(gt - pred.astype(np.float32))
            prev = pred

    gop_diff = np.stack(diff_rows, axis=0).astype(np.float32)
    gop_cond = np.stack(cond_rows, axis=0).astype(np.float32)
    return gop_diff, gop_cond


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def build_populations(latents_path: str, output_dir: str,
                      ridge_alpha: float = 1.0,
                      pca_dims: int = 256,
                      gop_sizes: list[int] | None = None) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    gop_sizes = gop_sizes or []

    latents = np.load(latents_path).astype(np.float32)  # (T, D)
    T, D = latents.shape
    print(f"[build_populations] loaded latents {latents.shape}")

    # Population A – Raw
    raw_path = os.path.join(output_dir, "raw.npy")
    np.save(raw_path, latents)
    print(f"[build_populations] raw    {latents.shape} → {raw_path}")

    # Population B – Linear Difference
    diff = build_diff(latents)
    diff_path = os.path.join(output_dir, "diff.npy")
    np.save(diff_path, diff)
    print(f"[build_populations] diff   {diff.shape} → {diff_path}")

    # Population C – Conditional Residual  +  fit predictor
    cond, predictor = build_conditional_residual(latents, alpha=ridge_alpha,
                                                 pca_dims=pca_dims)
    cond_path = os.path.join(output_dir, "cond.npy")
    np.save(cond_path, cond)
    print(f"[build_populations] cond   {cond.shape} → {cond_path}")

    # Save predictor for later evaluation stages
    predictor_path = os.path.join(output_dir, "predictor.pkl")
    with open(predictor_path, "wb") as f:
        pickle.dump(predictor, f)
    print(f"[build_populations] predictor → {predictor_path}")

    paths = {"raw": raw_path, "diff": diff_path, "cond": cond_path,
             "predictor": predictor_path}

    # Populations D/E – GOP residuals for each requested GOP size
    for K in gop_sizes:
        if K < 2:
            continue   # GOP=1 means all keyframes → identical to raw, skip
        n_keyframes = math.ceil(T / K)
        n_inter     = T - n_keyframes
        pct_saved   = 100.0 * n_inter / T
        print(f"\n[build_populations] GOP={K}  "
              f"keyframes={n_keyframes}/{T}  "
              f"inter={n_inter}  "
              f"memory saving={pct_saved:.1f}%")

        gop_diff, gop_cond = build_gop_populations(latents, predictor, K)

        gdiff_path = os.path.join(output_dir, f"gop_diff_{K}.npy")
        gcond_path = os.path.join(output_dir, f"gop_cond_{K}.npy")
        np.save(gdiff_path, gop_diff)
        np.save(gcond_path, gop_cond)
        print(f"[build_populations] gop_diff_{K} {gop_diff.shape} → {gdiff_path}")
        print(f"[build_populations] gop_cond_{K} {gop_cond.shape} → {gcond_path}")

        paths[f"gop_diff_{K}"] = gdiff_path
        paths[f"gop_cond_{K}"] = gcond_path

    return paths


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
    p.add_argument("--gop_sizes", default="",
                   help="Comma-separated GOP sizes, e.g. '3,8,10' (optional).")
    p.add_argument("--output", default="results/",
                   help="Output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gops = [int(x) for x in args.gop_sizes.split(",") if x.strip()] if args.gop_sizes else []
    build_populations(
        latents_path=args.latents,
        output_dir=args.output,
        ridge_alpha=args.ridge_alpha,
        pca_dims=args.pca_dims,
        gop_sizes=gops,
    )
