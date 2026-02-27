"""
Step 4 – UMAP 2-D scatter plot.

Trains UMAP on raw latents and transforms both raw and conditional residuals
into the same 2-D embedding space.

Inputs:
  <data_dir>/raw.npy
  <data_dir>/cond.npy

Outputs:
  <output>/umap_scatter.png
"""

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def subsample(M: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample to at most max_points rows while preserving temporal order.

    Returns (subsampled_matrix, original_indices) so callers can map points
    back to their position in time.
    """
    T = M.shape[0]
    if T <= max_points:
        return M, np.arange(T)
    # Evenly-spaced indices to keep temporal spread
    idx = np.linspace(0, T - 1, max_points, dtype=int)
    return M[idx], idx


def pca_reduce(M: np.ndarray, dims: int = 50) -> np.ndarray:
    """Quick PCA pre-reduction before UMAP (speeds up n_neighbors graph)."""
    from sklearn.decomposition import TruncatedSVD
    M_c = M - M.mean(axis=0, keepdims=True)
    svd = TruncatedSVD(n_components=min(dims, M.shape[1] - 1, M.shape[0] - 1),
                       random_state=0)
    return svd.fit_transform(M_c).astype(np.float32)


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def run_umap(data_dir: str, output_dir: str,
             n_neighbors: int = 15, min_dist: float = 0.1,
             max_points: int = 2000, pca_dims: int = 50):
    os.makedirs(output_dir, exist_ok=True)

    raw  = np.load(os.path.join(data_dir, "raw.npy")).astype(np.float32)
    cond = np.load(os.path.join(data_dir, "cond.npy")).astype(np.float32)

    # Align lengths (cond has T-1 rows, raw has T rows)
    min_len = min(raw.shape[0], cond.shape[0])
    raw  = raw[:min_len]
    cond = cond[:min_len]
    T = min_len

    # Subsample while keeping temporal order
    raw_sub,  raw_idx  = subsample(raw,  max_points)
    cond_sub, cond_idx = subsample(cond, max_points)

    print(f"[umap] raw_sub={raw_sub.shape}  cond_sub={cond_sub.shape}")

    # Normalised temporal position in [0, 1] for each kept point
    raw_t  = raw_idx  / (T - 1)   # 0 = first frame, 1 = last frame
    cond_t = cond_idx / (T - 1)

    # Size and alpha ramps:  early frames → small & faint, late → large & opaque
    def size_alpha(t: np.ndarray):
        size  = 4  + 18 * t        # 4 → 22 px²
        alpha = 0.2 + 0.7 * t      # 0.2 → 0.9
        return size, alpha

    raw_size,  raw_alpha  = size_alpha(raw_t)
    cond_size, cond_alpha = size_alpha(cond_t)

    # PCA pre-reduce
    print(f"[umap] PCA pre-reduction to {pca_dims} dims …")
    raw_pca  = pca_reduce(raw_sub,  dims=pca_dims)
    cond_pca = pca_reduce(cond_sub, dims=pca_dims)

    # Fit UMAP on raw, transform both
    print(f"[umap] fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}) …")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42, verbose=False)
    reducer.fit(raw_pca)

    raw_2d  = reducer.transform(raw_pca)
    cond_2d = reducer.transform(cond_pca)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"wspace": 0.35})

    for ax, xy, t, size, alpha, cmap, title in [
        (axes[0], raw_2d,  raw_t,  raw_size,  raw_alpha,
         "Blues",  "Raw Latents (Z)"),
        (axes[1], cond_2d, cond_t, cond_size, cond_alpha,
         "Reds",   "Conditional Residuals (r)"),
    ]:
        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=t,               # colour encodes time
            cmap=cmap,
            s=size,            # size   encodes time
            alpha=alpha,       # opacity encodes time
            linewidths=0,
            vmin=0, vmax=1,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Temporal position  (0=start → 1=end)", fontsize=9)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["start", "mid", "end"])

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP-1", fontsize=11)
        ax.set_ylabel("UMAP-2", fontsize=11)
        ax.grid(True, ls="--", alpha=0.3)

    fig.suptitle(
        "UMAP 2-D Projection  |  colour + size + opacity → temporal progression",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "umap_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[umap] scatter plot saved → {out_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UMAP scatter: raw vs conditional residuals.")
    p.add_argument("--data_dir", default="results/",
                   help="Directory containing raw.npy and cond.npy.")
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--max_points", type=int, default=2000,
                   help="Max points per population (subsampled for speed).")
    p.add_argument("--pca_dims", type=int, default=50,
                   help="PCA pre-reduction dimensions before UMAP.")
    p.add_argument("--output", default="results/",
                   help="Output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_umap(
        data_dir=args.data_dir,
        output_dir=args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        max_points=args.max_points,
        pca_dims=args.pca_dims,
    )
