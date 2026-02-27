"""
Step 4 – UMAP 2-D scatter plot.

Trains UMAP on raw latents and projects all populations into the same 2-D space.
Auto-discovers GOP populations in the data directory.

Inputs:
  <data_dir>/raw.npy
  <data_dir>/cond.npy
  <data_dir>/gop_cond_{K}.npy   (optional, one per GOP size)

Outputs:
  <output>/umap_scatter.png
"""

import argparse
import os
import re
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def subsample(M: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Subsample preserving temporal order. Returns (array, original_indices)."""
    T = M.shape[0]
    if T <= max_points:
        return M, np.arange(T)
    idx = np.linspace(0, T - 1, max_points, dtype=int)
    return M[idx], idx


def pca_reduce(M: np.ndarray, dims: int = 50) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    M_c = M - M.mean(axis=0, keepdims=True)
    k = min(dims, M.shape[1] - 1, M.shape[0] - 1)
    return TruncatedSVD(n_components=k, random_state=0).fit_transform(M_c).astype(np.float32)


def size_alpha(t: np.ndarray):
    """Temporal ramps: early → small & faint, late → large & opaque."""
    return 4 + 18 * t, 0.2 + 0.7 * t


def _discover_gop_cond(data_dir: str) -> list[tuple[int, str]]:
    """Return [(K, path), …] for all gop_cond_K.npy files, sorted by K."""
    found = []
    for fname in os.listdir(data_dir):
        m = re.match(r"^gop_cond_(\d+)\.npy$", fname)
        if m:
            found.append((int(m.group(1)), os.path.join(data_dir, fname)))
    return sorted(found)


# Colour maps per population type
_CMAPS = {
    "raw":  "Blues",
    "cond": "Reds",
}
_GOP_CMAPS = ["Purples", "Oranges", "Greens", "YlOrBr", "PuRd"]


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def run_umap(data_dir: str, output_dir: str,
             n_neighbors: int = 15, min_dist: float = 0.1,
             max_points: int = 2000, pca_dims: int = 50):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load populations ──────────────────────────────────────────────────
    raw  = np.load(os.path.join(data_dir, "raw.npy")).astype(np.float32)
    cond = np.load(os.path.join(data_dir, "cond.npy")).astype(np.float32)
    gop_entries = _discover_gop_cond(data_dir)   # [(K, path), …]

    # Align lengths (cond has T-1 rows; align everything to shortest)
    min_len = min(raw.shape[0], cond.shape[0])
    raw  = raw[:min_len]
    cond = cond[:min_len]

    # Build population list: (name, array, cmap)
    pops = [
        ("Raw Latents (Z)",           raw,  "Blues"),
        ("Conditional Residual (r)",  cond, "Reds"),
    ]
    for i, (K, path) in enumerate(gop_entries):
        arr = np.load(path).astype(np.float32)
        arr = arr[:min_len]          # keep same temporal window
        cmap = _GOP_CMAPS[i % len(_GOP_CMAPS)]
        pops.append((f"GOP-{K} Residual", arr, cmap))

    # ── Subsample + PCA ──────────────────────────────────────────────────
    print(f"[umap] PCA pre-reduction to {pca_dims} dims …")
    subsampled = []
    for name, arr, cmap in pops:
        sub, idx = subsample(arr, max_points)
        pca      = pca_reduce(sub, dims=pca_dims)
        t_norm   = idx / (min_len - 1) if min_len > 1 else np.zeros(len(idx))
        sz, al   = size_alpha(t_norm)
        subsampled.append((name, pca, t_norm, sz, al, cmap))

    # ── Fit UMAP on raw, transform all ───────────────────────────────────
    print(f"[umap] fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}) …")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42, verbose=False)
    reducer.fit(subsampled[0][1])    # fit on raw PCA

    projections = [(name, reducer.transform(pca), t, sz, al, cmap)
                   for name, pca, t, sz, al, cmap in subsampled]

    # ── Plot: one subplot per population ─────────────────────────────────
    n_pops = len(projections)
    ncols  = min(n_pops, 3)
    nrows  = (n_pops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 5 * nrows),
                              gridspec_kw={"wspace": 0.35, "hspace": 0.45})
    axes_flat = np.array(axes).flatten() if n_pops > 1 else [axes]

    for ax, (name, xy, t, size, alpha, cmap) in zip(axes_flat, projections):
        sc = ax.scatter(xy[:, 0], xy[:, 1],
                        c=t, cmap=cmap, s=size, alpha=alpha,
                        linewidths=0, vmin=0, vmax=1)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("time  (0=start → 1=end)", fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["start", "mid", "end"])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("UMAP-1", fontsize=10)
        ax.set_ylabel("UMAP-2", fontsize=10)
        ax.grid(True, ls="--", alpha=0.3)

    # Hide any spare axes
    for ax in axes_flat[n_pops:]:
        ax.set_visible(False)

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
    p = argparse.ArgumentParser(description="UMAP scatter of latent populations.")
    p.add_argument("--data_dir",  default="results/",
                   help="Directory with raw.npy, cond.npy (+ gop_cond_K.npy).")
    p.add_argument("--n_neighbors", type=int,   default=15)
    p.add_argument("--min_dist",    type=float, default=0.1)
    p.add_argument("--max_points",  type=int,   default=2000,
                   help="Max points per population for UMAP.")
    p.add_argument("--pca_dims",    type=int,   default=50,
                   help="PCA pre-reduction dimensionality.")
    p.add_argument("--output", default="results/", help="Output directory.")
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
