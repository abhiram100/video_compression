"""
Step 3 – Spectral analysis: SVD, ESD, Effective Rank (Participation Ratio),
          Cumulative Variance, and Scree Plot.

Inputs:
  <data_dir>/raw.npy
  <data_dir>/diff.npy
  <data_dir>/cond.npy

Outputs:
  <output>/scree_plot.png
  <output>/rank_table.txt   (also printed to stdout)
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def compute_svd_spectrum(M: np.ndarray, n_components: int = None) -> np.ndarray:
    """
    Return normalised eigenvalues λ̄_i = λ_i / Σλ  (descending order).

    Uses TruncatedSVD which is efficient for large matrices.
    We cap n_components at min(T, D) - 1.
    """
    T, D = M.shape
    max_k = min(T, D) - 1
    k = min(n_components or max_k, max_k)

    # Centre the matrix (removes mean shift contribution)
    M_c = M - M.mean(axis=0, keepdims=True)

    svd = TruncatedSVD(n_components=k, random_state=0)
    svd.fit(M_c)

    singular_values = svd.singular_values_         # σ_i
    eigenvalues = singular_values ** 2             # λ_i = σ_i²
    norm_eigenvalues = eigenvalues / eigenvalues.sum()
    return norm_eigenvalues                        # λ̄_i, shape (k,)


def participation_ratio(norm_eigs: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ²  on normalised eigenvalues where Σλ=1  →  PR = 1/Σλ²."""
    return float(1.0 / np.sum(norm_eigs ** 2))


def dims_for_variance(norm_eigs: np.ndarray, threshold: float = 0.95) -> int:
    """Number of leading dimensions required to explain `threshold` of variance."""
    cumvar = np.cumsum(norm_eigs)
    indices = np.where(cumvar >= threshold)[0]
    if len(indices) == 0:
        return len(norm_eigs)   # all dims used, still under threshold
    return int(indices[0]) + 1


def analyse_population(M: np.ndarray, name: str,
                       n_components: int = None) -> dict:
    print(f"[spectral] analysing '{name}'  shape={M.shape} …")
    norm_eigs = compute_svd_spectrum(M, n_components=n_components)
    pr = participation_ratio(norm_eigs)
    d95 = dims_for_variance(norm_eigs, 0.95)
    d99 = dims_for_variance(norm_eigs, 0.99)
    print(f"  PR={pr:.1f}   dims@95%={d95}   dims@99%={d99}")
    return {
        "name": name,
        "norm_eigs": norm_eigs,
        "pr": pr,
        "dims_95": d95,
        "dims_99": d99,
        "n_dims": len(norm_eigs),
    }


# ──────────────────────────────────────────────
# plotting
# ──────────────────────────────────────────────

def plot_scree(results: list[dict], output_path: str):
    """Log-scale normalised eigenvalue scree plot for all populations."""
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {"raw": "#1f77b4", "diff": "#ff7f0e", "cond": "#2ca02c"}
    labels = {"raw": "Raw Latents (Z)",
              "diff": "Linear Residual (Δz)",
              "cond": "Conditional Residual (r)"}

    for r in results:
        k = r["n_dims"]
        name = r["name"]
        color = colors.get(name, None)
        label = labels.get(name, name)
        ax.plot(range(1, k + 1), r["norm_eigs"],
                label=label, color=color, linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel("Component index", fontsize=12)
    ax.set_ylabel("Normalised eigenvalue  λ̄_i", fontsize=12)
    ax.set_title("Eigen Spectral Density (ESD) – Scree Plot", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[spectral] scree plot saved → {output_path}")


# ──────────────────────────────────────────────
# report
# ──────────────────────────────────────────────

def print_rank_table(results: list[dict], save_path: str):
    header = f"{'Metric':<28} {'Raw Latents':>16} {'Linear Residual':>16} {'Conditional Residual':>22}"
    sep = "-" * len(header)

    row_pr   = f"{'Effective Rank (PR)':<28}"
    row_d95  = f"{'Dims for 95% Var':<28}"
    row_d99  = f"{'Dims for 99% Var':<28}"

    for r in results:
        val_pr  = f"{r['pr']:>16.1f}"
        val_d95 = f"{r['dims_95']:>16d}"
        val_d99 = f"{r['dims_99']:>16d}"

        if r["name"] == "raw":
            row_pr  += val_pr
            row_d95 += val_d95
            row_d99 += val_d99
        elif r["name"] == "diff":
            row_pr  += val_pr
            row_d95 += val_d95
            row_d99 += val_d99
        elif r["name"] == "cond":
            row_pr  += f"{r['pr']:>22.1f}"
            row_d95 += f"{r['dims_95']:>22d}"
            row_d99 += f"{r['dims_99']:>22d}"

    # Conclusion
    raw_res  = next(r for r in results if r["name"] == "raw")
    cond_res = next(r for r in results if r["name"] == "cond")

    if cond_res["pr"] < raw_res["pr"]:
        verdict = (f"✅  HYPOTHESIS CONFIRMED:  PR_cond ({cond_res['pr']:.1f}) "
                   f"< PR_raw ({raw_res['pr']:.1f})")
    else:
        verdict = (f"❌  HYPOTHESIS NOT CONFIRMED:  PR_cond ({cond_res['pr']:.1f}) "
                   f">= PR_raw ({raw_res['pr']:.1f})")

    table = "\n".join([sep, header, sep, row_pr, row_d95, row_d99, sep,
                       "", "Conclusion:", verdict, ""])

    print(table)
    with open(save_path, "w") as f:
        f.write(table)
    print(f"[spectral] rank table saved → {save_path}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def run_spectral_analysis(data_dir: str, output_dir: str,
                          n_components: int = None):
    os.makedirs(output_dir, exist_ok=True)

    populations = [
        ("raw",  os.path.join(data_dir, "raw.npy")),
        ("diff", os.path.join(data_dir, "diff.npy")),
        ("cond", os.path.join(data_dir, "cond.npy")),
    ]

    results = []
    for name, path in populations:
        M = np.load(path).astype(np.float32)
        r = analyse_population(M, name, n_components=n_components)
        results.append(r)

    plot_scree(results, os.path.join(output_dir, "scree_plot.png"))
    print_rank_table(results, os.path.join(output_dir, "rank_table.txt"))
    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Spectral analysis of latent populations.")
    p.add_argument("--data_dir", default="results/",
                   help="Directory containing raw.npy / diff.npy / cond.npy.")
    p.add_argument("--n_components", type=int, default=None,
                   help="Max SVD components (default: min(T,D)-1).")
    p.add_argument("--output", default="results/",
                   help="Output directory for plots and table.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_spectral_analysis(
        data_dir=args.data_dir,
        output_dir=args.output,
        n_components=args.n_components,
    )
