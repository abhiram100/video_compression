"""
Step 3 – Spectral analysis: SVD, ESD, Effective Rank (Participation Ratio),
          Cumulative Variance, and Scree Plot.

Inputs:
  <data_dir>/raw.npy
  <data_dir>/diff.npy
  <data_dir>/cond.npy
  <data_dir>/gop_diff_{K}.npy   (optional, one per GOP size)
  <data_dir>/gop_cond_{K}.npy   (optional, one per GOP size)

Outputs:
  <output>/scree_plot.png
  <output>/rank_table.txt   (also printed to stdout)
"""

import argparse
import os
import re

import matplotlib
import numpy as np

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


def _discover_gop_populations(data_dir: str) -> list[tuple[str, str]]:
    """
    Scan data_dir for gop_diff_K.npy / gop_cond_K.npy files.
    Returns list of (name, path) sorted by K.
    """
    found = {}
    for fname in os.listdir(data_dir):
        m = re.match(r"^gop_(diff|cond)_(\d+)\.npy$", fname)
        if m:
            kind, K = m.group(1), int(m.group(2))
            found.setdefault(K, {})[kind] = os.path.join(data_dir, fname)

    entries = []
    for K in sorted(found):
        for kind in ("diff", "cond"):
            if kind in found[K]:
                entries.append((f"gop_{kind}_{K}", found[K][kind]))
    return entries


# ──────────────────────────────────────────────
# plotting
# ──────────────────────────────────────────────

# Colour / style catalogue: base populations get solid lines, GOP gets dashed
_BASE_COLOURS = {
    "raw":  "#1f77b4",
    "diff": "#ff7f0e",
    "cond": "#2ca02c",
}
_BASE_LABELS = {
    "raw":  "Raw Latents (Z)",
    "diff": "Linear Residual (Δz)",
    "cond": "Conditional Residual (r)",
}
# Colour pairs for each GOP index
_GOP_DIFF_COLOURS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
_GOP_COND_COLOURS = ["#d62728", "#17becf", "#aec7e8", "#ffbb78", "#98df8a"]


def _result_style(name: str, gop_diff_idx: dict, gop_cond_idx: dict):
    """Return (color, linestyle, label) for a named population."""
    if name in _BASE_COLOURS:
        return _BASE_COLOURS[name], "-", _BASE_LABELS[name]

    m = re.match(r"^gop_(diff|cond)_(\d+)$", name)
    if m:
        kind, K = m.group(1), int(m.group(2))
        if kind == "diff":
            idx = gop_diff_idx.get(K, 0)
            col = _GOP_DIFF_COLOURS[idx % len(_GOP_DIFF_COLOURS)]
            return col, "--", f"GOP-{K} Δz (keyframe→inter)"
        else:
            idx = gop_cond_idx.get(K, 0)
            col = _GOP_COND_COLOURS[idx % len(_GOP_COND_COLOURS)]
            return col, ":", f"GOP-{K} residual (predicted→inter)"
    return None, "-", name


def plot_scree(results: list[dict], output_path: str):
    """Log-scale normalised eigenvalue scree plot for all populations."""
    # Build index maps so each GOP size gets a consistent colour
    gop_ks = sorted({int(re.match(r"gop_(?:diff|cond)_(\d+)", r["name"]).group(1))
                     for r in results
                     if re.match(r"gop_(?:diff|cond)_(\d+)", r["name"])})
    gop_diff_idx = {K: i for i, K in enumerate(gop_ks)}
    gop_cond_idx = {K: i for i, K in enumerate(gop_ks)}

    # Two axes: left = base populations, right = GOP populations
    has_gop = any(r["name"].startswith("gop_") for r in results)
    if has_gop:
        fig, (ax_base, ax_gop) = plt.subplots(1, 2, figsize=(16, 5),
                                               sharey=True,
                                               gridspec_kw={"wspace": 0.08})
    else:
        fig, ax_base = plt.subplots(figsize=(9, 5))
        ax_gop = None

    for r in results:
        k     = r["n_dims"]
        color, ls, label = _result_style(r["name"], gop_diff_idx, gop_cond_idx)
        ax    = ax_gop if (ax_gop and r["name"].startswith("gop_")) else ax_base
        ax.plot(range(1, k + 1), r["norm_eigs"],
                label=label, color=color, linestyle=ls, linewidth=1.6)

    for ax, title in [(ax_base, "Base populations"),
                      (ax_gop,  "GOP residual populations (dashed=Δz, dotted=cond)")]:
        if ax is None:
            continue
        ax.set_yscale("log")
        ax.set_xlabel("Component index", fontsize=11)
        ax.set_ylabel("Normalised eigenvalue  λ̄_i", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", ls="--", alpha=0.35)

    fig.suptitle("Eigen Spectral Density – Scree Plot", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[spectral] scree plot saved → {output_path}")


# ──────────────────────────────────────────────
# report
# ──────────────────────────────────────────────

def print_rank_table(results: list[dict], save_path: str):
    # Gather base and GOP results
    base_order = ["raw", "diff", "cond"]
    base  = [r for r in results if r["name"] in base_order]
    base  = sorted(base, key=lambda r: base_order.index(r["name"]))
    gops  = [r for r in results if r["name"].startswith("gop_")]

    col_w = 18  # column width

    def row(label, key):
        cells = f"{label:<28}"
        for r in base:
            v = r[key]
            cells += f"{v:{col_w}.1f}" if isinstance(v, float) else f"{v:{col_w}d}"
        return cells

    header = f"{'Metric':<28}" + "".join(
        f"{r['name']:>{col_w}}" for r in base
    )
    sep = "-" * len(header)
    rows = [sep, header, sep,
            row("Effective Rank (PR)", "pr"),
            row("Dims for 95% Var",   "dims_95"),
            row("Dims for 99% Var",   "dims_99"),
            sep]

    # Conclusion for base populations
    raw_r  = next((r for r in base if r["name"] == "raw"),  None)
    cond_r = next((r for r in base if r["name"] == "cond"), None)
    if raw_r and cond_r:
        if cond_r["pr"] < raw_r["pr"]:
            verdict = (f"✅  CONFIRMED:  PR_cond ({cond_r['pr']:.1f}) "
                       f"< PR_raw ({raw_r['pr']:.1f})")
        else:
            verdict = (f"❌  NOT CONFIRMED:  PR_cond ({cond_r['pr']:.1f}) "
                       f">= PR_raw ({raw_r['pr']:.1f})")
        rows += ["", "Hypothesis (base):", verdict, ""]

    # GOP table
    if gops:
        rows += ["", "GOP Residual Populations:", sep]
        gop_header = (f"{'Population':<24}  {'PR':>8}  {'dims@95%':>10}  "
                      f"{'dims@99%':>10}  {'vs raw PR':>12}")
        rows += [gop_header, "-" * len(gop_header)]
        raw_pr = raw_r["pr"] if raw_r else float("nan")
        for r in sorted(gops, key=lambda x: x["name"]):
            delta = r["pr"] - raw_pr
            sign  = "↓" if delta < 0 else "↑"
            rows.append(
                f"{r['name']:<24}  {r['pr']:>8.1f}  {r['dims_95']:>10d}  "
                f"{r['dims_99']:>10d}  "
                f"{sign}{abs(delta):>9.1f} ({100*delta/raw_pr:+.1f}%)"
            )
        rows.append(sep)

    table = "\n".join(rows)
    print("\n" + table)
    with open(save_path, "w") as f:
        f.write(table + "\n")
    print(f"[spectral] rank table saved → {save_path}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def run_spectral_analysis(data_dir: str, output_dir: str,
                          n_components: int = None):
    os.makedirs(output_dir, exist_ok=True)

    # Always load base populations
    populations = [
        ("raw",  os.path.join(data_dir, "raw.npy")),
        ("diff", os.path.join(data_dir, "diff.npy")),
        ("cond", os.path.join(data_dir, "cond.npy")),
    ]

    # Auto-discover any GOP populations written by build_populations
    populations += _discover_gop_populations(data_dir)

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
                   help="Directory containing raw.npy / diff.npy / cond.npy "
                        "(+ any gop_diff_K.npy / gop_cond_K.npy).")
    p.add_argument("--n_components", type=int, default=None,
                   help="Max SVD components (default: min(T,D)-1).")
    p.add_argument("--output", default="results/", help="Output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_spectral_analysis(
        data_dir=args.data_dir,
        output_dir=args.output,
        n_components=args.n_components,
    )
