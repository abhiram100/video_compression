"""
Stride Sensitivity Study
========================
Fixed GOP size of 8.  Extract exactly 8 consecutive latents from the video
at different temporal strides S in {1, 2, 4, 8, 16}.

  stride=S  means we read video frames  0, S, 2S, 3S, … (7S)

So the 8 latents always span a GOP of exactly 8 pictures, but those pictures
are S real video frames apart.

  • Frame 0 is the keyframe (stored verbatim).
  • Frames 1-7 are reconstructed by chaining the Ridge predictor.

PCA dimension sweep
-------------------
The Ridge predictor uses PCA(pca_dims) as a pre-processing step.  This study
also sweeps pca_dims so you can see quality vs model capacity.  A predictor is
fit separately for each pca_dims value using a training latent file.

For each (stride, pca_dims) pair we compute:
  - Per-frame SSIM and PSNR  (pixel space, after VAE decode)
  - Latent-space L2 and cosine distance  (ẑ_t vs z_t, no decode needed)
  - FID over the 7 inter-frames
  - Qualitative panels: Keyframe | GT | Predicted  for every inter-frame

Outputs (all inside <output>/stride_study/):
  stride_S_pca_P/frame_N.png      – 7 qualitative panels per (stride, pca_dims)
  stride_metrics.txt              – full results table
  stride_summary_chart.png        – bar chart (stride axis, best pca highlighted)
  pca_summary_chart.png           – bar chart (pca axis, best stride highlighted)
  heatmap_ssim.png                – SSIM heatmap  stride × pca_dims
  heatmap_psnr.png                – PSNR heatmap  stride × pca_dims
  heatmap_l2.png                  – Latent L2 heatmap
"""

import argparse
import math
import os

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from PIL import Image
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from torchvision import transforms
from torchvision.models import inception_v3
from tqdm import tqdm


# ──────────────────────────────────────────────
# Predictor fitting
# ──────────────────────────────────────────────

def fit_predictor(train_latents: np.ndarray,
                  pca_dims: int,
                  ridge_alpha: float = 1.0) -> dict:
    """
    Fit a PCA(pca_dims) + Ridge predictor on consecutive frame pairs from
    `train_latents` (T, D).  Returns a predictor dict with keys 'svd', 'ridge'.
    """
    X = train_latents[:-1]   # (T-1, D)  predictors
    Y = train_latents[1:]    # (T-1, D)  targets
    T, D = X.shape

    actual_dims = min(pca_dims, T - 1, D)
    svd = TruncatedSVD(n_components=actual_dims, random_state=0)
    X_low = svd.fit_transform(X)              # (T-1, actual_dims)

    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(X_low, Y)

    return {"svd": svd, "ridge": ridge, "pca_dims": actual_dims}


# ──────────────────────────────────────────────
# VAE encode / decode
# ──────────────────────────────────────────────

def load_vae(model_id: str, device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(model_id)
    return vae.to(device).eval()


def frame_to_tensor(frame_bgr: np.ndarray, size: int = 512) -> torch.Tensor:
    """BGR numpy → normalised RGB tensor [-1,1], shape (1,3,H,W)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return tf(img).unsqueeze(0)


@torch.no_grad()
def encode_frame(vae: AutoencoderKL, tensor: torch.Tensor) -> np.ndarray:
    """Returns posterior mean, flat (D,)."""
    tensor = tensor.to(next(vae.parameters()).device)
    z = vae.encode(tensor).latent_dist.mean   # (1,C,H,W)
    return z.squeeze(0).cpu().float().numpy().reshape(-1)


@torch.no_grad()
def decode_latent(vae: AutoencoderKL,
                  z_flat: np.ndarray,
                  latent_shape: tuple) -> np.ndarray:
    """(D,) → (H_img, W_img, 3) uint8 RGB."""
    device = next(vae.parameters()).device
    C, H, W = latent_shape
    z = torch.from_numpy(z_flat).to(device).reshape(1, C, H, W)
    out = vae.decode(z).sample                          # (1,3,H,W)
    out = (out.clamp(-1, 1) + 1) / 2
    out = (out * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return out[0].transpose(1, 2, 0)                   # (H,W,3) RGB


# ──────────────────────────────────────────────
# Video frame extraction
# ──────────────────────────────────────────────

def extract_gop_latents(video_path: str,
                         vae: AutoencoderKL,
                         stride: int,
                         gop_size: int = 8,
                         start_frame: int = 0,
                         frame_size: int = 512) -> tuple[np.ndarray, list[int]]:
    """
    Read exactly `gop_size` frames from the video starting at `start_frame`,
    stepping by `stride` real video frames each time.

    Returns
    -------
    latents      : (gop_size, D) float32
    frame_indices: list of actual video frame numbers read
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Clamp so we don't go past the end of the video
    frame_indices = [start_frame + i * stride for i in range(gop_size)]
    frame_indices = [min(idx, total - 1) for idx in frame_indices]

    latents = []
    device = next(vae.parameters()).device
    for idx in tqdm(frame_indices, desc=f"  encoding stride={stride}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {idx} from {video_path}")
        tensor = frame_to_tensor(frame, size=frame_size).to(device)
        z = encode_frame(vae, tensor)
        latents.append(z)

    cap.release()
    return np.stack(latents, axis=0).astype(np.float32), frame_indices


# ──────────────────────────────────────────────
# GOP reconstruction
# ──────────────────────────────────────────────

def reconstruct_gop(latents: np.ndarray, predictor: dict) -> np.ndarray:
    """
    Given (gop_size, D) latents where latents[0] is the keyframe,
    reconstruct latents[1..] by chaining the Ridge predictor.

    Returns (gop_size, D) float32 — frame 0 is exact, frames 1+ are predicted.
    """
    T, D = latents.shape
    svd   = predictor["svd"]
    ridge = predictor["ridge"]

    recon = np.empty_like(latents)
    recon[0] = latents[0]          # keyframe stored verbatim
    prev = latents[0]
    for t in range(1, T):
        prev_low = svd.transform(prev[np.newaxis])   # (1, pca_dims)
        pred     = ridge.predict(prev_low)[0]        # (D,)
        recon[t] = pred.astype(np.float32)
        prev     = recon[t]
    return recon


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def image_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """gt, pred: (H,W,3) uint8.  Returns ssim, psnr."""
    g = gt.astype(np.float32) / 255.0
    p = pred.astype(np.float32) / 255.0
    return {
        "ssim": float(ssim_fn(g, p, data_range=1.0, channel_axis=2)),
        "psnr": float(psnr_fn(g, p, data_range=1.0)),
    }


def latent_metrics(z_gt: np.ndarray, z_pred: np.ndarray) -> dict:
    """z_gt, z_pred: (D,).  Returns l2, cosine."""
    l2  = float(np.linalg.norm(z_gt - z_pred))
    cos = float(np.dot(z_gt, z_pred) /
                (np.linalg.norm(z_gt) * np.linalg.norm(z_pred) + 1e-12))
    return {"l2": l2, "cosine": cos}


class InceptionFeatureExtractor:
    def __init__(self, device):
        self.device = device
        model = inception_v3(weights="DEFAULT", transform_input=False)
        model.fc = torch.nn.Identity()
        model.aux_logits = False
        self.model = model.to(device).eval()
        self.prep = transforms.Compose([
            transforms.Resize((299, 299),
                               interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_features(self, imgs_uint8: np.ndarray) -> np.ndarray:
        """imgs_uint8: (N,H,W,3) → (N,2048)."""
        feats = []
        for i in range(len(imgs_uint8)):
            t = (torch.from_numpy(imgs_uint8[i]).float()
                 .permute(2, 0, 1).unsqueeze(0) / 255.0)
            feats.append(self.model(self.prep(t).to(self.device)).cpu().numpy())
        return np.concatenate(feats, axis=0)


def compute_fid(f1: np.ndarray, f2: np.ndarray) -> float:
    if len(f1) < 2:
        return float("nan")
    mu1, s1 = f1.mean(0), np.cov(f1, rowvar=False)
    mu2, s2 = f2.mean(0), np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    # sqrtm on 2048×2048 can have small imaginary parts
    covmean, _ = sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


# ──────────────────────────────────────────────
# Qualitative panels
# ──────────────────────────────────────────────

def save_panel(kf_img: np.ndarray,
               gt_img: np.ndarray,
               pred_img: np.ndarray,
               offset: int,
               stride: int,
               frame_idx: int,
               metrics: dict,
               lat_metrics: dict,
               out_path: str):
    """
    Three-column panel: Keyframe | Ground Truth | Predicted.
    Annotated with pixel metrics + latent metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                              gridspec_kw={"wspace": 0.06})
    entries = [
        (kf_img,
         f"Keyframe  (frame {frame_idx - offset * stride})"),
        (gt_img,
         f"GT  +{offset} steps  (frame {frame_idx})\n"
         f"stride={stride}"),
        (pred_img,
         f"Predicted\n"
         f"SSIM={metrics['ssim']:.3f}  PSNR={metrics['psnr']:.1f} dB\n"
         f"L2={lat_metrics['l2']:.3f}  cos={lat_metrics['cosine']:.4f}"),
    ]
    for ax, (img, title) in zip(axes, entries):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_error_strip(gt_imgs: list[np.ndarray],
                     pred_imgs: list[np.ndarray],
                     stride: int,
                     out_path: str):
    """
    One row per inter-frame showing |GT - Pred| error heat-map (amplified 4x).
    """
    n = len(gt_imgs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, (gt, pred) in enumerate(zip(gt_imgs, pred_imgs)):
        err = np.abs(gt.astype(np.int16) - pred.astype(np.int16))
        err = np.clip(err * 4, 0, 255).astype(np.uint8)
        axes[i].imshow(err)
        axes[i].set_title(f"+{i+1}", fontsize=8)
        axes[i].axis("off")
    fig.suptitle(f"|GT − Predicted| × 4  (stride={stride})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# Per-stride evaluation
# ──────────────────────────────────────────────

def evaluate_stride(video_path: str,
                     predictor: dict,
                     vae: AutoencoderKL,
                     latent_shape: tuple,
                     stride: int,
                     gop_size: int,
                     start_frame: int,
                     frame_size: int,
                     output_dir: str,
                     compute_fid_flag: bool,
                     inception: "InceptionFeatureExtractor | None") -> dict:
    """
    Run the full evaluation for one (stride, pca_dims) combination.
    Returns a result dict.
    """
    pca_dims = predictor.get("pca_dims", "?")
    print(f"\n{'='*60}")
    print(f"  Stride={stride}  PCA dims={pca_dims}  "
          f"(frames {start_frame}, {start_frame+stride}, ..., "
          f"{start_frame + (gop_size-1)*stride})")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract latents
    latents, frame_indices = extract_gop_latents(
        video_path, vae, stride=stride, gop_size=gop_size,
        start_frame=start_frame, frame_size=frame_size)

    # 2. Reconstruct with predictor
    recon = reconstruct_gop(latents, predictor)

    # 3. Decode all frames to images (GT + predicted + keyframe repeated)
    imgs_gt   = [decode_latent(vae, latents[t], latent_shape) for t in range(gop_size)]
    imgs_pred = [decode_latent(vae, recon[t],   latent_shape) for t in range(gop_size)]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    kf_img = imgs_gt[0]    # keyframe decoded from GT latent

    # 4. Pixel metrics + latent metrics (inter-frames only: indices 1..gop_size-1)
    px_metrics  = []
    lat_metrics = []
    for t in range(1, gop_size):
        px_metrics.append(image_metrics(imgs_gt[t], imgs_pred[t]))
        lat_metrics.append(latent_metrics(latents[t], recon[t]))

    ssim_vals = np.array([m["ssim"] for m in px_metrics])
    psnr_vals = np.array([m["psnr"] for m in px_metrics])
    l2_vals   = np.array([m["l2"]   for m in lat_metrics])
    cos_vals  = np.array([m["cosine"] for m in lat_metrics])

    # 5. FID (inter-frames: 7 images — small N, treat as indicative)
    fid_score = float("nan")
    if compute_fid_flag and inception is not None:
        gt_arr   = np.stack(imgs_gt[1:],   axis=0)   # (7, H, W, 3)
        pred_arr = np.stack(imgs_pred[1:], axis=0)
        feats_gt   = inception.get_features(gt_arr)
        feats_pred = inception.get_features(pred_arr)
        fid_score  = compute_fid(feats_gt, feats_pred)
        print(f"  FID (inter-frames) = {fid_score:.2f}")

    # 6. Save qualitative panels (one per inter-frame)
    for t in range(1, gop_size):
        panel_path = os.path.join(output_dir, f"frame_{t:02d}.png")
        save_panel(
            kf_img=kf_img,
            gt_img=imgs_gt[t],
            pred_img=imgs_pred[t],
            offset=t,
            stride=stride,
            frame_idx=frame_indices[t],
            metrics=px_metrics[t - 1],
            lat_metrics=lat_metrics[t - 1],
            out_path=panel_path,
        )

    # 7. Error strip
    save_error_strip(
        gt_imgs=imgs_gt[1:],
        pred_imgs=imgs_pred[1:],
        stride=stride,
        out_path=os.path.join(output_dir, "error_strip.png"),
    )

    print(f"  SSIM  {ssim_vals.mean():.4f}  ±{ssim_vals.std():.4f}  "
          f"[{ssim_vals.min():.4f} - {ssim_vals.max():.4f}]")
    print(f"  PSNR  {psnr_vals.mean():.2f}  ±{psnr_vals.std():.2f} dB")
    print(f"  L2    {l2_vals.mean():.4f}    cosine {cos_vals.mean():.6f}")

    return {
        "stride": stride,
        "pca_dims": pca_dims,
        "frame_indices": frame_indices,
        # aggregate pixel metrics
        "ssim_mean":   float(ssim_vals.mean()),
        "ssim_std":    float(ssim_vals.std()),
        "ssim_min":    float(ssim_vals.min()),
        "ssim_max":    float(ssim_vals.max()),
        "psnr_mean":   float(psnr_vals.mean()),
        "psnr_std":    float(psnr_vals.std()),
        "psnr_min":    float(psnr_vals.min()),
        "psnr_max":    float(psnr_vals.max()),
        # per-frame arrays (for plotting)
        "ssim_per_frame": ssim_vals.tolist(),
        "psnr_per_frame": psnr_vals.tolist(),
        "l2_per_frame":   l2_vals.tolist(),
        "cos_per_frame":  cos_vals.tolist(),
        # latent metrics
        "l2_mean":  float(l2_vals.mean()),
        "l2_std":   float(l2_vals.std()),
        "cos_mean": float(cos_vals.mean()),
        "cos_std":  float(cos_vals.std()),
        # FID
        "fid": fid_score,
    }


# ──────────────────────────────────────────────
# Summary charts
# ──────────────────────────────────────────────

def _bar_panel(ax, x, vals, errs, labels, ylabel, title, color):
    """Helper: annotated bar chart with error bars."""
    bars = ax.bar(x, vals, width=0.55, color=color, alpha=0.8,
                  yerr=errs, capsize=5, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.4)
    max_err = max(errs) if errs else 0
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_err * 0.05 + 1e-9,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)


def save_stride_chart(results_by_stride: dict[int, dict],
                       best_pca: int,
                       out_path: str):
    """
    4-panel bar chart with stride on the x-axis, using metrics from best_pca.
    """
    strides = sorted(results_by_stride)
    data    = [results_by_stride[s][best_pca] for s in strides]
    x       = np.arange(len(strides))
    xlabels = [f"S={s}" for s in strides]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    configs = [
        (axes[0, 0], "ssim_mean", "ssim_std", "SSIM",         "Mean SSIM vs Stride",       "steelblue"),
        (axes[0, 1], "psnr_mean", "psnr_std", "PSNR (dB)",    "Mean PSNR vs Stride",       "darkorange"),
        (axes[1, 0], "l2_mean",   "l2_std",   "Latent L2",    "Mean Latent L2 vs Stride",  "seagreen"),
        (axes[1, 1], "cos_mean",  "cos_std",  "Cosine sim",   "Mean Cosine Sim vs Stride", "crimson"),
    ]
    for ax, km, ks, ylabel, title, color in configs:
        _bar_panel(ax, x, [d[km] for d in data], [d[ks] for d in data],
                   xlabels, ylabel, title, color)

    fig.suptitle(
        f"Stride Sensitivity  |  GOP=8  |  PCA dims={best_pca}\n"
        f"Inter-frame reconstruction quality (frames 1-7)",
        fontsize=12)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[stride_study] stride chart → {out_path}")


def save_pca_chart(results_by_pca: dict[int, dict],
                   best_stride: int,
                   out_path: str):
    """
    4-panel bar chart with pca_dims on the x-axis, using metrics from best_stride.
    """
    pcas    = sorted(results_by_pca)
    data    = [results_by_pca[p][best_stride] for p in pcas]
    x       = np.arange(len(pcas))
    xlabels = [f"P={p}" for p in pcas]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    configs = [
        (axes[0, 0], "ssim_mean", "ssim_std", "SSIM",       "Mean SSIM vs PCA dims",      "steelblue"),
        (axes[0, 1], "psnr_mean", "psnr_std", "PSNR (dB)",  "Mean PSNR vs PCA dims",      "darkorange"),
        (axes[1, 0], "l2_mean",   "l2_std",   "Latent L2",  "Mean Latent L2 vs PCA dims", "seagreen"),
        (axes[1, 1], "cos_mean",  "cos_std",  "Cosine sim", "Mean Cos Sim vs PCA dims",   "crimson"),
    ]
    for ax, km, ks, ylabel, title, color in configs:
        _bar_panel(ax, x, [d[km] for d in data], [d[ks] for d in data],
                   xlabels, ylabel, title, color)

    fig.suptitle(
        f"PCA Dimension Sensitivity  |  GOP=8  |  stride={best_stride}\n"
        f"Inter-frame reconstruction quality (frames 1-7)",
        fontsize=12)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[stride_study] PCA chart → {out_path}")


def save_heatmaps(all_results: list[dict], out_dir: str):
    """
    One heatmap per metric (SSIM, PSNR, L2) — rows=stride, cols=pca_dims.
    """
    strides  = sorted({r["stride"]   for r in all_results})
    pca_list = sorted({r["pca_dims"] for r in all_results})

    # Build 2-D grids
    lookup = {(r["stride"], r["pca_dims"]): r for r in all_results}

    for metric_key, title, cmap, fmt in [
        ("ssim_mean", "Mean SSIM",    "RdYlGn",  ".3f"),
        ("psnr_mean", "Mean PSNR (dB)", "RdYlGn", ".2f"),
        ("l2_mean",   "Mean Latent L2", "RdYlGn_r", ".3f"),
        ("cos_mean",  "Mean Cosine Sim", "RdYlGn", ".4f"),
    ]:
        grid = np.array([[lookup[(s, p)][metric_key]
                          for p in pca_list]
                         for s in strides])   # (n_strides, n_pcas)

        fig, ax = plt.subplots(figsize=(max(6, len(pca_list) * 1.4),
                                         max(4, len(strides) * 1.0)))
        im = ax.imshow(grid, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(pca_list)))
        ax.set_xticklabels([f"P={p}" for p in pca_list], fontsize=9)
        ax.set_yticks(range(len(strides)))
        ax.set_yticklabels([f"S={s}" for s in strides], fontsize=9)
        ax.set_xlabel("PCA dims", fontsize=11)
        ax.set_ylabel("Stride", fontsize=11)
        ax.set_title(f"Heatmap: {title}  (rows=stride, cols=PCA dims)", fontsize=12)

        # Annotate cells
        for si, s in enumerate(strides):
            for pi, p in enumerate(pca_list):
                val = grid[si, pi]
                ax.text(pi, si, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8,
                        color="white" if im.norm(val) < 0.4 else "black")

        plt.tight_layout()
        fname = metric_key.replace("_mean", "")
        path  = os.path.join(out_dir, f"heatmap_{fname}.png")
        plt.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[stride_study] heatmap ({metric_key}) → {path}")


def save_per_frame_curves(results: list[dict], out_path: str, label_key: str = "stride"):
    """
    Line plots showing per-frame metric degradation within the GOP.
    One curve per result; labelled by label_key.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                              gridspec_kw={"hspace": 0.4, "wspace": 0.32})
    offsets = list(range(1, 8))
    colors  = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

    configs = [
        (axes[0, 0], "ssim_per_frame", "SSIM",      "SSIM vs GOP offset"),
        (axes[0, 1], "psnr_per_frame", "PSNR (dB)", "PSNR vs GOP offset"),
        (axes[1, 0], "l2_per_frame",   "Latent L2", "Latent L2 vs GOP offset"),
        (axes[1, 1], "cos_per_frame",  "Cosine sim","Cosine similarity vs GOP offset"),
    ]
    for ax, key, ylabel, title in configs:
        for r, c in zip(results, colors):
            lbl = f"S={r['stride']}, P={r['pca_dims']}"
            ax.plot(offsets, r[key], marker="o", color=c,
                    label=lbl, linewidth=1.8, markersize=5)
        ax.set_xlabel("Offset within GOP", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(offsets)
        ax.legend(fontsize=7, loc="best")
        ax.grid(ls="--", alpha=0.4)

    fig.suptitle("Per-frame metric degradation within GOP", fontsize=12)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[stride_study] per-frame curves → {out_path}")


# ──────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────

def save_summary_table(all_results: list[dict], out_path: str):
    col = 11
    header = (f"{'Stride':>7}  {'PCA':>5}  {'SSIM':>{col}}  {'PSNR (dB)':>{col}}  "
              f"{'Latent L2':>{col}}  {'Cosine sim':>{col}}  {'FID':>{col}}")
    sep = "-" * len(header)

    rows = [
        "=" * len(header),
        "  Stride × PCA Sensitivity Study  |  GOP size = 8, inter-frames only",
        "=" * len(header),
        header, sep,
    ]
    prev_stride = None
    for r in sorted(all_results, key=lambda x: (x["stride"], x["pca_dims"])):
        if prev_stride is not None and r["stride"] != prev_stride:
            rows.append(sep)   # blank separator between stride groups
        fid_str = f"{r['fid']:.2f}" if not math.isnan(r["fid"]) else "   n/a"
        rows.append(
            f"{r['stride']:>7}  {r['pca_dims']:>5}  "
            f"{r['ssim_mean']:>{col}.4f}  "
            f"{r['psnr_mean']:>{col}.2f}  "
            f"{r['l2_mean']:>{col}.4f}  "
            f"{r['cos_mean']:>{col}.6f}  "
            f"{fid_str:>{col}}"
        )
        prev_stride = r["stride"]

    rows.append("=" * len(header))

    best  = max(all_results, key=lambda r: r["ssim_mean"])
    worst = min(all_results, key=lambda r: r["ssim_mean"])
    rows += [
        "",
        f"  Best  (stride={best['stride']}, pca={best['pca_dims']}):  "
        f"SSIM={best['ssim_mean']:.4f}  PSNR={best['psnr_mean']:.2f} dB",
        f"  Worst (stride={worst['stride']}, pca={worst['pca_dims']}):  "
        f"SSIM={worst['ssim_mean']:.4f}  PSNR={worst['psnr_mean']:.2f} dB",
        "=" * len(header),
    ]

    table = "\n".join(rows)
    print("\n" + table)
    with open(out_path, "w") as fh:
        fh.write(table + "\n")
    print(f"[stride_study] table → {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_stride_study(
    video_path: str,
    train_latents_path: str,
    output_dir: str,
    strides: list[int] = (1, 2, 4, 8, 16),
    pca_dims_list: list[int] = (16, 32, 64, 128, 256),
    gop_size: int = 8,
    start_frame: int = 0,
    ridge_alpha: float = 1.0,
    model_id: str = "stabilityai/sd-vae-ft-mse",
    latent_shape: tuple = (4, 64, 64),
    frame_size: int = 512,
    compute_fid_flag: bool = True,
) -> list[dict]:
    """
    Run the full stride × pca_dims sensitivity study and save all outputs
    to `output_dir`.

    For each (stride, pca_dims) combination a fresh Ridge predictor is fitted
    from `train_latents_path`, the GOP is reconstructed from a keyframe, and
    SSIM / PSNR / latent-L2 / cosine / FID metrics are recorded.

    Final outputs
    -------------
    heatmap_ssim.png / heatmap_psnr.png / heatmap_l2.png / heatmap_cos.png
    stride_summary_chart.png   – stride axis, best pca_dims
    pca_summary_chart.png      – pca_dims axis, stride=1
    stride_per_frame_curves.png
    stride_metrics.txt
    stride_S_pca_P/            – per-combination panels & error strips
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load training latents (used to fit predictors) ──────────────────
    print(f"[stride_study] loading training latents from {train_latents_path} …")
    train_latents = np.load(train_latents_path).astype(np.float32)
    print(f"[stride_study] train_latents shape: {train_latents.shape}")

    # ── Fit one predictor per pca_dims (cheap, done once) ───────────────
    pca_dims_list = list(pca_dims_list)
    predictors: dict[int, dict] = {}
    for pca_dims in pca_dims_list:
        print(f"[stride_study] fitting predictor  pca_dims={pca_dims} …")
        predictors[pca_dims] = fit_predictor(train_latents, pca_dims, ridge_alpha)

    # ── Load VAE (once, shared) ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stride_study] loading VAE ({model_id}) on {device} …")
    vae = load_vae(model_id, device)

    # ── Load Inception once if FID requested ─────────────────────────────
    inception = None
    if compute_fid_flag:
        print("[stride_study] loading Inception-v3 for FID …")
        inception = InceptionFeatureExtractor(device)

    # ── Double sweep ─────────────────────────────────────────────────────
    all_results: list[dict] = []
    for pca_dims in pca_dims_list:
        predictor = predictors[pca_dims]
        for stride in strides:
            run_dir = os.path.join(output_dir, f"stride_{stride}_pca_{pca_dims}")
            r = evaluate_stride(
                video_path=video_path,
                predictor=predictor,
                vae=vae,
                latent_shape=latent_shape,
                stride=stride,
                gop_size=gop_size,
                start_frame=start_frame,
                frame_size=frame_size,
                output_dir=run_dir,
                compute_fid_flag=compute_fid_flag,
                inception=inception,
            )
            all_results.append(r)

    # ── Aggregate heatmaps ───────────────────────────────────────────────
    save_heatmaps(all_results, output_dir)

    # ── Per-frame degradation curves (all combos) ─────────────────────
    save_per_frame_curves(
        all_results,
        os.path.join(output_dir, "stride_per_frame_curves.png"),
    )

    # ── Stride bar chart (best pca_dims = highest mean SSIM at stride=1) ─
    by_pca_at_s1 = {r["pca_dims"]: r for r in all_results if r["stride"] == strides[0]}
    best_pca = max(by_pca_at_s1, key=lambda p: by_pca_at_s1[p]["ssim_mean"])
    stride_slice = {s: {best_pca: next(r for r in all_results
                                        if r["stride"] == s and r["pca_dims"] == best_pca)}
                    for s in strides}
    save_stride_chart(
        stride_slice,
        best_pca,
        os.path.join(output_dir, "stride_summary_chart.png"),
    )

    # ── PCA bar chart (stride=1 slice) ────────────────────────────────
    pca_slice = {p: {strides[0]: next(r for r in all_results
                                       if r["pca_dims"] == p and r["stride"] == strides[0])}
                 for p in pca_dims_list}
    save_pca_chart(
        pca_slice,
        strides[0],
        os.path.join(output_dir, "pca_summary_chart.png"),
    )

    # ── Summary table ──────────────────────────────────────────────────
    save_summary_table(all_results, os.path.join(output_dir, "stride_metrics.txt"))

    print(f"\n[stride_study] all outputs in {os.path.abspath(output_dir)}")
    return all_results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stride × PCA-dims sensitivity study: fixed GOP=8, "
            "sweep temporal strides and predictor PCA dimensions."
        )
    )
    p.add_argument("--video", required=True,
                   help="Path to input video file.")
    p.add_argument("--train_latents", default="results/latents.npy",
                   help="Path to latents.npy used to fit the Ridge predictors "
                        "(default: results/latents.npy).")
    p.add_argument("--strides", default="1,2,4,8,16",
                   help="Comma-separated strides to test (default: 1,2,4,8,16).")
    p.add_argument("--pca_dims", default="16,32,64,128,256",
                   help="Comma-separated PCA dimension values to sweep "
                        "(default: 16,32,64,128,256).")
    p.add_argument("--ridge_alpha", type=float, default=1.0,
                   help="Ridge regression regularisation strength (default: 1.0).")
    p.add_argument("--gop_size", type=int, default=8,
                   help="Number of frames per GOP (default: 8).")
    p.add_argument("--start_frame", type=int, default=0,
                   help="First video frame index of the GOP (default: 0).")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse",
                   help="HuggingFace VAE model ID.")
    p.add_argument("--latent_shape", default="4,64,64",
                   help="C,H,W of one latent (default: 4,64,64 for SD VAE @ 512px).")
    p.add_argument("--frame_size", type=int, default=512,
                   help="Resize frames to this square size.")
    p.add_argument("--no_fid", action="store_true",
                   help="Skip FID computation (faster).")
    p.add_argument("--output", default="results/stride_study/",
                   help="Output directory (default: results/stride_study/).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    C, H, W    = map(int, args.latent_shape.split(","))
    strides    = [int(s) for s in args.strides.split(",")]
    pca_dims   = [int(p) for p in args.pca_dims.split(",")]
    run_stride_study(
        video_path=args.video,
        train_latents_path=args.train_latents,
        output_dir=args.output,
        strides=strides,
        pca_dims_list=pca_dims,
        gop_size=args.gop_size,
        start_frame=args.start_frame,
        ridge_alpha=args.ridge_alpha,
        model_id=args.vae,
        latent_shape=(C, H, W),
        frame_size=args.frame_size,
        compute_fid_flag=not args.no_fid,
    )
