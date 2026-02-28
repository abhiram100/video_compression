"""
Step 5 – Evaluate the conditional latent predictor.

For each consecutive frame pair (z_{t-1}, z_t):
  1. Predict ẑ_t = Ridge(PCA(z_{t-1}))
  2. Decode ground-truth z_t  →  img_gt
  3. Decode predicted  ẑ_t   →  img_pred
  4. Render side-by-side comparison panels  (saved to <output>/eval_frames/)
  5. Compute per-frame SSIM and PSNR
  6. Compute FID over the full set
  7. Print + save a metrics summary

Inputs:
  <data_dir>/latents.npy      – (T, D) float32
  <data_dir>/predictor.pkl    – saved SVD + Ridge from build_populations.py

Outputs:
  <output>/eval_frames/frame_XXXX.png  – side-by-side panels
  <output>/eval_metrics.txt            – SSIM / PSNR / FID summary
"""

import argparse
import os
import pickle

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from torchvision import transforms
from torchvision.models import inception_v3


# ──────────────────────────────────────────────
# VAE decode
# ──────────────────────────────────────────────


def load_vae(model_id: str, device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(model_id)
    return vae.to(device).eval()


@torch.no_grad()
def _decode_batch(vae: AutoencoderKL,
                  batch_flat: np.ndarray,
                  latent_shape: tuple[int, int, int]) -> np.ndarray:
    """Decode a single small batch of latents → (B, H, W, 3) uint8."""
    device = next(vae.parameters()).device
    C, H, W = latent_shape
    B = batch_flat.shape[0]
    z = torch.from_numpy(batch_flat).to(device).reshape(B, C, H, W)
    out = vae.decode(z).sample                      # (B, 3, H_img, W_img)
    out = (out.clamp(-1, 1) + 1) / 2
    out = (out * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return out.transpose(0, 2, 3, 1)               # (B, H, W, 3)  RGB uint8


def decode_latents_to_memmap(vae: AutoencoderKL,
                              latents_flat: np.ndarray,
                              latent_shape: tuple[int, int, int],
                              save_path: str,
                              batch_size: int = 4,
                              desc: str = "Decoding") -> np.ndarray:
    """
    Decode latents one small batch at a time and stream results directly to a
    memory-mapped file on disk.  Peak RAM = one batch of images, not all N.

    Args:
        latents_flat: (N, D) float32
        latent_shape: (C, H, W)
        save_path:    path for the .npy memmap file
        batch_size:   frames per GPU forward pass (reduce if OOM)

    Returns:
        numpy memmap array (N, H_img, W_img, 3) uint8
    """
    N = latents_flat.shape[0]

    # Decode one frame to learn the output spatial size
    sample = _decode_batch(vae, latents_flat[:1], latent_shape)  # (1, H, W, 3)
    _, H_img, W_img, _ = sample.shape

    # Pre-allocate the memmap on disk
    mmap = np.lib.format.open_memmap(
        save_path, mode="w+", dtype=np.uint8, shape=(N, H_img, W_img, 3)
    )
    mmap[0] = sample[0]

    for start in tqdm(range(1, N, batch_size), desc=desc, unit="batch"):
        end = min(start + batch_size, N)
        batch = _decode_batch(vae, latents_flat[start:end], latent_shape)
        mmap[start:end] = batch
        # Flush + free GPU cache after every batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mmap.flush()
    return mmap


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────


def predict_latents(latents: np.ndarray, predictor: dict) -> np.ndarray:
    """
    Run the saved SVD + Ridge predictor.

    Returns:
        Y_hat: (T-1, D) – predicted latents for frames 1 … T-1
    """
    svd: object = predictor["svd"]
    ridge: object = predictor["ridge"]

    X = latents[:-1]  # (T-1, D)
    X_low = svd.transform(X)  # (T-1, pca_dims)
    Y_hat = ridge.predict(X_low)  # (T-1, D)
    return Y_hat.astype(np.float32)


# ──────────────────────────────────────────────
# Per-frame metrics
# ──────────────────────────────────────────────


def compute_image_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    gt, pred: (H, W, 3) uint8
    Returns dict with ssim and psnr.
    """
    gt_f = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0
    s = ssim_fn(gt_f, pred_f, data_range=1.0, channel_axis=2)
    p = psnr_fn(gt_f, pred_f, data_range=1.0)
    return {"ssim": float(s), "psnr": float(p)}


# ──────────────────────────────────────────────
# FID
# ──────────────────────────────────────────────


class InceptionFeatureExtractor:
    """Extracts pool3 features from Inception-v3 for FID."""

    def __init__(self, device: torch.device):
        self.device = device
        model = inception_v3(weights="DEFAULT", transform_input=False)
        # Truncate at pool3 (2048-d)
        model.fc = torch.nn.Identity()
        model.aux_logits = False
        self.model = model.to(device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (299, 299), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def get_features(
        self, images_uint8: np.ndarray, batch_size: int = 32
    ) -> np.ndarray:
        """images_uint8: (N, H, W, 3)  →  features: (N, 2048)"""
        feats = []
        N = images_uint8.shape[0]
        for i in range(0, N, batch_size):
            batch = images_uint8[i : i + batch_size]  # (B, H, W, 3)
            t = torch.from_numpy(batch).float().permute(0, 3, 1, 2)  # (B, 3, H, W)
            t = t / 255.0
            t = self.preprocess(t).to(self.device)
            f = self.model(t)  # (B, 2048)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)


def compute_fid(feats_gt: np.ndarray, feats_pred: np.ndarray) -> float:
    """Fréchet distance between two Gaussian distributions fit to feature sets."""
    from scipy.linalg import sqrtm

    mu1, sigma1 = feats_gt.mean(0), np.cov(feats_gt, rowvar=False)
    mu2, sigma2 = feats_pred.mean(0), np.cov(feats_pred, rowvar=False)

    diff = mu1 - mu2
    # Numerically stable matrix sqrt
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


# ──────────────────────────────────────────────
# Side-by-side visualisation
# ──────────────────────────────────────────────


def _save_panel(
    prev: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    frame_idx: int,
    metrics: dict,
    out_path: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), gridspec_kw={"wspace": 0.05})

    data = [
        (prev, f"Previous frame  (t={frame_idx})"),
        (gt, f"Ground truth  (t={frame_idx + 1})"),
        (
            pred,
            f"Predicted  (t={frame_idx + 1})\n"
            f"SSIM={metrics['ssim']:.3f}   PSNR={metrics['psnr']:.1f} dB",
        ),
    ]
    for ax, (img, title) in zip(axes, data):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def run_evaluation(
    data_dir: str,
    output_dir: str,
    model_id: str = "stabilityai/sd-vae-ft-mse",
    latent_shape: tuple[int, int, int] = (4, 64, 64),
    n_viz: int = 20,
    decode_batch: int = 8,
    compute_fid_flag: bool = True,
):
    """
    Args:
        latent_shape: (C, H, W) of a single latent – (4, 64, 64) for SD VAE @ 512px.
        n_viz:        Number of side-by-side panels to save (evenly spaced).
        decode_batch: Batch size for VAE decoding (reduce if OOM).
        compute_fid_flag: Whether to compute FID (requires Inception + scipy).
    """
    frames_dir = os.path.join(output_dir, "eval_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # ── Load latents + predictor ──────────────────────────────────────────
    latents = np.load(os.path.join(data_dir, "latents.npy")).astype(np.float32)
    print(f"[eval] loaded latents {latents.shape}")

    with open(os.path.join(data_dir, "predictor.pkl"), "rb") as f:
        predictor = pickle.load(f)

    # ── Predict ──────────────────────────────────────────────────────────
    print("[eval] running Ridge predictor …")
    Y_hat = predict_latents(latents, predictor)  # (T-1, D)
    Y_gt = latents[1:]  # (T-1, D)
    Y_prev = latents[:-1]  # (T-1, D)  previous frames
    N = Y_gt.shape[0]

    # ── Decode (streaming batch-by-batch → memmap on disk) ───────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] loading VAE ({model_id}) on {device} …")
    vae = load_vae(model_id, device)

    print("[eval] decoding ground-truth frames …")
    imgs_gt = decode_latents_to_memmap(
        vae, Y_gt, latent_shape,
        save_path=os.path.join(output_dir, "_imgs_gt.npy"),
        batch_size=decode_batch, desc="GT decode",
    )

    print("[eval] decoding predicted frames …")
    imgs_pred = decode_latents_to_memmap(
        vae, Y_hat, latent_shape,
        save_path=os.path.join(output_dir, "_imgs_pred.npy"),
        batch_size=decode_batch, desc="Pred decode",
    )

    print("[eval] decoding previous frames (for panel context) …")
    imgs_prev = decode_latents_to_memmap(
        vae, Y_prev, latent_shape,
        save_path=os.path.join(output_dir, "_imgs_prev.npy"),
        batch_size=decode_batch, desc="Prev decode",
    )

    # ── Per-frame metrics ────────────────────────────────────────────────
    print("[eval] computing per-frame SSIM / PSNR …")
    all_metrics = []
    for i in tqdm(range(N), desc="Metrics"):
        m = compute_image_metrics(imgs_gt[i], imgs_pred[i])
        all_metrics.append(m)

    ssim_vals = np.array([m["ssim"] for m in all_metrics])
    psnr_vals = np.array([m["psnr"] for m in all_metrics])

    # ── FID ──────────────────────────────────────────────────────────────
    fid_score = None
    if compute_fid_flag and N >= 2:
        print("[eval] computing FID …")
        extractor = InceptionFeatureExtractor(device)
        feats_gt = extractor.get_features(imgs_gt)
        feats_pred = extractor.get_features(imgs_pred)
        fid_score = compute_fid(feats_gt, feats_pred)
        print(f"[eval] FID = {fid_score:.2f}")

    # ── Save side-by-side panels ──────────────────────────────────────────
    viz_indices = np.linspace(0, N - 1, min(n_viz, N), dtype=int)
    print(f"[eval] saving {len(viz_indices)} comparison panels …")
    for i in tqdm(viz_indices, desc="Panels"):
        out_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        _save_panel(
            imgs_prev[i],
            imgs_gt[i],
            imgs_pred[i],
            frame_idx=i,
            metrics=all_metrics[i],
            out_path=out_path,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    lines = [
        "=" * 52,
        "  Conditional Predictor Evaluation",
        "=" * 52,
        f"  Frames evaluated : {N}",
        f"  SSIM  mean ± std : {ssim_vals.mean():.4f} ± {ssim_vals.std():.4f}",
        f"  SSIM  median     : {np.median(ssim_vals):.4f}",
        f"  SSIM  min / max  : {ssim_vals.min():.4f} / {ssim_vals.max():.4f}",
        "",
        f"  PSNR  mean ± std : {psnr_vals.mean():.2f} ± {psnr_vals.std():.2f} dB",
        f"  PSNR  median     : {np.median(psnr_vals):.2f} dB",
        f"  PSNR  min / max  : {psnr_vals.min():.2f} / {psnr_vals.max():.2f} dB",
        "",
        f"  FID              : {fid_score:.2f}"
        if fid_score is not None
        else "  FID              : (skipped)",
        "=" * 52,
    ]
    summary = "\n".join(lines)
    print("\n" + summary)

    metrics_path = os.path.join(output_dir, "eval_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(summary + "\n")
    print(f"[eval] metrics saved → {metrics_path}")
    print(f"[eval] panels saved  → {frames_dir}/")

    return {
        "ssim_mean": float(ssim_vals.mean()),
        "psnr_mean": float(psnr_vals.mean()),
        "fid": fid_score,
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate conditional latent predictor quality."
    )
    p.add_argument(
        "--data_dir",
        default="results/",
        help="Directory with latents.npy and predictor.pkl.",
    )
    p.add_argument(
        "--vae",
        default="stabilityai/sd-vae-ft-mse",
        help="HuggingFace VAE model ID (must match extraction).",
    )
    p.add_argument(
        "--latent_shape",
        default="4,64,64",
        help="C,H,W of one latent vector (default: 4,64,64 for SD @ 512px).",
    )
    p.add_argument(
        "--n_viz",
        type=int,
        default=20,
        help="Number of side-by-side comparison panels to save.",
    )
    p.add_argument(
        "--decode_batch",
        type=int,
        default=1,
        help="VAE decode batch size (lower if OOM).",
    )
    p.add_argument(
        "--no_fid",
        action="store_true",
        help="Skip FID computation (faster, no scipy needed).",
    )
    p.add_argument("--output", default="results/", help="Output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    C, H, W = map(int, args.latent_shape.split(","))
    run_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output,
        model_id=args.vae,
        latent_shape=(C, H, W),
        n_viz=args.n_viz,
        decode_batch=args.decode_batch,
        compute_fid_flag=not args.no_fid,
    )
