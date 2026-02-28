"""
Step 6 – Keyframe + predictor GOP evaluation.

Scheme
------
Divide the T latent frames into Groups of Pictures (GOPs) of size `gop_size`.
  • Frame 0 of each GOP is stored verbatim as a keyframe latent.
  • Frames 1 … gop_size-1 within the GOP are *not* stored; they are
    reconstructed by chaining the Ridge predictor forward from the keyframe.

Memory saving
-------------
  Full storage  : T  × D float32 values
  GOP storage   : ceil(T / gop_size) × D float32 values
  Saving ratio  : T / ceil(T / gop_size)

The script evaluates reconstruction quality (SSIM, PSNR, FID) and prints a
table comparing different GOP sizes side-by-side.

Inputs:
  <data_dir>/latents.npy      – (T, D) float32
  <data_dir>/predictor.pkl    – SVD + Ridge saved by build_populations.py

Outputs:
  <output>/gop_eval/
      panels_gop{K}/frame_NNNN.png   – Keyframe | GT | Reconstructed panels
      gop_metrics.txt                – Summary table for all GOP sizes tested
"""

import argparse
import math
import os
import pickle

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torchvision import transforms
from torchvision.models import inception_v3
from tqdm import tqdm


# ──────────────────────────────────────────────
# VAE helpers  (shared logic with evaluate_predictor.py)
# ──────────────────────────────────────────────


def load_vae(model_id: str, device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(model_id)
    return vae.to(device).eval()


@torch.no_grad()
def _decode_batch(vae: AutoencoderKL,
                  batch_flat: np.ndarray,
                  latent_shape: tuple) -> np.ndarray:
    """(B, D) → (B, H_img, W_img, 3) uint8 RGB."""
    device = next(vae.parameters()).device
    C, H, W = latent_shape
    B = batch_flat.shape[0]
    z = torch.from_numpy(batch_flat).to(device).reshape(B, C, H, W)
    out = vae.decode(z).sample                           # (B, 3, H, W)
    out = (out.clamp(-1, 1) + 1) / 2
    out = (out * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return out.transpose(0, 2, 3, 1)                    # (B, H, W, 3)


def decode_to_memmap(vae, latents_flat, latent_shape, save_path,
                     batch_size=4, desc="Decoding"):
    """Stream-decode all latents to a memory-mapped file. Returns the mmap."""
    N = latents_flat.shape[0]
    sample = _decode_batch(vae, latents_flat[:1], latent_shape)
    _, H_img, W_img, _ = sample.shape
    mmap = np.lib.format.open_memmap(
        save_path, mode="w+", dtype=np.uint8, shape=(N, H_img, W_img, 3))
    mmap[0] = sample[0]
    for s in tqdm(range(1, N, batch_size), desc=desc, unit="batch"):
        e = min(s + batch_size, N)
        mmap[s:e] = _decode_batch(vae, latents_flat[s:e], latent_shape)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mmap.flush()
    return mmap


# ──────────────────────────────────────────────
# GOP reconstruction
# ──────────────────────────────────────────────


def reconstruct_with_gop(latents: np.ndarray,
                          predictor: dict,
                          gop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct all T latents using keyframes every `gop_size` frames.

    Keyframe indices: 0, gop_size, 2*gop_size, …
    Inter-frames are predicted by chaining the Ridge predictor forward:
        ẑ_{k+1} = Ridge(ẑ_k)  starting from the stored keyframe ẑ_k.

    Returns
    -------
    reconstructed : (T, D) float32
        Full reconstructed sequence (keyframes exact, inter-frames predicted).
    keyframe_mask : (T,) bool
        True where a real keyframe was stored.
    """
    T, D = latents.shape
    svd = predictor["svd"]
    ridge = predictor["ridge"]

    reconstructed = np.empty_like(latents)
    keyframe_mask = np.zeros(T, dtype=bool)

    for gop_start in range(0, T, gop_size):
        # Store the keyframe verbatim
        kf = latents[gop_start]
        reconstructed[gop_start] = kf
        keyframe_mask[gop_start] = True

        # Chain-predict every subsequent frame in this GOP
        prev = kf
        for offset in range(1, gop_size):
            t = gop_start + offset
            if t >= T:
                break
            prev_low = svd.transform(prev[np.newaxis])  # (1, pca_dims)
            pred = ridge.predict(prev_low)[0]           # (D,)
            reconstructed[t] = pred.astype(np.float32)
            prev = pred

    return reconstructed, keyframe_mask


def memory_stats(T: int, D: int, gop_size: int,
                 dtype_bytes: int = 4) -> dict:
    """Return storage statistics for a given GOP size."""
    n_keyframes = math.ceil(T / gop_size)
    full_bytes = T * D * dtype_bytes
    kf_bytes = n_keyframes * D * dtype_bytes
    saved_bytes = full_bytes - kf_bytes
    ratio = T / n_keyframes          # compression ratio
    return {
        "gop_size": gop_size,
        "T": T,
        "n_keyframes": n_keyframes,
        "full_MB": full_bytes / 1e6,
        "kf_MB": kf_bytes / 1e6,
        "saved_MB": saved_bytes / 1e6,
        "ratio": ratio,
        "pct_saved": 100.0 * saved_bytes / full_bytes,
    }


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────


def compute_image_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_f = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0
    s = ssim_fn(gt_f, pred_f, data_range=1.0, channel_axis=2)
    p = psnr_fn(gt_f, pred_f, data_range=1.0)
    return {"ssim": float(s), "psnr": float(p)}


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
    def get_features(self, images_uint8: np.ndarray, batch_size: int = 32):
        feats = []
        for i in range(0, len(images_uint8), batch_size):
            b = images_uint8[i: i + batch_size]
            t = torch.from_numpy(b).float().permute(0, 3, 1, 2) / 255.0
            feats.append(self.model(self.prep(t).to(self.device)).cpu().numpy())
        return np.concatenate(feats, axis=0)


def compute_fid(f1: np.ndarray, f2: np.ndarray) -> float:
    mu1, s1 = f1.mean(0), np.cov(f1, rowvar=False)
    mu2, s2 = f2.mean(0), np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────


def _save_panel(keyframe_img: np.ndarray,
                gt_img: np.ndarray,
                pred_img: np.ndarray,
                frame_idx: int,
                gop_size: int,
                is_keyframe: bool,
                metrics: dict,
                out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                              gridspec_kw={"wspace": 0.05})
    gop_start = (frame_idx // gop_size) * gop_size
    label = "KEYFRAME" if is_keyframe else f"inter (offset +{frame_idx - gop_start})"
    data = [
        (keyframe_img,
         f"Keyframe  (t={gop_start})"),
        (gt_img,
         f"GT  (t={frame_idx})  [{label}]"),
        (pred_img,
         f"Reconstructed  (t={frame_idx})\n"
         f"SSIM={metrics['ssim']:.3f}   PSNR={metrics['psnr']:.1f} dB"),
    ]
    for ax, (img, title) in zip(axes, data):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# Per-GOP-size evaluation
# ──────────────────────────────────────────────


def evaluate_one_gop(
    latents: np.ndarray,
    predictor: dict,
    gop_size: int,
    vae: AutoencoderKL,
    latent_shape: tuple,
    output_dir: str,
    n_viz: int = 8,
    decode_batch: int = 4,
    compute_fid_flag: bool = True,
) -> dict:
    """
    Evaluate reconstruction quality for a single GOP size.
    Returns a dict of aggregated metrics + memory stats.
    """
    T, D = latents.shape
    print(f"\n[gop={gop_size}] reconstructing {T} frames …")
    recon, kf_mask = reconstruct_with_gop(latents, predictor, gop_size)
    mem = memory_stats(T, D, gop_size)

    # Decode GT and reconstructed to images (stream to memmap)
    tmp_gt   = os.path.join(output_dir, f"_tmp_gt_gop{gop_size}.npy")
    tmp_rec  = os.path.join(output_dir, f"_tmp_rec_gop{gop_size}.npy")
    tmp_kf   = os.path.join(output_dir, f"_tmp_kf_gop{gop_size}.npy")

    # Build an array of "corresponding keyframe" images (first frame of each GOP)
    kf_latents = np.array([
        latents[(t // gop_size) * gop_size] for t in range(T)
    ])  # (T, D)  – repeated keyframes for panel context

    imgs_gt  = decode_to_memmap(vae, latents, latent_shape, tmp_gt,
                                 decode_batch, f"gop={gop_size} GT")
    imgs_rec = decode_to_memmap(vae, recon,   latent_shape, tmp_rec,
                                 decode_batch, f"gop={gop_size} Recon")
    imgs_kf  = decode_to_memmap(vae, kf_latents, latent_shape, tmp_kf,
                                 decode_batch, f"gop={gop_size} KF")

    # Per-frame metrics (skip keyframes so we measure inter-frame quality)
    print(f"[gop={gop_size}] computing per-frame SSIM / PSNR …")
    all_metrics, inter_metrics = [], []
    for i in tqdm(range(T), desc=f"Metrics gop={gop_size}"):
        m = compute_image_metrics(imgs_gt[i], imgs_rec[i])
        all_metrics.append(m)
        if not kf_mask[i]:
            inter_metrics.append(m)

    ssim_all  = np.array([m["ssim"] for m in all_metrics])
    psnr_all  = np.array([m["psnr"] for m in all_metrics])
    ssim_inter = np.array([m["ssim"] for m in inter_metrics]) if inter_metrics else ssim_all
    psnr_inter = np.array([m["psnr"] for m in inter_metrics]) if inter_metrics else psnr_all

    # FID (all frames)
    fid_score = None
    if compute_fid_flag and T >= 2:
        print(f"[gop={gop_size}] computing FID …")
        ext = InceptionFeatureExtractor(next(vae.parameters()).device)
        fid_score = compute_fid(ext.get_features(imgs_gt),
                                ext.get_features(imgs_rec))
        print(f"[gop={gop_size}] FID = {fid_score:.2f}")

    # Save panels (spread evenly, prefer inter-frames for visual interest)
    panels_dir = os.path.join(output_dir, f"panels_gop{gop_size}")
    os.makedirs(panels_dir, exist_ok=True)
    viz_indices = np.linspace(0, T - 1, min(n_viz, T), dtype=int)
    for i in tqdm(viz_indices, desc=f"Panels gop={gop_size}"):
        _save_panel(
            keyframe_img=imgs_kf[i],
            gt_img=imgs_gt[i],
            pred_img=imgs_rec[i],
            frame_idx=int(i),
            gop_size=gop_size,
            is_keyframe=bool(kf_mask[i]),
            metrics=all_metrics[i],
            out_path=os.path.join(panels_dir, f"frame_{i:04d}.png"),
        )

    # Clean up temp memmaps
    for p in [tmp_gt, tmp_rec, tmp_kf]:
        try:
            os.remove(p)
        except OSError:
            pass

    return {
        "gop_size": gop_size,
        "ssim_mean": float(ssim_all.mean()),
        "ssim_inter_mean": float(ssim_inter.mean()),
        "psnr_mean": float(psnr_all.mean()),
        "psnr_inter_mean": float(psnr_inter.mean()),
        "fid": fid_score,
        **mem,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def run_gop_evaluation(
    data_dir: str,
    output_dir: str,
    gop_sizes: list[int] = (1, 3, 8, 10),
    model_id: str = "stabilityai/sd-vae-ft-mse",
    latent_shape: tuple = (4, 64, 64),
    n_viz: int = 8,
    decode_batch: int = 1,
    compute_fid_flag: bool = True,
) -> list[dict]:
    """
    Evaluate keyframe+predictor reconstruction for multiple GOP sizes.

    gop_size=1 means every frame is a keyframe → equivalent to storing
    everything (baseline, should give perfect metrics).

    Returns list of per-gop result dicts.
    """
    gop_dir = os.path.join(output_dir, "gop_eval")
    os.makedirs(gop_dir, exist_ok=True)

    # Load inputs
    latents = np.load(os.path.join(data_dir, "latents.npy")).astype(np.float32)
    print(f"[gop_eval] latents {latents.shape}")

    with open(os.path.join(data_dir, "predictor.pkl"), "rb") as fh:
        predictor = pickle.load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gop_eval] loading VAE on {device} …")
    vae = load_vae(model_id, device)

    results = []
    for gs in gop_sizes:
        if gs < 1:
            raise ValueError(f"gop_size must be ≥ 1, got {gs}")
        r = evaluate_one_gop(
            latents=latents,
            predictor=predictor,
            gop_size=gs,
            vae=vae,
            latent_shape=latent_shape,
            output_dir=gop_dir,
            n_viz=n_viz,
            decode_batch=decode_batch,
            compute_fid_flag=compute_fid_flag,
        )
        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────
    T, D = latents.shape
    full_mb = T * D * 4 / 1e6

    header = (
        f"\n{'GOP':>6}  {'KFs':>5}  {'Full MB':>8}  {'KF MB':>8}  "
        f"{'Saved MB':>9}  {'Ratio':>6}  {'%Saved':>7}  "
        f"{'SSIM(all)':>10}  {'SSIM(inter)':>12}  "
        f"{'PSNR(all)':>10}  {'PSNR(inter)':>12}  {'FID':>8}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        fid_str = f"{r['fid']:.2f}" if r["fid"] is not None else "  —"
        rows.append(
            f"{r['gop_size']:>6}  {r['n_keyframes']:>5}  "
            f"{r['full_MB']:>8.2f}  {r['kf_MB']:>8.2f}  "
            f"{r['saved_MB']:>9.2f}  {r['ratio']:>6.2f}x  "
            f"{r['pct_saved']:>6.1f}%  "
            f"{r['ssim_mean']:>10.4f}  {r['ssim_inter_mean']:>12.4f}  "
            f"{r['psnr_mean']:>10.2f}  {r['psnr_inter_mean']:>12.2f}  "
            f"{fid_str:>8}"
        )

    title_lines = [
        "=" * (len(header) + 2),
        "  Keyframe GOP Evaluation  —  latent memory vs reconstruction quality",
        f"  Sequence: T={T} frames,  D={D},  Full storage = {full_mb:.2f} MB",
        "=" * (len(header) + 2),
    ]
    summary = "\n".join(title_lines + rows + ["=" * (len(header) + 2)])
    print(summary)

    metrics_path = os.path.join(gop_dir, "gop_metrics.txt")
    with open(metrics_path, "w") as fh:
        fh.write(summary + "\n")
    print(f"[gop_eval] metrics saved → {metrics_path}")

    # ── Memory saving bar chart ────────────────────────────────────────────
    _plot_memory_quality(results, gop_dir)

    return results


def _plot_memory_quality(results: list[dict], output_dir: str):
    """Bar chart: memory saving % and SSIM(inter) for each GOP size."""
    gops     = [r["gop_size"] for r in results]
    pct      = [r["pct_saved"] for r in results]
    ssim_i   = [r["ssim_inter_mean"] for r in results]
    psnr_i   = [r["psnr_inter_mean"] for r in results]

    x = np.arange(len(gops))
    labels = [f"GOP={g}" for g in gops]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1 – memory saved %
    axes[0].bar(x, pct, width=0.6, color="steelblue", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Latent storage saved (%)")
    axes[0].set_title("Memory saving vs GOP size")
    axes[0].set_ylim(0, 105)
    for xi, v in zip(x, pct):
        axes[0].text(xi, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    # Panel 2 – SSIM (inter-frames only)
    axes[1].bar(x, ssim_i, width=0.6, color="darkorange", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("SSIM (inter-frames)")
    axes[1].set_title("Reconstruction quality (SSIM)")
    axes[1].set_ylim(0, 1.05)
    for xi, v in zip(x, ssim_i):
        axes[1].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Panel 3 – PSNR (inter-frames only)
    axes[2].bar(x, psnr_i, width=0.6, color="seagreen", alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("PSNR dB (inter-frames)")
    axes[2].set_title("Reconstruction quality (PSNR)")
    for xi, v in zip(x, psnr_i):
        axes[2].text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "gop_memory_quality.png")
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[gop_eval] chart saved → {chart_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate keyframe+predictor GOP scheme: quality vs memory.")
    p.add_argument("--data_dir", default="results/",
                   help="Dir with latents.npy and predictor.pkl.")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse",
                   help="HuggingFace VAE model ID (must match extraction).")
    p.add_argument("--latent_shape", default="4,64,64",
                   help="C,H,W of one latent (default: 4,64,64 for SD VAE @ 512px).")
    p.add_argument("--gop_sizes", default="1,3,8,10",
                   help="Comma-separated list of GOP sizes to evaluate (default: 1,3,8,10).")
    p.add_argument("--n_viz", type=int, default=8,
                   help="Number of comparison panels to save per GOP size.")
    p.add_argument("--decode_batch", type=int, default=1,
                   help="VAE decode batch size (lower if OOM).")
    p.add_argument("--no_fid", action="store_true",
                   help="Skip FID computation.")
    p.add_argument("--output", default="results/",
                   help="Output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    C, H, W = map(int, args.latent_shape.split(","))
    gop_sizes = [int(x) for x in args.gop_sizes.split(",")]
    run_gop_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output,
        gop_sizes=gop_sizes,
        model_id=args.vae,
        latent_shape=(C, H, W),
        n_viz=args.n_viz,
        decode_batch=args.decode_batch,
        compute_fid_flag=not args.no_fid,
    )
