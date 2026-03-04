"""
Step 1 – Extract VAE latents from a video file.

Outputs:
  <output>/latents.npy  – float32 array of shape (T, C*H*W)
"""

import argparse
import os

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def load_vae(model_id: str, device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(model_id)
    vae = vae.to(device).eval()
    return vae


def frame_to_tensor(frame_rgb: np.ndarray, size: int = 512) -> torch.Tensor:
    """RGB numpy → normalised tensor in [-1, 1], shape (1, 3, H, W)."""
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),                    # [0, 1]
        transforms.Normalize([0.5], [0.5]),       # [-1, 1]
    ])
    return transform(img).unsqueeze(0)            # (1, 3, H, W)


@torch.no_grad()
def encode_frame(vae: AutoencoderKL, tensor: torch.Tensor) -> np.ndarray:
    """Returns the mean of the posterior as a flat numpy vector."""
    tensor = tensor.to(next(vae.parameters()).device)
    posterior = vae.encode(tensor).latent_dist
    z = posterior.mean                            # (1, C, H, W)
    return z.squeeze(0).cpu().float().numpy()     # (C, H, W)


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def _frame_indices(total: int, n_frames: int, sampling: str) -> list[int]:
    """
    Return a sorted list of frame indices to extract.

    sampling='uniform'  – evenly spaced across the whole video
    sampling='dense'    – consecutive frames starting from frame 0
    """
    n = min(n_frames, total)
    if sampling == "dense":
        return list(range(n))
    else:  # uniform
        step = max(1, total // n)
        return [i * step for i in range(n)]


def extract_latents(video_path: str, n_frames: int, model_id: str,
                    output_dir: str, frame_size: int = 512,
                    sampling: str = "uniform") -> str:
    """
    Extract VAE latents from a video.

    Args:
        sampling: 'uniform' – evenly spaced across the full video (default).
                  'dense'   – first n_frames consecutive frames.
    """
    if sampling not in ("uniform", "dense"):
        raise ValueError(f"sampling must be 'uniform' or 'dense', got '{sampling}'")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device}  model={model_id}  sampling={sampling}")

    vae = load_vae(model_id, device)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _frame_indices(total, n_frames, sampling)
    print(f"[extract] video frames={total}  extracting {len(indices)} frames "
          f"({'every %d frame(s)' % max(1, total // n_frames) if sampling == 'uniform' else 'consecutive'})")

    latents = []
    collected = 0

    with tqdm(total=len(indices), desc="Encoding frames") as pbar:
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                break
            tensor = frame_to_tensor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), size=frame_size)
            z = encode_frame(vae, tensor)         # (C, H, W)
            latents.append(z.reshape(-1))         # flatten → (C*H*W,)
            collected += 1
            pbar.update(1)

    cap.release()

    latents_np = np.stack(latents, axis=0).astype(np.float32)  # (T, D)
    out_path = os.path.join(output_dir, "latents.npy")
    np.save(out_path, latents_np)
    print(f"[extract] saved {latents_np.shape} → {out_path}")
    return out_path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Extract VAE latents from a video.")
    p.add_argument("--video", required=True, help="Path to input video file.")
    p.add_argument("--frames", type=int, default=300,
                   help="Number of frames to extract (default: 300).")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse",
                   help="HuggingFace VAE model ID.")
    p.add_argument("--frame_size", type=int, default=512,
                   help="Resize frames to this square size before encoding.")
    p.add_argument("--sampling", default="uniform", choices=["uniform", "dense"],
                   help="'uniform': evenly spaced across video (default). "
                        "'dense': first N consecutive frames.")
    p.add_argument("--output", default="results/",
                   help="Output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_latents(
        video_path=args.video,
        n_frames=args.frames,
        model_id=args.vae,
        output_dir=args.output,
        frame_size=args.frame_size,
        sampling=args.sampling,
    )
