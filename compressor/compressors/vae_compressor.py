import numpy as np
import torch
from pathlib import Path
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms

from compressor.compressors.base_compressor import BaseCompressor


class VAECompressor(BaseCompressor):
    """
    Compressor that uses a pretrained VAE to encode each frame into a latent
    vector as the compressed representation, then decodes back to a PIL Image.

    The latent is stored as a float32 .npy file per frame.
    Decoded frames are resized back to the original input resolution.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/sd-vae-ft-mse",
        encode_size: int = 512,
    ):
        super().__init__()
        self.model_id = model_id
        self.encode_size = encode_size
        self._vae: AutoencoderKL | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def vae(self) -> AutoencoderKL:
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(self.model_id)
            self._vae = self._vae.to(self.device).eval()
        return self._vae

    # ------------------------------------------------------------------
    # Encode / decode helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → normalised tensor (1, 3, encode_size, encode_size) in [-1, 1]."""
        transform = transforms.Compose([
            transforms.Resize((self.encode_size, self.encode_size)),
            transforms.ToTensor(),           # [0, 1]
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])
        return transform(image.convert("RGB")).unsqueeze(0)

    @torch.no_grad()
    def _encode(self, image: Image.Image) -> np.ndarray:
        """Returns the posterior mean as a float32 numpy array (C, H, W)."""
        tensor = self._to_tensor(image).to(self.device)
        z = self.vae.encode(tensor).latent_dist.mean  # (1, C, H, W)
        return z.squeeze(0).cpu().float().numpy()     # (C, H, W)

    @torch.no_grad()
    def _decode(self, latent: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
        """
        Decodes a (C, H, W) float32 latent back to a PIL Image and
        resizes to original_size (W, H).
        """
        z = torch.from_numpy(latent).unsqueeze(0).to(self.device)  # (1, C, H, W)
        decoded = self.vae.decode(z).sample                         # (1, 3, H, W) in [-1, 1]
        decoded = decoded.squeeze(0).clamp(-1, 1)                   # (3, H, W)
        decoded = (decoded + 1.0) / 2.0                             # [0, 1]
        decoded = (decoded * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8
        img = Image.fromarray(decoded, mode="RGB")
        return img.resize(original_size, Image.LANCZOS)

    # ------------------------------------------------------------------
    # BaseCompressor interface
    # ------------------------------------------------------------------

    def compress(self, frames: list[Image.Image]) -> list[dict]:
        """
        Encode each frame to its VAE latent.

        Returns a list of dicts with keys:
          - 'latent': np.ndarray (C, H, W) float32
          - 'original_size': (W, H) tuple – the original PIL image size
        """
        compressed = []
        for frame in frames:
            compressed.append({
                "latent": self._encode(frame),
                "original_size": frame.size,  # PIL size is (W, H)
            })
        return compressed

    def write_compressed_data(self, compressed_frames: list[dict], output_dir, batch_index: int):
        """Save each latent as a .npy file alongside a metadata .txt for original_size."""
        batch_dir = Path(output_dir) / f"batch_{batch_index:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(compressed_frames):
            np.save(batch_dir / f"frame_{i:04d}.npy", item["latent"])
            w, h = item["original_size"]
            (batch_dir / f"frame_{i:04d}.size").write_text(f"{w},{h}")

    def read_compressed_data(self, input_dir, batch_index: int) -> list[dict]:
        """Load latents and original sizes from disk."""
        batch_dir = Path(input_dir) / f"batch_{batch_index:06d}"
        items = []
        for npy_path in sorted(batch_dir.glob("frame_*.npy")):
            size_path = npy_path.with_suffix(".size")
            w, h = map(int, size_path.read_text().split(","))
            items.append({
                "latent": np.load(npy_path),
                "original_size": (w, h),
            })
        return items

    def compressed_batch_size_bytes(self, input_dir, batch_index: int) -> int:
        """Sum the .npy file sizes (the actual compressed data; .size files are negligible)."""
        batch_dir = Path(input_dir) / f"batch_{batch_index:06d}"
        return sum(p.stat().st_size for p in batch_dir.glob("frame_*.npy"))

    def decompress(self, compressed_frames: list[dict]) -> list[Image.Image]:
        """Decode each latent back to a PIL Image at its original resolution."""
        return [
            self._decode(item["latent"], item["original_size"])
            for item in compressed_frames
        ]
