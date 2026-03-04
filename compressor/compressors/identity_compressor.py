from pathlib import Path

from compressor.compressors.base_compressor import BaseCompressor
from compressor.data.image_utils import read_image, write_image


class IdentityCompressor(BaseCompressor):
    """Pass-through compressor – frames are written as PNGs and read back unchanged."""

    def compress(self, frames):
        return frames

    def write_compressed_data(self, compressed_frames, output_dir, batch_index: int):
        batch_dir = Path(output_dir) / f"batch_{batch_index:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(compressed_frames):
            write_image(frame, batch_dir / f"frame_{i:04d}.png")

    def read_compressed_data(self, input_dir, batch_index: int):
        batch_dir = Path(input_dir) / f"batch_{batch_index:06d}"
        paths = sorted(batch_dir.glob("frame_*.png"))
        return [read_image(p) for p in paths]

    def decompress(self, compressed_frames):
        return compressed_frames