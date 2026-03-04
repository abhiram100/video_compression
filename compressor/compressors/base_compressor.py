from abc import ABC, abstractmethod
from pathlib import Path

from compressor.data.image_utils import read_image, write_image


class BaseCompressor(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def compress(self, frames):
        """
        Compress a list of frames.

        Args:
            frames (list): A list of frames to be compressed.

        Returns:
            list: A list of compressed frames.
        """
        pass

    def write_compressed_data(self, compressed_frames, output_dir, batch_index: int):
        """
        Write compressed frames for a single batch to disk as PNG images.
        Each frame is saved to <output_dir>/batch_<batch_index>/frame_<N>.png.

        Subclasses may override this to use a different storage format.
        """
        batch_dir = Path(output_dir) / f"batch_{batch_index:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(compressed_frames):
            write_image(frame, batch_dir / f"frame_{i:04d}.png")

    def read_compressed_data(self, input_dir, batch_index: int):
        """
        Read compressed frames for a single batch from disk.
        Expects frames saved by write_compressed_data at
        <input_dir>/batch_<batch_index>/frame_<N>.png.

        Subclasses may override this to match a custom storage format.

        Returns:
            list[PIL.Image]: Frames in order.
        """
        batch_dir = Path(input_dir) / f"batch_{batch_index:06d}"
        paths = sorted(batch_dir.glob("frame_*.png"))
        return [read_image(p) for p in paths]

    @abstractmethod
    def decompress(self, compressed_frames):
        """
        Decompress a list of compressed frames.

        Args:
            compressed_frames (list): A list of compressed frames to be decompressed.

        Returns:
            list: A list of decompressed frames.
        """
        return compressed_frames