import subprocess
import tempfile
from pathlib import Path
from PIL import Image
from .base_compressor import BaseCompressor

class HEVCCompressor(BaseCompressor):
    """
    Standard HEVC (H.265) baseline using FFmpeg.
    """
    def __init__(self, crf=28, preset="medium", fps=30):
        super().__init__()
        self.crf = crf
        self.preset = preset
        self.fps = fps

    def compress(self, frames):
        """
        Converts a list of PIL frames into a single HEVC bitstream (mp4).
        Returns the raw bytes of the video file to keep it in memory during the batch.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            # 1. Save frames as numbered PNGs for FFmpeg
            for i, frame in enumerate(frames):
                frame.save(tmp / f"frame_{i:04d}.png")

            output_mp4 = tmp / "temp_compressed.mp4"

            # 2. Run FFmpeg: libx265
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.fps),
                "-i", str(tmp / "frame_%04d.png"),
                "-c:v", "libx265",
                "-crf", str(self.crf),
                "-preset", self.preset,
                "-pix_fmt", "yuv420p",
                str(output_mp4)
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            with open(output_mp4, "rb") as f:
                return f.read()

    def write_compressed_data(self, compressed_bytes, output_dir, batch_index: int):
        """Writes the binary mp4 to the batch directory."""
        batch_dir = Path(output_dir) / f"batch_{batch_index:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = batch_dir / "video.mp4"
        with open(output_file, "wb") as f:
            f.write(compressed_bytes)

    def read_compressed_data(self, input_dir, batch_index: int):
        """Reads the binary mp4 back from disk."""
        batch_dir = Path(input_dir) / f"batch_{batch_index:06d}"
        video_file = batch_dir / "video.mp4"
        
        with open(video_file, "rb") as f:
            return f.read()

    def compressed_batch_size_bytes(self, input_dir, batch_index: int) -> int:
        return (Path(input_dir) / f"batch_{batch_index:06d}" / "video.mp4").stat().st_size

    def decompress(self, video_bytes):
        """
        Takes the binary bytes, writes to a temp file, and uses FFmpeg
        to extract the frames back into PIL images.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            temp_mp4 = tmp / "to_decompress.mp4"
            with open(temp_mp4, "wb") as f:
                f.write(video_bytes)

            out_pattern = tmp / "out_%04d.png"
            cmd = ["ffmpeg", "-y", "-i", str(temp_mp4), str(out_pattern)]
            subprocess.run(cmd, check=True, capture_output=True)

            frame_paths = sorted(tmp.glob("out_*.png"))
            return [Image.open(p).copy() for p in frame_paths]