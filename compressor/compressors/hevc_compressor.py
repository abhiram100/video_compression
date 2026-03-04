import subprocess
import shutil
import numpy as np
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
        # Temporary directory for frames during the ffmpeg pipe
        self.temp_workspace = Path("temp_hevc_workspace")

    def compress(self, frames):
        """
        Converts a list of PIL frames into a single HEVC bitstream (mp4).
        Returns the raw bytes of the video file to keep it in memory during the batch.
        """
        if self.temp_workspace.exists():
            shutil.rmtree(self.temp_workspace)
        self.temp_workspace.mkdir(parents=True)

        # 1. Save frames as numbered PNGs for FFmpeg
        for i, frame in enumerate(frames):
            frame.save(self.temp_workspace / f"frame_{i:04d}.png")

        output_mp4 = self.temp_workspace / "temp_compressed.mp4"
        
        # 2. Run FFmpeg: libx265
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", str(self.temp_workspace / "frame_%04d.png"),
            "-c:v", "libx265",
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", "yuv420p", # Essential for standard player compatibility
            str(output_mp4)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Read the resulting binary file into memory
        with open(output_mp4, "rb") as f:
            video_data = f.read()
            
        return video_data

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
        if self.temp_workspace.exists():
            shutil.rmtree(self.temp_workspace)
        self.temp_workspace.mkdir(parents=True)

        # Write bytes to temp file for ffmpeg to read
        temp_mp4 = self.temp_workspace / "to_decompress.mp4"
        with open(temp_mp4, "wb") as f:
            f.write(video_bytes)

        # Extract frames to PNG
        out_pattern = self.temp_workspace / "out_%04d.png"
        cmd = ["ffmpeg", "-y", "-i", str(temp_mp4), str(out_pattern)]
        subprocess.run(cmd, check=True, capture_output=True)

        # Load back to PIL
        frame_paths = sorted(self.temp_workspace.glob("out_*.png"))
        frames = [Image.open(p).copy() for p in frame_paths]
        
        return frames