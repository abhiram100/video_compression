import logging
import time
from pathlib import Path
from tqdm import tqdm

from compressor.data.video_reader import VideoReader
from compressor.data.video_writer import VideoWriter
from compressor.data.image_utils import read_image, write_image, bgr_to_pil, image_pixel_bytes
from compressor.compressors.base_compressor import BaseCompressor
from compressor.pipeline.measurement_utils import compute_frame_stats, mb_from_bytes
from compressor.compressors.identity_compressor import IdentityCompressor

logger = logging.getLogger(__name__)


class BasePipeline:
    def __init__(
        self,
        input_video_path: str,
        output_video_dir: str,
        compressor: BaseCompressor,
        batch_size: int = 4,
        start_time_s: float = 0.0,
        end_time_s: float = None,
    ):
        self.input_video_path = Path(input_video_path)
        output_video_dir = Path(output_video_dir)
        self.reader = VideoReader(
            input_video_path,
            batch_size=batch_size,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        self.writer = VideoWriter(
            output_video_dir / "output.mp4", fps=self.reader.get_frame_rate()
        )
        self.compressed_data_dir = output_video_dir / "compressed_data"
        self.output_frames_dir = output_video_dir / "output_frames"
        self.compressor = compressor

    # ------------------------------------------------------------------
    # GT frame cache
    # ------------------------------------------------------------------

    @property
    def gt_frames_dir(self) -> Path:
        """<video_dir>/<video_filename>.frames  e.g. /data/clip.mp4.frames"""
        return self.input_video_path.parent / (self.input_video_path.name + ".frames")

    def _ensure_gt_frames(self) -> None:
        """Inflate the source video to PNG frames on disk if not already done."""
        d = self.gt_frames_dir
        if d.exists() and any(d.iterdir()):
            logger.debug("GT frames already exist at %s, skipping inflation.", d)
            return

        logger.info("Inflating GT frames to %s ...", d)
        d.mkdir(parents=True, exist_ok=True)
        cap_reader = VideoReader(
            self.input_video_path, batch_size=1, full_batches_only=False
        )
        for frame_idx in tqdm(
            range(cap_reader.frame_count),
            desc="Inflating GT frames",
            unit="frame",
        ):
            frame = cap_reader.get_frame(frame_idx)  # BGR numpy
            write_image(bgr_to_pil(frame), d / f"frame_{frame_idx:06d}.png")
        logger.info("Wrote %d GT frames to %s.", cap_reader.frame_count, d)

    def _load_gt_frame(self, frame_idx: int):
        """Load a single GT frame from the cache as a PIL Image."""

        path = self.gt_frames_dir / f"frame_{frame_idx:06d}.png"
        if not path.exists():
            raise FileNotFoundError(f"GT frame not found: {path}")
        return read_image(path)

    # ------------------------------------------------------------------
    # Phase 1 – compress
    # ------------------------------------------------------------------

    def compress_video(self):
        """Stream through the video batch-by-batch and write compressed data."""
        self.compressed_data_dir.mkdir(parents=True, exist_ok=True)
        n_batches = len(self.reader)
        logger.info("Starting compression: %d batches.", n_batches)
        t0 = time.perf_counter()
        for i in tqdm(range(n_batches), desc="Compressing", unit="batch"):
            batch_frames = self.reader[i]
            compressed_frames = self.compressor.compress(batch_frames)
            self.compressor.write_compressed_data(
                compressed_frames, self.compressed_data_dir, batch_index=i
            )
        elapsed = time.perf_counter() - t0
        logger.info("compress_video done in %.3fs (%d batches).", elapsed, n_batches)

    # ------------------------------------------------------------------
    # Phase 2 – decompress
    # ------------------------------------------------------------------

    def decompress_video(self):
        """Decompress all batches, write the output video, and save output frames."""
        self.output_frames_dir.mkdir(parents=True, exist_ok=True)
        n_batches = len(self.reader)
        logger.info("Starting decompression: %d batches.", n_batches)
        frame_idx = 0
        t0 = time.perf_counter()
        for i in tqdm(range(n_batches), desc="Decompressing", unit="batch"):
            compressed_frames = self.compressor.read_compressed_data(
                self.compressed_data_dir, batch_index=i
            )
            decompressed_frames = self.compressor.decompress(compressed_frames)
            self.writer.add_frames(decompressed_frames)
            for frame in decompressed_frames:
                write_image(
                    frame, self.output_frames_dir / f"frame_{frame_idx:06d}.png"
                )
                frame_idx += 1
        elapsed = time.perf_counter() - t0
        logger.info("decompress_video done in %.3fs (%d frames).", elapsed, frame_idx)
        logger.info("Writing output video ...")
        self.writer.write_video()
        logger.info("Output video written to %s.", self.writer.output_path)

    # ------------------------------------------------------------------
    # Phase 3 – statistics
    # ------------------------------------------------------------------

    def measure_statistics(self) -> dict:
        """
        Pull GT / processed frame pairs one at a time and compute stats.
        Inflates GT frames first if the cache directory does not exist.
        Returns a dict of aggregated metrics.
        """
        self._ensure_gt_frames()

        output_paths = sorted(self.output_frames_dir.glob("frame_*.png"))
        if not output_paths:
            raise RuntimeError("No output frames found – run decompress_video() first.")

        logger.info("Computing frame-level statistics over %d frames.", len(output_paths))
        stats_list = []
        for out_path in tqdm(output_paths, desc="Measuring statistics", unit="frame"):
            frame_idx = int(out_path.stem.split("_")[1])
            gt_frame = self._load_gt_frame(frame_idx)
            pred_frame = read_image(out_path)
            stats_list.append(compute_frame_stats(gt_frame, pred_frame))

        # Aggregate quality metrics
        keys = stats_list[0].keys()
        aggregated = {k: sum(s[k] for s in stats_list) / len(stats_list) for k in keys}

        # Memory: raw uncompressed video vs compressed data on disk
        raw_bytes = sum(
            image_pixel_bytes(read_image(p))
            for p in sorted(self.gt_frames_dir.glob("frame_*.png"))
        )
        compressed_bytes = sum(
            f.stat().st_size for f in self.compressed_data_dir.iterdir() if f.is_file()
        )
        aggregated["raw_mb"] = mb_from_bytes(raw_bytes)
        aggregated["compressed_mb"] = mb_from_bytes(compressed_bytes)
        aggregated["compression_ratio"] = (
            raw_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")
        )

        for k, v in aggregated.items():
            print(f"{k}: {v:.4f}")
        return aggregated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the video compression pipeline.")
    parser.add_argument("input_video", type=str, help="Path to input video file.")
    parser.add_argument("output_dir", type=str, help="Directory to write outputs.")
    parser.add_argument("--start_time_s", type=float, default=0.0, help="Start time in seconds for processing a video segment.")
    parser.add_argument("--end_time_s", type=float, default=None, help="End time in seconds for processing a video segment.")
    args = parser.parse_args()

    pipeline = BasePipeline(
        args.input_video,
        args.output_dir,
        compressor=IdentityCompressor(),
        start_time_s=args.start_time_s,
        end_time_s=args.end_time_s,
    )
    pipeline.compress_video()
    pipeline.decompress_video()
    pipeline.measure_statistics()
