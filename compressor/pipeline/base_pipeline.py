import logging
import time
from pathlib import Path
from tqdm import tqdm

from compressor.data.video_reader import VideoReader
from compressor.data.video_writer import VideoWriter
from compressor.data.image_utils import read_image, write_image
from compressor.compressors.base_compressor import BaseCompressor
from compressor.pipeline.measurement_utils import compute_frame_stats, mb_from_bytes
from compressor.compressors import *


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

    @property
    def _gt_reader(self) -> VideoReader:
        """Lazy VideoReader over the full source video for GT frame extraction."""
        if not hasattr(self, "_gt_reader_instance"):
            self._gt_reader_instance = VideoReader(
                self.input_video_path, batch_size=1, full_batches_only=False
            )
        return self._gt_reader_instance

    def _ensure_gt_frame(self, frame_idx: int) -> Path:
        """
        Ensure a single GT frame is inflated to disk and return its path.
        Only writes the file if it doesn't already exist.
        """
        d = self.gt_frames_dir
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"frame_{frame_idx:06d}.png"
        if not path.exists():
            frame = self._gt_reader.get_frame(frame_idx)  # RGB numpy
            write_image(frame, path)
        return path

    def _load_gt_frame(self, frame_idx: int):
        """Inflate (if needed) and load a single GT frame as a PIL Image."""
        path = self._ensure_gt_frame(frame_idx)
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

    def measure_statistics(self, n_frames: int | None = None, seed: int = 0) -> dict:
        """
        Compute quality metrics between GT and decompressed frames.

        Args:
            n_frames: If given, evaluate on a random subset of this many frames.
                      If None, evaluate all output frames.
            seed:     Random seed for reproducible subset selection.

        GT frames are inflated on demand – only the required frames are written,
        and any that already exist on disk are reused.
        """
        import random

        output_paths = sorted(self.output_frames_dir.glob("frame_*.png"))
        if not output_paths:
            raise RuntimeError("No output frames found – run decompress_video() first.")

        if n_frames is not None and n_frames < len(output_paths):
            rng = random.Random(seed)
            output_paths = sorted(rng.sample(output_paths, n_frames))
            logger.info("Evaluating on %d / %d frames (seed=%d).", n_frames, len(output_paths), seed)
        else:
            logger.info("Evaluating on all %d frames.", len(output_paths))

        stats_list = []
        for out_path in tqdm(output_paths, desc="Measuring statistics", unit="frame"):
            frame_idx = int(out_path.stem.split("_")[1])
            gt_frame = self._load_gt_frame(frame_idx)
            pred_frame = read_image(out_path)
            stats_list.append(compute_frame_stats(gt_frame, pred_frame))

        # Aggregate quality metrics
        keys = stats_list[0].keys()
        aggregated = {k: sum(s[k] for s in stats_list) / len(stats_list) for k in keys}

        # Memory: raw uncompressed vs compressed for the same evaluated frames
        evaluated_indices = [int(p.stem.split("_")[1]) for p in output_paths]
        raw_bytes = sum(
            self._ensure_gt_frame(idx).stat().st_size for idx in evaluated_indices
        )
        evaluated_batches = set(idx // self.reader.batch_size for idx in evaluated_indices)
        compressed_bytes = sum(
            self.compressor.compressed_batch_size_bytes(self.compressed_data_dir, batch_idx)
            for batch_idx in evaluated_batches
        )
        aggregated["raw_mb"] = mb_from_bytes(raw_bytes)
        aggregated["compressed_mb"] = mb_from_bytes(compressed_bytes)
        aggregated["compression_ratio"] = (
            raw_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")
        )

        for k, v in aggregated.items():
            logger.info("%s: %.4f", k, v)
        return aggregated


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    parser = argparse.ArgumentParser(description="Run the video compression pipeline.")
    parser.add_argument("input_video", type=str, help="Path to input video file.")
    parser.add_argument("output_dir", type=str, help="Directory to write outputs.")
    parser.add_argument("--start_time_s", type=float, default=0.0, help="Start time in seconds for processing a video segment.")
    parser.add_argument("--end_time_s", type=float, default=None, help="End time in seconds for processing a video segment.")
    parser.add_argument("--n_frames", type=int, default=50, help="Evaluate on a random subset of N frames (default: all).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subset selection.")
    parser.add_argument("--only_eval", action="store_true", help="Only run the evaluation phase (assumes compressed data and output frames already exist).")
    args = parser.parse_args()

    pipeline = BasePipeline(
        args.input_video,
        args.output_dir,
        compressor=VAECompressor(),
        start_time_s=args.start_time_s,
        end_time_s=args.end_time_s,
    )
    if not args.only_eval:
        pipeline.compress_video()
        pipeline.decompress_video()
    pipeline.measure_statistics(n_frames=args.n_frames, seed=args.seed)
