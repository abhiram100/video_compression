import numpy as np


def mb_from_bytes(num_bytes: int) -> float:
    """Convert bytes to megabytes."""
    return num_bytes / (1024 * 1024)


def compute_frame_stats(gt_frame, pred_frame) -> dict:
    """
    Compute quality metrics between a GT and a predicted frame.

    Args:
        gt_frame:   PIL Image (RGB)
        pred_frame: PIL Image (RGB)

    Returns:
        dict with keys: psnr, mse
    """
    gt = np.array(gt_frame, dtype=np.float32)
    pred = np.array(pred_frame, dtype=np.float32)

    mse = float(np.mean((gt - pred) ** 2))
    psnr = float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")

    return {"psnr": psnr, "mse": mse}

