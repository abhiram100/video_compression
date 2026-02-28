"""
run_all.py – One-shot entry point.

Runs all pipeline stages in sequence:
  1. extract_latents
  2. build_populations
  3. spectral_analysis
  4. visualize_umap
  5. evaluate_predictor
  6. evaluate_keyframe_gop
  7. stride_sensitivity

Usage:
  python run_all.py --video path/to/video.mp4 --frames 300 --output results/
"""

import argparse
import os

from extract_latents import extract_latents
from build_populations import build_populations
from spectral_analysis import run_spectral_analysis
from visualize_umap import run_umap
from evaluate_predictor import run_evaluation
from evaluate_keyframe_gop import run_gop_evaluation
from stride_sensitivity import run_stride_study


def parse_args():
    p = argparse.ArgumentParser(
        description="Full pipeline: video → latents → spectral analysis → UMAP.")
    p.add_argument("--video", required=True,
                   help="Path to input video file.")
    p.add_argument("--frames", type=int, default=300,
                   help="Number of frames to extract (default: 300).")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse",
                   help="HuggingFace VAE model ID.")
    p.add_argument("--frame_size", type=int, default=512,
                   help="Resize frames to this square size before encoding.")
    p.add_argument("--sampling", default="uniform", choices=["uniform", "dense"],
                   help="'uniform': evenly spaced across video (default). "
                        "'dense': first N consecutive frames.")
    p.add_argument("--ridge_alpha", type=float, default=1.0,
                   help="Ridge regularisation strength.")
    p.add_argument("--pca_dims", type=int, default=256,
                   help="PCA dims used before Ridge predictor.")
    p.add_argument("--n_svd_components", type=int, default=None,
                   help="Max SVD components for spectral analysis.")
    p.add_argument("--umap_max_points", type=int, default=2000,
                   help="Max points per population for UMAP.")
    p.add_argument("--n_viz", type=int, default=20,
                   help="Number of side-by-side comparison panels to save.")
    p.add_argument("--latent_shape", default="4,64,64",
                   help="C,H,W of one latent (default: 4,64,64 for SD VAE @ 512px).")
    p.add_argument("--no_fid", action="store_true",
                   help="Skip FID computation in evaluation step.")
    p.add_argument("--gop_sizes", default="1,3,5,10",
                   help="Comma-separated GOP sizes for keyframe eval (default: 1,3,5,10).")
    p.add_argument("--decode_batch", type=int, default=1,
                   help="VAE decode batch size (lower if OOM).")
    p.add_argument("--strides", default="1,2,4,8,16",
                   help="Comma-separated strides for step 7 stride study (default: 1,2,4,8,16).")
    p.add_argument("--pca_dims_sweep", default="16,32,64,128,256",
                   help="Comma-separated PCA dims for step 7 stride study "
                        "(default: 16,32,64,128,256).")
    p.add_argument("--skip_stride_study", action="store_true",
                   help="Skip step 7 (stride × PCA sensitivity study).")
    p.add_argument("--output", default="results/",
                   help="Output directory for all artefacts.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("STEP 1/6  Extract latents")
    print("=" * 60)
    latents_path = extract_latents(
        video_path=args.video,
        n_frames=args.frames,
        model_id=args.vae,
        output_dir=args.output,
        frame_size=args.frame_size,
        sampling=args.sampling,
    )

    print("\n" + "=" * 60)
    print("STEP 2/7  Build populations")
    print("=" * 60)
    gop_sizes = [int(x) for x in args.gop_sizes.split(",") if x.strip()]
    build_populations(
        latents_path=latents_path,
        output_dir=args.output,
        ridge_alpha=args.ridge_alpha,
        pca_dims=args.pca_dims,
        gop_sizes=gop_sizes,
    )

    print("\n" + "=" * 60)
    print("STEP 3/7  Spectral analysis")
    print("=" * 60)
    run_spectral_analysis(
        data_dir=args.output,
        output_dir=args.output,
        n_components=args.n_svd_components,
    )

    print("\n" + "=" * 60)
    print("STEP 4/7  UMAP visualisation")
    print("=" * 60)
    run_umap(
        data_dir=args.output,
        output_dir=args.output,
        max_points=args.umap_max_points,
    )

    print("\n" + "=" * 60)
    print("STEP 5/7  Evaluate predictor (SSIM / PSNR / FID)")
    print("=" * 60)
    C, H, W = map(int, args.latent_shape.split(","))
    run_evaluation(
        data_dir=args.output,
        output_dir=args.output,
        model_id=args.vae,
        latent_shape=(C, H, W),
        n_viz=args.n_viz,
        compute_fid_flag=not args.no_fid,
        decode_batch=args.decode_batch,
    )

    print("\n" + "=" * 60)
    print("STEP 6/7  Keyframe GOP evaluation (memory vs quality)")
    print("=" * 60)
    run_gop_evaluation(
        data_dir=args.output,
        output_dir=args.output,
        gop_sizes=gop_sizes,
        model_id=args.vae,
        latent_shape=(C, H, W),
        n_viz=args.n_viz,
        decode_batch=args.decode_batch,
        compute_fid_flag=not args.no_fid,
    )

    if not args.skip_stride_study:
        print("\n" + "=" * 60)
        print("STEP 7/7  Stride × PCA sensitivity study")
        print("=" * 60)
        strides_list   = [int(s) for s in args.strides.split(",") if s.strip()]
        pca_dims_sweep = [int(p) for p in args.pca_dims_sweep.split(",") if p.strip()]
        run_stride_study(
            video_path=args.video,
            train_latents_path=latents_path,
            output_dir=os.path.join(args.output, "stride_study"),
            strides=strides_list,
            pca_dims_list=pca_dims_sweep,
            gop_size=8,
            start_frame=0,
            ridge_alpha=args.ridge_alpha,
            model_id=args.vae,
            latent_shape=(C, H, W),
            frame_size=args.frame_size,
            compute_fid_flag=not args.no_fid,
        )
    else:
        print("\n[run_all] Skipping step 7 (--skip_stride_study).")

    print("\n" + "=" * 60)
    print("Done! Outputs in:", os.path.abspath(args.output))
    print("  latents.npy, raw.npy, diff.npy, cond.npy, predictor.pkl")
    print("  scree_plot.png, umap_scatter.png, rank_table.txt")
    print("  eval_metrics.txt, eval_frames/frame_XXXX.png")
    print("  gop_eval/gop_metrics.txt, gop_eval/gop_memory_quality.png")
    print("  gop_eval/panels_gop{K}/frame_NNNN.png")
    print("  stride_study/heatmap_*.png, stride_study/stride_metrics.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
