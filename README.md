# Conditional Latent Residual Rank Analysis

A minimal, hackable experiment to test whether **Conditional Latent Residuals** occupy a lower-rank subspace than raw I-frame latents.

## Hypothesis

Given a sequence of video frame latents $Z_t$, we expect:

$$PR_{cond} < PR_{diff} < PR_{raw}$$

where $PR$ is the **Participation Ratio** (Effective Rank).

## Repo Structure

```
video_compression/
├── extract_latents.py   # Step 1: encode video frames → latent tensors
├── build_populations.py # Step 2: build Raw / Diff / Conditional-Residual matrices
├── spectral_analysis.py # Step 3: SVD, ESD, Effective Rank, Cumulative Variance
├── visualize_umap.py    # Step 4: UMAP scatter plot
├── run_all.py           # One-shot entry point
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Run everything end-to-end on a video file
python run_all.py --video path/to/video.mp4 --frames 300 --output results/
```

### Run stages individually

```bash
# 1. Extract latents (saves latents.npy)
python extract_latents.py --video path/to/video.mp4 --frames 300 --output results/

# 2. Build the three population matrices (saves raw.npy, diff.npy, cond.npy)
python build_populations.py --latents results/latents.npy --output results/

# 3. Spectral analysis (prints rank table, saves scree plot)
python spectral_analysis.py --data_dir results/ --output results/

# 4. UMAP visualization
python visualize_umap.py --data_dir results/ --output results/
```

## Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | — | Path to input video |
| `--frames` | 300 | Number of frames to process |
| `--vae` | `stabilityai/sd-vae-ft-mse` | HuggingFace VAE model ID |
| `--ridge_alpha` | 1.0 | Regularisation for Ridge predictor |
| `--output` | `results/` | Directory for all outputs |

## Outputs

- `results/latents.npy` — raw latent tensor `(T, C*H*W)`
- `results/raw.npy`, `diff.npy`, `cond.npy` — three population matrices
- `results/scree_plot.png` — log-scale normalised eigenvalue plot
- `results/umap_scatter.png` — UMAP 2-D projection
- `results/rank_table.txt` — printed rank metrics table
