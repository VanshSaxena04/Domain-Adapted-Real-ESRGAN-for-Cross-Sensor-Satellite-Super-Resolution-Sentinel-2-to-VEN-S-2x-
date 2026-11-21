#cat << 'EOF' > eval_full.py
"""
Evaluation Script for Cross-Sensor Satellite Super-Resolution (Sentinel-2 -> VENuS)
Computes:
  1. PSNR  (Peak Signal-to-Noise Ratio, dB)        -> Higher is better (pixel accuracy)
  2. SSIM  (Structural Similarity Index)           -> Higher is better (structural fidelity)
  3. LPIPS (Learned Perceptual Image Patch Sim.)   -> Lower is better  (perceptual sharpness)
  4. SAM   (Spectral Angle Mapper, degrees)        -> Lower is better  (spectral/color fidelity)

Generates:
  - Terminal summary table (Table 1 for the paper)
  - Side-by-side RGB crops + pixel-wise residual error heatmaps in results/paper_figures/
"""

import os
import glob
import cv2
import torch
import lpips
import numpy as np
import matplotlib.pyplot as plt
from basicsr.metrics import calculate_psnr, calculate_ssim


# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================
HR_DIR = "../datasets/terrain/test_hr"       # 5m VENuS Ground Truth
LR_DIR = "../datasets/terrain/test_lr"       # 10m Sentinel-2 Input
VANILLA_DIR = "results/vanilla_x2"           # Untouched RealESRGAN_x2plus outputs
TERRAIN_DIR = "results/terrain_x2"           # Our fine-tuned Terrain Real-ESRGAN outputs
OUTPUT_FIG_DIR = "results/paper_figures"     # Where figures for the paper are saved

CROP_BORDER = 2                              # Standard 2px border crop for x2 SR evaluation
NUM_VISUAL_SAMPLES = 4                       # How many sample figures to export


# ==============================================================================
# 2. METRIC HELPER FUNCTIONS
# ==============================================================================
def prepare_lpips_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert an OpenCV BGR image [0, 255] into a normalized RGB tensor [-1, 1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
    return tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]
-

def compute_sam_degrees(pred_bgr: np.ndarray, target_bgr: np.ndarray, border: int = 2) -> float:
    """
    Compute Spectral Angle Mapper (SAM) in degrees between prediction and ground truth.
    Measures how faithfully RGB spectral ratios are preserved (0° = identical spectrum).
    """
    p = pred_bgr[border:-border, border:-border].astype(np.float64)
    t = target_bgr[border:-border, border:-border].astype(np.float64)

    dot_product = np.sum(p * t, axis=2)
    norm_product = np.maximum(np.linalg.norm(p, axis=2) * np.linalg.norm(t, axis=2), 1e-8)

    cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(cos_angle))
    return float(np.mean(angle_degrees))


def load_prediction(folder: str, stem: str, original_fname: str) -> np.ndarray:
    """Load model output image, handling Real-ESRGAN's default '_out.png' naming."""
    path_with_out = os.path.join(folder, f"{stem}_out.png")
    path_plain = os.path.join(folder, original_fname)
    target_path = path_with_out if os.path.exists(path_with_out) else path_plain

    img = cv2.imread(target_path)
    if img is None:
        raise FileNotFoundError(f"Could not read prediction image: {target_path}")
    return img


# ==============================================================================
# 3. FIGURE GENERATION (RGB STRIP + RESIDUAL ERROR HEATMAP)
# ==============================================================================
def save_comparison_figures(stem: str, bicubic: np.ndarray, vanilla: np.ndarray,
                            terrain: np.ndarray, hr: np.ndarray, save_dir: str) -> None:
    """Save a 128x128 center crop comparing RGB visual sharpness and pixel-wise error maps."""
    h, w, _ = hr.shape
    cy, cx = h // 2 - 64, w // 2 - 64

    crop_bic = bicubic[cy:cy + 128, cx:cx + 128]
    crop_van = vanilla[cy:cy + 128, cx:cx + 128]
    crop_ter = terrain[cy:cy + 128, cx:cx + 128]
    crop_hr  = hr[cy:cy + 128, cx:cx + 128]

    # A. Raw 4-panel horizontal RGB strip (Bicubic | Vanilla | Ours | Ground Truth)
    rgb_strip = np.hstack([crop_bic, crop_van, crop_ter, crop_hr])
    cv2.imwrite(os.path.join(save_dir, f"compare_{stem}.png"), rgb_strip)

    # B. 2-Row Publication Figure: Top = RGB Crops, Bottom = |Prediction - HR| Error Heatmaps
    err_bic = np.mean(np.abs(crop_bic.astype(np.float32) - crop_hr.astype(np.float32)), axis=2)
    err_van = np.mean(np.abs(crop_van.astype(np.float32) - crop_hr.astype(np.float32)), axis=2)
    err_ter = np.mean(np.abs(crop_ter.astype(np.float32) - crop_hr.astype(np.float32)), axis=2)
    vmax = max(np.percentile(err_van, 98), 25.0)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), dpi=300)
    titles = ["Bicubic x2", "Vanilla Real-ESRGAN", "Terrain Real-ESRGAN (Ours)", "VENuS 5m Ground Truth"]
    rgb_crops = [crop_bic, crop_van, crop_ter, crop_hr]

    for col in range(4):
        axes[0, col].imshow(cv2.cvtColor(rgb_crops[col], cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(titles[col], fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

    error_maps = [err_bic, err_van, err_ter]
    for col in range(3):
        im = axes[1, col].imshow(error_maps[col], cmap="inferno", vmin=0, vmax=vmax)
        axes[1, col].set_title(f"Abs. Error (MAE: {np.mean(error_maps[col]):.2f})", fontsize=9)
        axes[1, col].axis("off")

    axes[1, 3].axis("off")
    cbar = fig.colorbar(im, ax=axes[1, 3], fraction=0.45, pad=0.05)
    cbar.set_label("Pixel Error (0-255 scale)", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"heatmap_{stem}.png"), bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 4. MAIN EVALUATION PIPELINE
# ==============================================================================
def main():
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    lpips_model = lpips.LPIPS(net="alex")

    results = {
        "Bicubic x2": {"psnr": [], "ssim": [], "lpips": [], "sam": []},
        "Vanilla Real-ESRGAN": {"psnr": [], "ssim": [], "lpips": [], "sam": []},
        "Terrain Real-ESRGAN (Ours)": {"psnr": [], "ssim": [], "lpips": [], "sam": []},
    }

    hr_paths = sorted(glob.glob(os.path.join(HR_DIR, "*.png")))
    sample_indices = set(np.linspace(0, len(hr_paths) - 1, NUM_VISUAL_SAMPLES, dtype=int))
    print(f"Running full evaluation across {len(hr_paths)} test patches...\n")

    for idx, hr_path in enumerate(hr_paths):
        fname = os.path.basename(hr_path)
        stem = os.path.splitext(fname)[0]

        # Load Ground Truth, Low-Res Input, Bicubic Baseline, and Model Outputs
        hr = cv2.imread(hr_path)
        lr = cv2.imread(os.path.join(LR_DIR, fname))
        bicubic = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        vanilla = load_prediction(VANILLA_DIR, stem, fname)
        terrain = load_prediction(TERRAIN_DIR, stem, fname)

        hr_tensor = prepare_lpips_tensor(hr)

        # Evaluate all 3 methods on this patch
        candidates = [
            ("Bicubic x2", bicubic),
            ("Vanilla Real-ESRGAN", vanilla),
            ("Terrain Real-ESRGAN (Ours)", terrain),
        ]
        for method_name, pred in candidates:
            results[method_name]["psnr"].append(calculate_psnr(hr, pred, crop_border=CROP_BORDER))
            results[method_name]["ssim"].append(calculate_ssim(hr, pred, crop_border=CROP_BORDER))
            results[method_name]["sam"].append(compute_sam_degrees(pred, hr, border=CROP_BORDER))

            with torch.no_grad():
                score_lpips = lpips_model(hr_tensor, prepare_lpips_tensor(pred)).item()
                results[method_name]["lpips"].append(score_lpips)

        # Export visual comparison figures for selected patches
        if idx in sample_indices:
            save_comparison_figures(stem, bicubic, vanilla, terrain, hr, OUTPUT_FIG_DIR)

    # Print formatted Table 1
    print("=" * 82)
    print(f"TABLE 1: QUANTITATIVE BENCHMARK ON ESTUAMAR TEST SET (N = {len(hr_paths)})")
    print("=" * 82)
    print(f"{'Method':<28} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'LPIPS ↓':<10} | {'SAM (°) ↓':<10}")
    print("-" * 82)
    for method_name, scores in results.items():
        print(
            f"{method_name:<28} | "
            f"{np.mean(scores['psnr']):<12.4f} | "
            f"{np.mean(scores['ssim']):<10.4f} | "
            f"{np.mean(scores['lpips']):<10.4f} | "
            f"{np.mean(scores['sam']):<10.4f}"
        )
    print("=" * 82)
    print(f"Visual figures saved to: {OUTPUT_FIG_DIR}/")


if __name__ == "__main__":
    main()
EOF