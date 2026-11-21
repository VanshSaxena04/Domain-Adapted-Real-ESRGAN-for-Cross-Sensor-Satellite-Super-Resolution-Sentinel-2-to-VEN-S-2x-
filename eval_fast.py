import os, glob, cv2
import numpy as np
from basicsr.metrics import calculate_psnr, calculate_ssim

hr_dir = "../datasets/terrain/test_hr"
lr_dir = "../datasets/terrain/test_lr"
vanilla_dir = "results/vanilla_x2"
terrain_dir = "results/terrain_x2"
fig_dir = "results/paper_figures"
os.makedirs(fig_dir, exist_ok=True)

metrics = {
    "Bicubic x2": {"psnr": [], "ssim": []},
    "Vanilla Real-ESRGAN": {"psnr": [], "ssim": []},
    "Terrain Real-ESRGAN (Ours)": {"psnr": [], "ssim": []},
}

hr_paths = sorted(glob.glob(f"{hr_dir}/*.png"))
print(f"Evaluating {len(hr_paths)} test patches...\n")

for idx, hr_path in enumerate(hr_paths):
    fname = os.path.basename(hr_path)
    stem = os.path.splitext(fname)[0]

    hr = cv2.imread(hr_path)
    lr = cv2.imread(os.path.join(lr_dir, fname))
    bicubic = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

    van_path = os.path.join(vanilla_dir, f"{stem}_out.png")
    ter_path = os.path.join(terrain_dir, f"{stem}_out.png")
    if not os.path.exists(van_path): van_path = os.path.join(vanilla_dir, fname)
    if not os.path.exists(ter_path): ter_path = os.path.join(terrain_dir, fname)

    vanilla = cv2.imread(van_path)
    terrain = cv2.imread(ter_path)

    for name, pred in [("Bicubic x2", bicubic), ("Vanilla Real-ESRGAN", vanilla), ("Terrain Real-ESRGAN (Ours)", terrain)]:
        metrics[name]["psnr"].append(calculate_psnr(hr, pred, crop_border=2))
        metrics[name]["ssim"].append(calculate_ssim(hr, pred, crop_border=2))

    if idx in np.linspace(0, len(hr_paths) - 1, 4, dtype=int):
        h, w, _ = hr.shape
        cy, cx = h // 2 - 64, w // 2 - 64
        strip = np.hstack([
            bicubic[cy:cy+128, cx:cx+128],
            vanilla[cy:cy+128, cx:cx+128],
            terrain[cy:cy+128, cx:cx+128],
            hr[cy:cy+128, cx:cx+128],
        ])
        cv2.imwrite(os.path.join(fig_dir, f"compare_{stem}.png"), strip)

print("=== TABLE 1: QUANTITATIVE RESULTS ON ESTUAMAR TEST SET ===")
print(f"{'Method':<28} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10}")
print("-" * 56)
for name, vals in metrics.items():
    print(f"{name:<28} | {np.mean(vals['psnr']):<12.4f} | {np.mean(vals['ssim']):<10.4f}")
print(f"\nSaved 4 side-by-side crops (Bicubic | Vanilla | Ours | Ground Truth) to {fig_dir}/")
