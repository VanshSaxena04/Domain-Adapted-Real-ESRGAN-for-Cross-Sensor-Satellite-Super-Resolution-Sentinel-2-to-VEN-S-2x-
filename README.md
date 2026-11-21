Terrain-ESRGAN: Cross-Sensor Optical Satellite Super-Resolution ($\text{Sentinel-2} \rightarrow \text{VEN}\mu\text{S}$, $\times 2$)Domain-adaptive generative super-resolution for Earth Observation imagery: bridging the radiometric and perceptual domain gap between $10\text{ m}$ Sentinel-2 multispectral acquisitions and $5\text{ m}$ VEN$\mu$S reference scenes via transfer-learned RRDBNet architectures.Abstract & Research MotivationSingle-Image Super-Resolution (SISR) models trained on terrestrial consumer photography (e.g., DIV2K, Flickr2K, OutdoorSceneTraining) exhibit severe cross-domain degradation when applied zero-shot to orbital remote sensing imagery. Pre-trained perceptual and adversarial priors hallucinate high-frequency textures—such as masonry, foliage grain, and edge ringing—that violate physical surface reflectance and corrupt inter-band spectral ratios.This repository presents a rigorous, leakage-free domain adaptation pipeline built atop the Real-ESRGAN and BasicSR frameworks, evaluated on real, co-registered same-day orbital acquisitions from the SEN2VEN$\mu$S (ESTUAMAR coastal/agricultural site) benchmark:Input Sensor: Copernicus Sentinel-2 L2A ($10\text{ m}$ Ground Sampling Distance, $128 \times 128\text{ px}$ patches)Target Reference: CNES/ISA VEN$\mu$S ($5\text{ m}$ Ground Sampling Distance, $256 \times 256\text{ px}$ patches)Headline Finding: Domain-adapting RealESRGAN_x2plus for $10,000$ iterations recovers $+1.45\text{ dB}$ PSNR, $+0.0986$ SSIM, and a $38.8\%$ reduction in LPIPS over the untouched baseline, while achieving the lowest Spectral Angle Mapper ($\text{SAM} = 5.1642^\circ$) across all tested methods—surpassing both off-the-shelf GAN priors ($8.3524^\circ$) and Bicubic interpolation ($5.3097^\circ$).Quantitative Benchmark Results (ESTUAMAR, $N = 130$)All evaluations are conducted strictly on the temporally held-out final acquisition date ($N = 130$ paired patches) using a standard $2$-pixel border crop (crop_border=2).MethodSupervision / AdaptationPSNR (dB) ↑SSIM ↑LPIPS (AlexNet) ↓SAM (∘) ↓Bicubic $\times 2$Analytical Kernel (Non-Learned)36.83220.90780.11875.3097Vanilla RealESRGAN_x2plusZero-Shot Natural Image Priors34.66080.79740.21648.3524Terrain Real-ESRGAN (Ours)Domain-Adapted ($10\text{k}$ Iters, ESTUAMAR)36.11300.89600.13255.1642Visual Performance Charts1. Spectral Angle Mapper ($\text{SAM} \,[^\circ] \downarrow$) — Radiometric FidelityLower values indicate superior preservation of physical RGB surface reflectance ratios.PlaintextBicubic x2                  |█████████████████████▏                    | 5.3097°
Vanilla Real-ESRGAN         |█████████████████████████████████▍        | 8.3524°  (+3.04° spectral distortion)
Terrain Real-ESRGAN (Ours)  |████████████████████▋                     | 5.1642°  (Best — beats Bicubic & Vanilla)
                            0°       2°       4°       6°       8°      10°
2. Perceptual Distance ($\text{LPIPS} \downarrow$) — Deep Feature RealismLower values indicate closer perceptual alignment in AlexNet feature space.PlaintextBicubic x2                  |███████████████████                       | 0.1187
Vanilla Real-ESRGAN         |██████████████████████████████████▌       | 0.2164  (Severe texture hallucination)
Terrain Real-ESRGAN (Ours)  |█████████████████████▏                    | 0.1325  (-38.8% error vs. Vanilla)
                            0.00     0.05     0.10     0.15     0.20     0.25
3. Peak Signal-to-Noise Ratio ($\text{PSNR dB} \uparrow$) & Structural Similarity ($\text{SSIM} \uparrow$)PlaintextPSNR (dB) ↑
Bicubic x2                  |████████████████████████████████████▊     | 36.8322 dB
Vanilla Real-ESRGAN         |██████████████████████████████████▋       | 34.6608 dB
Terrain Real-ESRGAN (Ours)  |████████████████████████████████████      | 36.1130 dB (+1.45 dB over Vanilla)
                            0        10       20       30       35       40

SSIM ↑
Bicubic x2                  |████████████████████████████████████▎     | 0.9078
Vanilla Real-ESRGAN         |███████████████████████████████▉          | 0.7974
Terrain Real-ESRGAN (Ours)  |███████████████████████████████████▊      | 0.8960 (+0.0986 over Vanilla)
                            0.0      0.2      0.4      0.6      0.8      1.0
Key Scientific Takeaways (The Perception–Distortion–Spectral Trade-Off)Domain Gap Quantification: Off-the-shelf RealESRGAN_x2plus drops $-2.17\text{ dB}$ in PSNR and $+3.04^\circ$ in SAM relative to Bicubic interpolation, confirming that terrestrial photographic priors inject radiometric shifts and high-frequency artifacts into orbital scenes.Spectral Preservation Without Blur: While Bicubic interpolation maximizes pixel-wise PSNR/SSIM by smoothing spatial gradients, our domain-adapted model achieves the lowest spectral angle error ($5.1642^\circ$) across all methods while recovering $98.7\%$ of Bicubic's structural similarity ($0.8960$ vs. $0.9078$).System Architecture & End-to-End PipelineCode snippetflowchart LR
    subgraph Data["1. Radiometric & Leakage-Free Data Pipeline"]
        A["SEN2VENµS Raw Tensors (.pt)<br/>16-bit Reflectance (ESTUAMAR)"] --> B["Drop NIR & Reorder<br/>[B2, B3, B4] → [R, G, B]"]
        B --> C["Linear Radiometric Stretch<br/>[0.0, 0.3] → [0, 255] uint8"]
        C --> D{"Temporal Split<br/>by Acquisition Date"}
        D -->|"Earlier Dates"| E["train_hr (5m VENµS, 256×256)"]
        D -->|"Final Date (N=130)"| F["test_lr (10m S2) & test_hr (5m VENµS)"]
    end

    subgraph Train["2. Domain Adaptation (Real-ESRGAN x2)"]
        E --> G["Second-Order On-the-Fly<br/>Degradation Pipeline"]
        G --> H["Generator G: RRDBNet (23 Blocks)<br/>PixelUnshuffle (scale=2, 12ch)<br/>Init: RealESRGAN_x2plus.pth"]
        H --> I["Composite Loss<br/>L1 + VGG-19 Perceptual + GAN"]
        I --> J["Discriminator D: UNetDiscriminatorSN<br/>EMA Decay = 0.999 | 10,000 Iters"]
    end

    subgraph Eval["3. Multi-Metric & Qualitative Evaluation"]
        F --> K["Inference (--outscale 2 --fp32)"]
        J --> K
        K --> L["Quantitative Suite (eval_full.py)<br/>PSNR | SSIM | LPIPS | SAM (°)"]
        K --> M["Qualitative Suite<br/>128×128 Crops & Inferno Error Maps"]
    end
Qualitative Visual Proof & Residual Error HeatmapsOur evaluation pipeline exports both $128 \times 128$ center RGB comparison strips (compare_<patch>.png) and 2-row publication figures (heatmap_<patch>.png) displaying pixel-wise Mean Absolute Error ($\vert{}\hat{I} - I_{\text{HR}}\vert{}$) locked to a unified inferno scale:Figure FileLayoutVisual Diagnostic Purposeresults/paper_figures/compare_<stem>.pngBicubic | Vanilla | Ours | VENµS 5m GTDirect inspection of linear boundaries (field borders, roads, coastlines) and texture sharpness.results/paper_figures/heatmap_<stem>.pngRow 1: RGB CropsRow 2: $\Vert{}\hat{I} - I_{\text{HR}}\Vert{}$ (inferno)Exposes high-amplitude hallucinated noise in Vanilla Real-ESRGAN over homogeneous water and terrain, contrasted against error suppression in Terrain Real-ESRGAN.(Place your selected results/paper_figures/heatmap_*.png preview here once pushed to GitHub:)Markdown![Qualitative Comparison and Residual Error Heatmap](Real-ESRGAN/results/paper_figures/heatmap_test_00000.png)
Repository StructurePlaintextResearch/
├── datasets/
│   └── terrain/
│       ├── train_hr/                     # 5m VENµS training patches (256x256 PNG, earlier dates)
│       ├── test_hr/                      # 5m VENµS held-out ground truth (N=130, final acquisition date)
│       ├── test_lr/                      # 10m Sentinel-2 held-out inputs (N=130, 128x128 PNG)
│       └── meta_info_terrain.txt         # Clean BasicSR training manifest
├── experiments/
│   └── pretrained_models/
│       ├── RealESRGAN_x2plus.pth         # Pre-trained x2 generator initialization
│       ├── RealESRGAN_x2plus_netD.pth    # Pre-trained x2 spectral-normalized U-Net discriminator
│       └── net_g_10000.pth               # Domain-adapted generator weights (10,000 iterations)
└── Real-ESRGAN/                          # Core training, inference & evaluation codebase
    ├── BasicSR/                          # Underlying PyTorch image restoration engine
    ├── realesrgan/                       # Real-ESRGAN model, dataset & degradation modules
    ├── options/
    │   └── finetune_terrain_x2.yml       # Domain adaptation hyperparameters & loss weights
    ├── scripts/
    │   └── generate_meta_info.py         # Dataset manifest generator
    ├── inference_realesrgan.py           # CLI inference runner
    ├── eval_full.py                      # Full 4-metric benchmark & error heatmap generator
    └── results/
        ├── vanilla_x2/                   # Zero-shot baseline outputs (N=130)
        ├── terrain_x2/                   # Domain-adapted model outputs (N=130)
        └── paper_figures/                # Publication-ready RGB strips & residual heatmaps
Methodology & Experimental Configuration1. Radiometric Calibration & Leakage-Free SplittingBand Mapping: Raw 16-bit PyTorch tensors (.pt) from the SEN2VEN$\mu$S ESTUAMAR site were stripped of the Near-Infrared (NIR) channel and reordered from [B2, B3, B4] to standard [R, G, B].Dynamic Range Scaling: Digital numbers were divided by $10,000$ to recover physical top-of-canopy/surface reflectance $\rho \in [0, 1]$, then linearly stretched from $\rho \in [0.0, 0.3]$ into 8-bit $[0, 255]$ PNGs to preserve radiometric contrast across low-albedo water and vegetated terrain.Strict Temporal Split: Rather than random patch sampling (which leaks identical atmospheric and solar geometry across train/test splits), all patches from the final acquisition date ($N = 130$) were isolated strictly for testing.2. Model & Training Hyperparameters (finetune_terrain_x2.yml)ParameterValue / SpecificationArchitectural & Training RationaleGenerator ($G$)RRDBNet (num_block=23, num_feat=64, num_grow_ch=32, scale=2)Uses pixel_unshuffle ($3 \rightarrow 12\text{ ch}$) without BatchNorm to prevent spatial artifacts.Discriminator ($D$)UNetDiscriminatorSN (num_feat=64, skip_connection=True)Spectral-normalized U-Net providing dense pixel-level realism feedback.InitializationRealESRGAN_x2plus.pth & RealESRGAN_x2plus_netD.pthTransfer learning from large-scale natural image representations.Training Schedule$10,000\text{ iterations}$ on NVIDIA T4 GPUConverges domain adaptation without overfitting single-site geography.Batch & Queue Sizebatch_size_per_gpu = 8, queue_size = 160queue_size calibrated as an exact multiple of batch size ($160 \bmod 8 = 0$).StabilizationExponential Moving Average (ema_decay = 0.999)Smooths generator weight oscillations under adversarial updates.Composite Loss$L_1$ Pixel Loss + VGG-19 Perceptual Loss + GAN LossBalances pixel/spectral accuracy with crisp high-frequency boundary synthesis.Reproducibility & Quickstart1. Environment SetupTo avoid Python 3.13 locals() scoping issues in legacy setup.py scripts and torchvision.transforms.functional_tensor deprecations, use an isolated Python 3.10 Conda environment:Bashconda create -n realesargan python=3.10 -y
conda activate realesargan

cd Real-ESRGAN
pip install -r requirements.txt
pip install lpips scikit-image matplotlib
python setup.py develop
(If running on newer torchvision >= 0.17, patch the legacy import in basicsr/data/degradations.py:)Bashsed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' \
  $(python -c "import basicsr, os; print(os.path.join(os.path.dirname(basicsr.__file__), 'data/degradations.py'))")
2. Dataset Manifest & TrainingBash# Generate dataset manifest
python scripts/generate_meta_info.py \
  --input ../datasets/terrain/train_hr \
  --root ../datasets/terrain \
  --meta_info ../datasets/terrain/meta_info_terrain.txt

# Launch domain-adaptive fine-tuning
python realesrgan/train.py -opt options/finetune_terrain_x2.yml --auto_resume
3. Inference (Zero-Shot Baseline vs. Domain-Adapted Model)Bash# 1. Vanilla Real-ESRGAN x2 baseline
python inference_realesrgan.py \
  -n RealESRGAN_x2plus \
  -i ../datasets/terrain/test_lr \
  -o results/vanilla_x2 \
  --outscale 2 \
  --fp32

# 2. Domain-adapted Terrain Real-ESRGAN (Ours)
python inference_realesrgan.py \
  -n RealESRGAN_x2plus \
  --model_path ../experiments/pretrained_models/net_g_10000.pth \
  -i ../datasets/terrain/test_lr \
  -o results/terrain_x2 \
  --outscale 2 \
  --fp32
4. Quantitative Benchmark & Figure GenerationBashpython eval_full.py
Outputs the complete 4-metric Table 1 (PSNR, SSIM, LPIPS, SAM) to stdout and exports $128 \times 128$ center RGB strips and inferno residual error heatmaps to results/paper_figures/.Limitations & Future WorkRGB Spectral Restriction: Current VGG-19 perceptual loss constraints required excluding the $10\text{ m}$ Near-Infrared (NIR) band. Extending RRDBNet to $4$-channel (RGB + NIR) inputs with a spectral-aware perceptual branch will enable direct $5\text{ m}$ Normalized Difference Vegetation Index ($\text{NDVI}$) synthesis.Single-Site Training Scope: Training was localized to the ESTUAMAR coastal/agricultural site; evaluating zero-shot cross-biome generalization across arid, alpine, and dense urban SEN2VEN$\mu$S sites remains a priority next step.Synthetic vs. Real Sensor Degradation: Training leveraged Real-ESRGAN's second-order synthetic degradation model on $5\text{ m}$ VEN$\mu$S patches; direct paired supervision (RealESRGANPairedDataset) using co-registered $10\text{ m}$ Sentinel-2 inputs offers a promising ablation direction.References & Software AttributionsThis work builds directly upon open-source architectures, restoration toolkits, perceptual metrics, and calibrated Earth Observation datasets. If you use this code or experimental protocol in academic work, please cite the foundational works below:Real-ESRGAN (Core Architecture & Second-Order Degradation Pipeline):Wang, X., Xie, L., Dong, C., & Shan, Y. (2021). Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pp. 1905–1914. GitHubESRGAN / RRDBNet Backbone:Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., & Loy, C. C. (2018). ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. In Proceedings of the European Conference on Computer Vision Workshops (ECCVW).BasicSR (Training Engine, PSNR & SSIM Metrics):Wang, X., Xie, L., Yu, K., Chan, K. C., Loy, C. C., & Dong, C. (2022). BasicSR: Open Source Image and Video Restoration Toolbox. GitHubSEN2VEN$\mu$S Dataset (Cross-Sensor Sentinel-2 & VEN$\mu$S Benchmark):Michel, J., Vinasco-Salinas, J., Inglada, J., & Hagolle, O. (2022). SEN2VEN$\mu$S, a Dataset for the Training of Sentinel-2 Super-Resolution Algorithms. Data, 7(7), 96. https://doi.org/10.3390/data7070096LPIPS (Learned Perceptual Image Patch Similarity Metric):Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 586–595. GitHubSpectral Angle Mapper (SAM):Kruse, F. A., Lefkoff, A. B., Boardman, J. W., Heidebrecht, K. B., Shapiro, A. T., Barloon, P. J., & Goetz, A. F. H. (1993). The Spectral Image Processing System (SIPS)—Interactive Visualization and Analysis of Imaging Spectrometer Data. Remote Sensing of Environment, 44(2–3), 145–163.Core Scientific Python Ecosystem:PyTorch (torch, torchvision), OpenCV (opencv-python), NumPy, SciPy/scikit-image, and Matplotlib.CitationCode snippet@misc{saxena2026terrainesrgan,
  author       = {Saxena, Vansh},
  title        = {Terrain-ESRGAN: Cross-Sensor Optical Satellite Super-Resolution via Domain-Adapted Real-ESRGAN},
  year         = {2026},
  howpublished = {\url{https://github.com/<your-username>/<your-repo-name>}},
  note         = {Cross-sensor x2 super-resolution benchmark on SEN2VENuS (ESTUAMAR)}
}
