# PyramidPix2pix — Smoke Test Documentation

---

## Model Info

- **Model name:** PyramidPix2pix
- **Upstream repo URL:** https://github.com/bupt-ai-cz/BCI
- **Upstream last commit date:** 3/9/2024
- **Paper / citation:** Liu et al., "BCI: Breast Cancer Immunohistochemical Image Generation Through Pyramid Pix2pix", CVPR Workshops 2022
- **MIST dataset citation:** Li et al., "Adaptive supervised PatchNCE loss for learning H&E-to-IHC stain translation with inconsistent groundtruth image pairs", MICCAI 2023, pp. 632–641, Springer
- **Paired or unpaired assumption:** Paired
- **Intended staining task:** H&E to IHC (BCI dataset); generalised to H&E → HER2 / ER / Ki67 / PR IHC (MIST dataset)

---

## Environment Claimed by Authors

- **Python version:** >= 3.6
- **PyTorch version:** not specified
- **CUDA version:** not specified
- **OS:** Linux
- **Installation method:** pip
- **Requirements file present:** `requirements.txt`
- **Pretrained weights available:** yes — trained models for BCI and LLVIP datasets on GitHub Releases

---

## Environment Actually Used

- **Python version:** 3.11
- **PyTorch version:** 2.11.0 (standard macOS build, no CUDA)
- **CUDA version:** N/A — CPU only
- **OS:** macOS (Apple Silicon M2)
- **Date tested:** 2026-04-02
- **Hardware:** Apple M2 (CPU, `--gpu_ids -1`)

---

## Base Packages

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Deep learning framework — model training, inference, transforms |
| `dominate` | Generates HTML pages for visualising training results |
| `kornia` | Differentiable image processing (used in pyramid loss computation) |
| `opencv-contrib-python` | Image reading, writing, and stitching in dataset preparation scripts |
| `scikit-image` | PSNR and SSIM metric calculation in `evaluate.py` |
| `tqdm` | Progress bars during dataset preparation and evaluation |
| `visdom` | Optional live training visualisation server (disabled with `--display_id -1`) |
| `numpy` | Array operations, used throughout the pipeline |

Exact working package versions are in `PyramidPix2pix/requirements.txt`.

---

## Dataset Preparation

PyramidPix2pix is a paired model — it expects images as a single side-by-side file (left = H&E input, right = IHC target).

- **BCI download:** see `download_dataset.md` in the repo root
- **MIST download:** https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/MIST-HER2.md

### BCI

- **Format expected by model:** side-by-side stitched pairs under `datasets/BCI/train/` and `datasets/BCI/test/`
- **Conversion applied:**

```bash
python datasets/combine_A_and_B.py \
  --fold_A ./datasets/BCI_raw/HE \
  --fold_B ./datasets/BCI_raw/IHC \
  --fold_AB ./datasets/BCI \
  --no_multiprocessing
```

- **Final folder layout:**

```
datasets/BCI/
    train/
        00000_train_1+.png
        ...
    test/
        00000_test_1+.png
        ...
```

### MIST

MIST has a different folder structure (`trainA/trainB/valA/valB`) so a dedicated script was written.

- **Conversion applied:**

```bash
python datasets/combine_MIST.py --stain HER2
python datasets/combine_MIST.py --stain ER
python datasets/combine_MIST.py --stain Ki67
python datasets/combine_MIST.py --stain PR
```

- **Output folders:** `datasets/MIST_HER2/`, `datasets/MIST_ER/`, `datasets/MIST_Ki67/`, `datasets/MIST_PR/`

---

## Smoke Test Commands

### BCI

```bash
# Train
python train.py \
  --dataroot ./datasets/BCI \
  --name smoke_test \
  --gpu_ids -1 \
  --batch_size 1 \
  --preprocess crop --crop_size 256 \
  --n_epochs 1 --n_epochs_decay 0 \
  --display_id -1 \
  --save_epoch_freq 1

# Test
python test.py \
  --dataroot ./datasets/BCI \
  --name smoke_test \
  --gpu_ids -1 \
  --num_test 5

# Evaluate
python evaluate.py --result_path ./results/smoke_test
```

### MIST (repeat for each stain)

```bash
# Train
python train.py \
  --dataroot ./datasets/MIST_HER2 \
  --name smoke_test_HER2 \
  --gpu_ids -1 \
  --batch_size 1 \
  --preprocess crop --crop_size 256 \
  --n_epochs 1 --n_epochs_decay 0 \
  --display_id -1 \
  --save_epoch_freq 1

# Test
python test.py \
  --dataroot ./datasets/MIST_HER2 \
  --name smoke_test_HER2 \
  --gpu_ids -1 \
  --num_test 5

# Evaluate
python evaluate.py --result_path ./results/smoke_test_HER2
```

Replace `MIST_HER2` / `smoke_test_HER2` with `MIST_ER`, `MIST_Ki67`, `MIST_PR` for other stains.

**Expected outputs:**
- Training ends with `End of epoch 1 / 1`
- Checkpoint saved to `checkpoints/{name}/latest_net_G.pth`
- Inference saves 3 files per image to `results/{name}/test_latest/images/`: `real_A`, `fake_B`, `real_B`
- Evaluate prints average PSNR and SSIM scores

---

## Smoke Test Results

| Dataset | Stitch | Train | Test | Evaluate |
|---|---|---|---|---|
| BCI | ✓ | ✓ | ✓ | ✓ |
| MIST HER2 | ✓ | ✓ | ✓ | ✓ |
| MIST ER | ✓ | ✓ | ✓ | ✓ |
| MIST Ki67 | ✓ | ✓ | ✓ | ✓ |
| MIST PR | ✓ | ✓ | ✓ | ✓ |

---

## Changes Made to Original Code

No model architecture, loss functions, or training logic was changed. All changes are infrastructure fixes to make the code run outside the original Linux + CUDA environment.

| File | Change | Reason                                                                          |
|---|---|---------------------------------------------------------------------------------|
| `datasets/combine_A_and_B.py` | Filter `os.listdir()` to only include directories | macOS `.DS_Store` files caused a crash                                          |
| `datasets/combine_MIST.py` | New script created | MIST folder structure is incompatible with `combine_A_and_B.py`                 |
| `models/pix2pix_model.py` lines 161, 172, 183 | `.to(self.opt.gpu_ids[0])` → `.to(self.device)` | Hardcoded CUDA index crashes on CPU (sobel pattern)                             |
| `models/pix2pix_model.py` lines 297, 301 | `.to(self.opt.gpu_ids[0])` → `.to(self.device)` | Hardcoded CUDA index crashes on CPU (fft pattern)                               |
| `models/pix2pix_model.py` line 322 | `.cuda()` → `.to(self.device)` | Hardcoded CUDA crashes on CPU (conv pattern)                                    |
| `models/pix2pix_model.py` `frequency_division()` | Rewrote using `torch.fft.fft2` / `torch.fft.ifft2` | `torch.rfft` / `torch.irfft` removed in PyTorch 1.8+                            |
| `evaluate.py` line 30 | `multichannel=True` → `channel_axis=-1` | Parameter removed in scikit-image 0.19+                                         |
| `.gitignore` | Created from scratch | Repo had none — prevents committing venv, checkpoints, datasets, macOS metadata |

---

## Known Broken Features

These optional loss patterns (`--pattern` flag) remain broken and were not fixed.

| Pattern | Problem | What's needed to fix |
|---|---|---|
| `fft` | `weight_low_L1` / `weight_high_L1` CLI options missing from `train_options.py` | Add missing arguments — incomplete implementation by original authors |


Default training (`--pattern L1_L2_L3_L4`) is unaffected by these.

---

## HPC Readiness Notes

| Parameter | Phase 1 (macOS CPU) | Restore to (HPC) |
|---|---|---|
| `--gpu_ids` | `-1` (CPU) | `0` (or multi-GPU) |
| `--crop_size` | `256` | Remove flag — use full `1024×1024` |
| `--preprocess` | `crop` | `none` (original default) |
| `--n_epochs` | `1` | per benchmark protocol |
| `--n_epochs_decay` | `0` | per benchmark protocol |
| `--batch_size` | `1` | per benchmark protocol |
| `--display_id` | `-1` (disabled) | `1` (if visdom server available) |
| Dataset size | 5 images (smoke test) | Full dataset |

---

## Warnings Observed (Non-Critical)

### 1. LR scheduler order
```
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`
```
**Location:** `train.py` line 48 — `update_learning_rate()` is called at the start of each epoch, before `optimize_parameters()` which contains `optimizer.step()`. The scheduler step in `base_model.py` line 123 is where it physically fires, but the root cause is the call order in `train.py`.  
**Impact:** First learning rate value is skipped. Harmless for smoke test. Fix by moving `update_learning_rate()` to after the inner training loop.

### 2. Visdom SSL certificates
```
ERROR: certificate verify failed: unable to get local issuer certificate
```
**Location:** macOS Python SSL configuration  
**Impact:** Visdom server fails to load. Use `--display_id -1` to disable. Fix with `/Applications/Python\ 3.11/Install\ Certificates.command` if needed.

---

## Summary

**Overall result: PASS**

PyramidPix2pix smoke test completed on 2026-04-02. The full pipeline (stitch → train → test → evaluate) ran successfully on macOS M2 (CPU) for both BCI and all four MIST stains (HER2, ER, Ki67, PR). Eight infrastructure fixes were applied to resolve macOS and modern PyTorch/scikit-image compatibility issues — no model logic was changed. Frozen environment committed. Ready for HPC migration.