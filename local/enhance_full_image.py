#!/usr/bin/env python3
"""
enhance_full_image.py
=====================
Applies the full SNN-based enhancement pipeline from bee_enhance_pipeline.py
to the entire image.png in one pass (no cropping, no bounding boxes).

Output → pipeline_output/full_image_enhance/full_enhanced.png
"""

import cv2
import numpy as np
import os
import time

OUT_DIR = "pipeline_output/full_image_enhance"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

img_bgr  = cv2.imread("image.png")
if img_bgr is None:
    raise FileNotFoundError("image.png not found")
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_h, img_w = img_gray.shape
print(f"[load] {img_w}×{img_h} px")


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement primitives
# ─────────────────────────────────────────────────────────────────────────────

def multi_scale_retinex(gray_u8, sigmas=(15, 80, 250)):
    img_f   = gray_u8.astype(np.float32) + 1.0
    log_img = np.log(img_f)
    retinex = np.zeros_like(img_f)
    w = 1.0 / len(sigmas)
    for sigma in sigmas:
        blur    = cv2.GaussianBlur(img_f, (0, 0), sigma)
        blur    = np.maximum(blur, 1.0)
        retinex += w * (log_img - np.log(blur))
    retinex -= retinex.min()
    if retinex.max() > 1e-6:
        retinex = retinex / retinex.max() * 255.0
    return np.clip(retinex, 0, 255).astype(np.uint8)


def snn_filter(img_f64, win_sz=3):
    pad = win_sz // 2
    h, w = img_f64.shape
    padded = np.pad(img_f64, pad, mode='edge')
    offsets_top, offsets_bot = [], []
    for dy in range(-pad, 0):
        for dx in range(-pad, pad + 1):
            offsets_top.append((dy, dx))
            offsets_bot.append((-dy, -dx))
    for dx in range(-pad, 0):
        offsets_top.append((0, dx))
        offsets_bot.append((0, -dx))
    centre = padded[pad:pad + h, pad:pad + w]
    acc    = centre.copy()
    count  = np.ones_like(centre)
    for (dy_t, dx_t), (dy_b, dx_b) in zip(offsets_top, offsets_bot):
        v_top  = padded[pad + dy_t:pad + dy_t + h, pad + dx_t:pad + dx_t + w]
        v_bot  = padded[pad + dy_b:pad + dy_b + h, pad + dx_b:pad + dx_b + w]
        chosen = np.where(np.abs(v_top - centre) <= np.abs(v_bot - centre), v_top, v_bot)
        acc   += chosen
        count += 1
    return acc / count


def edge_adaptive_usm(gray_u8, amount=1.1, sigma=1.4, edge_boost=1.8):
    f     = gray_u8.astype(np.float32)
    blr   = cv2.GaussianBlur(f, (0, 0), sigma)
    hf    = f - blr
    gx    = cv2.Scharr(gray_u8, cv2.CV_32F, 1, 0)
    gy    = cv2.Scharr(gray_u8, cv2.CV_32F, 0, 1)
    emag  = np.sqrt(gx ** 2 + gy ** 2)
    emax  = np.percentile(emag, 95) + 1e-6
    emask = np.clip(emag / emax, 0, 1)
    emask = cv2.GaussianBlur(emask, (0, 0), 2.5)
    result = f + amount * hf + (edge_boost - amount) * emask * hf
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Full-image enhancement (no upscale — already at native resolution)
# ─────────────────────────────────────────────────────────────────────────────

work = img_gray.copy()
t0 = time.time()

# Step 1: Multi-Scale Retinex (25% blend)
print("[1/8] Multi-Scale Retinex …", flush=True)
t = time.time()
msr  = multi_scale_retinex(work, sigmas=(15, 80, 250))
work = np.clip(0.75 * work.astype(np.float32) +
               0.25 * msr.astype(np.float32), 0, 255).astype(np.uint8)
print(f"      {time.time()-t:.1f}s")

# Step 2: Adaptive gamma
print("[2/8] Adaptive gamma …", flush=True)
t = time.time()
mean_val = float(work.mean())
if mean_val > 5:
    gamma = np.log(118.0 / 255.0) / np.log(max(mean_val / 255.0, 1e-6))
    gamma = float(np.clip(gamma, 0.5, 1.8))
    lut   = np.array([((i / 255.0) ** gamma) * 255.0
                      for i in range(256)], dtype=np.uint8)
    work  = cv2.LUT(work, lut)
    print(f"      gamma={gamma:.3f}  mean {mean_val:.1f} → {float(work.mean()):.1f}  {time.time()-t:.1f}s")

# Step 3: NL-means denoising
print("[3/8] NL-means denoising …", flush=True)
t = time.time()
noise_est = float(cv2.Laplacian(work, cv2.CV_64F).std())
nlm_h     = int(np.clip(noise_est * 0.35, 4, 9))
print(f"      noise_est={noise_est:.2f}  h={nlm_h}", flush=True)
work = cv2.fastNlMeansDenoising(work, h=nlm_h,
                                 templateWindowSize=7,
                                 searchWindowSize=13)
print(f"      {time.time()-t:.1f}s")

# Step 4: Two-pass adaptive CLAHE
print("[4/8] Two-pass CLAHE …", flush=True)
t = time.time()
std_val = float(work.std())
clip1   = float(np.clip(1.8 * max(1.0, 8.0 / (std_val + 1e-3)), 1.2, 3.5))
work    = cv2.createCLAHE(clipLimit=clip1, tileGridSize=(8, 8)).apply(work)
work    = cv2.createCLAHE(clipLimit=1.0,   tileGridSize=(4, 4)).apply(work)
print(f"      clip1={clip1:.2f}  {time.time()-t:.1f}s")

# Step 5: SNN filter
print("[5/8] SNN filter …", flush=True)
t = time.time()
f    = work.astype(np.float64) / 255.0
f    = snn_filter(f, win_sz=3)
work = np.clip(f * 255.0, 0, 255).astype(np.uint8)
print(f"      {time.time()-t:.1f}s")

# Step 6: Multi-scale Laplacian detail boost
print("[6/8] Laplacian detail boost …", flush=True)
t = time.time()
img_f       = work.astype(np.float64)
base_coarse = cv2.GaussianBlur(img_f, (0, 0), 5.0)
base_fine   = cv2.GaussianBlur(img_f, (0, 0), 1.5)
detail_mid  = base_fine - base_coarse
detail_fine = img_f - base_fine
img_f       = base_coarse + detail_mid * 1.35 + detail_fine * 1.10
work        = np.clip(img_f, 0, 255).astype(np.uint8)
print(f"      {time.time()-t:.1f}s")

# Step 7: Edge-adaptive USM
print("[7/8] Edge-adaptive USM …", flush=True)
t = time.time()
work = edge_adaptive_usm(work, amount=1.1, sigma=1.4, edge_boost=1.8)
print(f"      {time.time()-t:.1f}s")

# Step 8: Percentile stretch
print("[8/8] Percentile stretch …", flush=True)
t = time.time()
p_lo, p_hi = np.percentile(work, 1), np.percentile(work, 99)
if p_hi - p_lo > 10:
    work = np.clip(
        (work.astype(np.float64) - p_lo) / (p_hi - p_lo) * 245.0 + 5.0,
        0, 255).astype(np.uint8)
print(f"      {time.time()-t:.1f}s")

print(f"\nTotal: {time.time()-t0:.1f}s")

out_path = f"{OUT_DIR}/full_enhanced.png"
cv2.imwrite(out_path, work)
print(f"→ {out_path}  ({img_w}×{img_h} px)")
