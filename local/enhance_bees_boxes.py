#!/usr/bin/env python3
"""
enhance_bees_boxes.py
=====================
Given explicit bounding boxes (centre-x, centre-y, width, height in px),
enhances each bee crop from image.png using the SNN-based pipeline from
bee_enhance_pipeline.py, then composites the enhanced crops back into the
full image.

Outputs (pipeline_output/bee_boxes/):
  full_enhanced.png  – full image with bee regions replaced by enhanced crops
  closeups/          – individual enhanced closeup tiles (before | after)
"""

import cv2
import numpy as np
import os

# ── Bounding boxes (cx, cy, w, h in pixels) ──────────────────────────────────
BOXES = [
    {"id": "6",  "cx": 698.6832, "cy": 155.4951, "w": 68.5873,  "h": 35.2715},
    {"id": "7",  "cx": 499.4137, "cy": 184.8913, "w": 24.9665,  "h": 53.8188},
    {"id": "9",  "cx": 417.5387, "cy": 170.6139, "w": 40.8736,  "h": 54.8964},
    {"id": "A",  "cx": 439.5960, "cy": 244.6781, "w": 40.0531,  "h": 61.9492},
    {"id": "B",  "cx": 450.0869, "cy": 306.8462, "w": 56.8548,  "h": 43.6157},
    {"id": "E",  "cx": 408.0596, "cy": 295.9660, "w": 56.4007,  "h": 41.8635},
    {"id": "F",  "cx": 338.5173, "cy": 209.3316, "w": 42.4513,  "h": 56.2915},
    {"id": "G",  "cx": 299.0689, "cy": 227.8361, "w": 49.7653,  "h": 44.8852},
    {"id": "H",  "cx": 220.1654, "cy": 303.9561, "w": 60.9940,  "h": 29.5408},
    {"id": "I",  "cx": 210.5338, "cy": 151.7967, "w": 31.8033,  "h": 63.7259},
    {"id": "J",  "cx": 131.4415, "cy": 397.6121, "w": 52.3145,  "h": 43.9147},
    {"id": "K",  "cx": 207.2421, "cy": 476.9187, "w": 40.0929,  "h": 55.6373},
    {"id": "L",  "cx": 182.3758, "cy": 652.5979, "w": 63.1779,  "h": 36.5905},
    {"id": "M",  "cx": 264.4465, "cy": 676.5070, "w": 35.6506,  "h": 58.7877},
    {"id": "N",  "cx": 187.1942, "cy": 807.9960, "w": 64.5137,  "h": 45.0434},
    {"id": "P",  "cx": 161.9829, "cy": 871.8765, "w": 24.0494,  "h": 65.6730},
    {"id": "Q",  "cx": 116.5844, "cy": 874.5727, "w": 30.1725,  "h": 54.5331},
    {"id": "R",  "cx": 662.3459, "cy": 885.1192, "w": 55.0107,  "h": 63.7453},
    {"id": "S",  "cx": 901.6663, "cy": 931.4459, "w": 47.2018,  "h": 56.7823},
    {"id": "T",  "cx": 1084.1013,"cy": 897.8776, "w": 45.2750,  "h": 60.5246},
    {"id": "U",  "cx": 1405.6372,"cy": 851.3550, "w": 67.0916,  "h": 27.8182},
    {"id": "V",  "cx": 1490.1791,"cy": 995.4167, "w": 33.9667,  "h": 51.6288},
    {"id": "W",  "cx": 1531.1001,"cy": 859.8426, "w": 58.8560,  "h": 51.5981},
    {"id": "X",  "cx": 1544.8882,"cy": 829.9457, "w": 48.2896,  "h": 39.0887},
    {"id": "Y",  "cx": 1574.3290,"cy": 840.5309, "w": 32.3370,  "h": 57.3038},
    {"id": "Z",  "cx": 1536.7007,"cy": 764.3961, "w": 32.4370,  "h": 59.6153},
    {"id": "a",  "cx": 1455.0285,"cy": 727.1362, "w": 44.6536,  "h": 60.2358},
    {"id": "b",  "cx": 1348.3452,"cy": 684.1762, "w": 42.3778,  "h": 56.5119},
    {"id": "c",  "cx": 1123.0660,"cy": 773.2455, "w": 41.1716,  "h": 58.4039},
    {"id": "d",  "cx": 1055.0488,"cy": 673.1868, "w": 56.7201,  "h": 50.7981},
    {"id": "e",  "cx": 964.1472, "cy": 586.2161, "w": 41.0175,  "h": 55.4485},
    {"id": "g",  "cx": 1030.0212,"cy": 493.9281, "w": 61.4706,  "h": 20.7042},
    {"id": "h",  "cx": 1020.5784,"cy": 550.9692, "w": 35.2679,  "h": 52.2696},
    {"id": "i",  "cx": 1062.9333,"cy": 548.5026, "w": 44.6086,  "h": 49.8878},
    {"id": "j",  "cx": 1076.1729,"cy": 504.7910, "w": 32.5533,  "h": 53.1881},
    {"id": "k",  "cx": 1156.7184,"cy": 520.9732, "w": 35.2118,  "h": 64.7138},
    {"id": "l",  "cx": 1179.9356,"cy": 609.7221, "w": 74.9679,  "h": 24.1117},
    {"id": "m",  "cx": 1291.9639,"cy": 526.9759, "w": 64.9434,  "h": 35.0720},
    {"id": "n",  "cx": 1306.6735,"cy": 562.1861, "w": 64.0576,  "h": 31.9257},
    {"id": "p",  "cx": 1447.0445,"cy": 580.3706, "w": 70.5899,  "h": 27.9559},
    {"id": "q",  "cx": 1510.3168,"cy": 489.9435, "w": 64.8372,  "h": 51.5328},
    {"id": "r",  "cx": 1386.2929,"cy": 403.7301, "w": 55.0505,  "h": 45.1291},
    {"id": "s",  "cx": 1311.1475,"cy": 387.0733, "w": 73.0054,  "h": 36.7772},
    {"id": "t",  "cx": 1324.4672,"cy": 340.2603, "w": 51.5747,  "h": 36.7976},
    {"id": "u",  "cx": 1207.3037,"cy": 245.5461, "w": 51.0818,  "h": 49.6979},
    {"id": "v",  "cx": 1082.3940,"cy": 309.8690, "w": 39.7704,  "h": 61.7053},
    {"id": "w",  "cx": 1068.0910,"cy": 344.6440, "w": 63.4147,  "h": 28.6305},
    {"id": "x",  "cx": 931.5178, "cy": 369.7771, "w": 38.9795,  "h": 64.0545},
    {"id": "y",  "cx": 820.1288, "cy": 429.8200, "w": 45.8578,  "h": 54.7393},
    {"id": "z",  "cx": 854.0451, "cy": 353.5704, "w": 65.3424,  "h": 29.0652},
    {"id": "11", "cx": 720.9223, "cy": 525.8349, "w": 59.3030,  "h": 37.7477},
    {"id": "12", "cx": 662.2402, "cy": 560.7487, "w": 53.3585,  "h": 46.2956},
    {"id": "13", "cx": 748.8930, "cy": 667.6780, "w": 48.3641,  "h": 59.2030},
    {"id": "14", "cx": 543.3531, "cy": 766.4167, "w": 57.8988,  "h": 51.3940},
    {"id": "15", "cx": 501.8621, "cy": 433.3729, "w": 57.4783,  "h": 43.4534},
    {"id": "16", "cx": 491.7192, "cy": 379.8643, "w": 65.0826,  "h": 45.5168},
    {"id": "17", "cx": 201.4444, "cy": 59.4771,  "w": 56.0194,  "h": 48.3387},
    {"id": "18", "cx": 339.6965, "cy": 40.9344,  "w": 33.6489,  "h": 43.8934},
    {"id": "19", "cx": 409.4095, "cy": 56.9340,  "w": 44.3583,  "h": 59.2743},
    {"id": "1A", "cx": 495.9423, "cy": 75.3853,  "w": 42.5270,  "h": 51.0337},
    {"id": "1B", "cx": 453.0229, "cy": 25.1136,  "w": 21.0841,  "h": 43.9482},
    {"id": "1C", "cx": 587.1090, "cy": 87.2069,  "w": 53.5711,  "h": 41.4291},
    {"id": "1D", "cx": 560.0567, "cy": 51.7301,  "w": 68.9761,  "h": 24.6417},
    {"id": "1E", "cx": 784.9210, "cy": 53.6265,  "w": 54.2623,  "h": 28.4530},
    {"id": "1F", "cx": 860.6227, "cy": 47.2225,  "w": 54.1664,  "h": 41.8004},
    {"id": "1G", "cx": 1154.7328,"cy": 61.5358,  "w": 34.0488,  "h": 62.3544},
    {"id": "1H", "cx": 1115.8223,"cy": 104.2153, "w": 53.3278,  "h": 32.3636},
    {"id": "1I", "cx": 1156.8338,"cy": 160.5026, "w": 56.3839,  "h": 24.4407},
    {"id": "1L", "cx": 1205.2495,"cy": 157.2083, "w": 40.3220,  "h": 54.7033},
    {"id": "1M", "cx": 1209.3721,"cy": 96.8070,  "w": 58.5111,  "h": 35.0975},
    {"id": "1N", "cx": 1258.4528,"cy": 118.6292, "w": 66.0068,  "h": 27.1346},
    {"id": "1O", "cx": 1252.9827,"cy": 78.1352,  "w": 47.8828,  "h": 48.4586},
    {"id": "1P", "cx": 1311.9028,"cy": 63.8561,  "w": 31.2372,  "h": 49.5245},
    {"id": "1Q", "cx": 1330.2685,"cy": 154.0500, "w": 37.7915,  "h": 55.1477},
    {"id": "1R", "cx": 1307.1399,"cy": 254.6717, "w": 41.1286,  "h": 64.7890},
    {"id": "1S", "cx": 1372.5841,"cy": 324.8080, "w": 40.4054,  "h": 53.4820},
    {"id": "1U", "cx": 1429.6717,"cy": 339.7782, "w": 47.5367,  "h": 55.7133},
    {"id": "1V", "cx": 1460.2223,"cy": 376.8813, "w": 59.5028,  "h": 30.6419},
    {"id": "1W", "cx": 1528.6477,"cy": 650.9132, "w": 30.8156,  "h": 61.6689},
    {"id": "1X", "cx": 1530.5733,"cy": 571.0323, "w": 39.9313,  "h": 45.3637},
    {"id": "1Y", "cx": 1578.6584,"cy": 733.8233, "w": 50.1152,  "h": 30.9698},
    {"id": "1Z", "cx": 1572.0229,"cy": 707.1754, "w": 47.3733,  "h": 22.8469},
    {"id": "1a", "cx": 1560.1819,"cy": 626.9158, "w": 27.7858,  "h": 56.7358},
    {"id": "1b", "cx": 1457.4084,"cy": 643.3604, "w": 31.5262,  "h": 46.3576},
    {"id": "1c", "cx": 1481.6194,"cy": 626.6764, "w": 23.3302,  "h": 56.3279},
    {"id": "1d", "cx": 1506.0384,"cy": 618.5421, "w": 16.5988,  "h": 61.9397},
    {"id": "1e", "cx": 1576.7116,"cy": 272.2416, "w": 51.4861,  "h": 56.0583},
    {"id": "1f", "cx": 1403.7111,"cy": 104.0824, "w": 57.0094,  "h": 46.2270},
    {"id": "1g", "cx": 1470.9780,"cy": 102.8608, "w": 27.5790,  "h": 58.7084},
    {"id": "1h", "cx": 1376.4162,"cy": 120.7412, "w": 37.3374,  "h": 49.4996},
    {"id": "1i", "cx": 1511.5415,"cy": 140.2891, "w": 40.3906,  "h": 51.3943},
    {"id": "1j", "cx": 1556.5365,"cy": 53.2149,  "w": 19.9180,  "h": 57.0262},
    {"id": "1k", "cx": 930.1751, "cy": 279.2355, "w": 67.2640,  "h": 47.0586},
]

PAD      = 28   # px padding added around each bounding box
TILE_SZ  = 192  # side length for individual closeup output tiles

OUT_ROOT = "pipeline_output/bee_boxes"
os.makedirs(f"{OUT_ROOT}/closeups", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load image
# ─────────────────────────────────────────────────────────────────────────────

img_bgr  = cv2.imread("image.png")
if img_bgr is None:
    raise FileNotFoundError("image.png not found in working directory")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_h, img_w = img_gray.shape
print(f"[load] image.png  {img_w}×{img_h}  colour + grayscale ready")


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement primitives (ported from bee_enhance_pipeline.py)
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
    f    = gray_u8.astype(np.float32)
    blr  = cv2.GaussianBlur(f, (0, 0), sigma)
    hf   = f - blr
    gx   = cv2.Scharr(gray_u8, cv2.CV_32F, 1, 0)
    gy   = cv2.Scharr(gray_u8, cv2.CV_32F, 0, 1)
    emag = np.sqrt(gx ** 2 + gy ** 2)
    emax = np.percentile(emag, 95) + 1e-6
    emask = np.clip(emag / emax, 0, 1)
    emask = cv2.GaussianBlur(emask, (0, 0), 2.5)
    result = f + amount * hf + (edge_boost - amount) * emask * hf
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance(crop_gray, out_h, out_w):
    """
    Full enhancement chain — outputs back to original crop dimensions (out_h×out_w).
    """
    # ── Step 0: Upscale to working resolution (1.2× longer side) ──────────────
    scale = (max(out_h, out_w) * 1.2) / max(crop_gray.shape)
    scale = max(scale, 1.0)
    nw    = max(out_w, int(crop_gray.shape[1] * scale))
    nh    = max(out_h, int(crop_gray.shape[0] * scale))
    work  = cv2.resize(crop_gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # ── Step 1: Multi-Scale Retinex (25% blend) ────────────────────────────────
    msr  = multi_scale_retinex(work, sigmas=(15, 80, 250))
    work = np.clip(0.75 * work.astype(np.float32) +
                   0.25 * msr.astype(np.float32), 0, 255).astype(np.uint8)

    # ── Step 2: Adaptive gamma → target mean ~118 ──────────────────────────────
    mean_val = float(work.mean())
    if mean_val > 5:
        gamma = np.log(118.0 / 255.0) / np.log(max(mean_val / 255.0, 1e-6))
        gamma = float(np.clip(gamma, 0.5, 1.8))
        lut   = np.array([((i / 255.0) ** gamma) * 255.0
                          for i in range(256)], dtype=np.uint8)
        work  = cv2.LUT(work, lut)

    # ── Step 3: NL-means denoising ─────────────────────────────────────────────
    noise_est = float(cv2.Laplacian(work, cv2.CV_64F).std())
    nlm_h     = int(np.clip(noise_est * 0.35, 4, 9))
    work      = cv2.fastNlMeansDenoising(work, h=nlm_h,
                                          templateWindowSize=7,
                                          searchWindowSize=13)

    # ── Step 4: Two-pass adaptive CLAHE ────────────────────────────────────────
    std_val = float(work.std())
    clip1   = float(np.clip(1.8 * max(1.0, 8.0 / (std_val + 1e-3)), 1.2, 3.5))
    work    = cv2.createCLAHE(clipLimit=clip1, tileGridSize=(8, 8)).apply(work)
    work    = cv2.createCLAHE(clipLimit=1.0,   tileGridSize=(4, 4)).apply(work)

    # ── Step 5: SNN filter ─────────────────────────────────────────────────────
    f    = work.astype(np.float64) / 255.0
    f    = snn_filter(f, win_sz=3)
    work = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    # ── Step 6: Multi-scale Laplacian detail boost ─────────────────────────────
    img_f       = work.astype(np.float64)
    base_coarse = cv2.GaussianBlur(img_f, (0, 0), 5.0)
    base_fine   = cv2.GaussianBlur(img_f, (0, 0), 1.5)
    detail_mid  = base_fine - base_coarse
    detail_fine = img_f - base_fine
    img_f       = base_coarse + detail_mid * 1.35 + detail_fine * 1.10
    work        = np.clip(img_f, 0, 255).astype(np.uint8)

    # ── Step 7: Edge-adaptive USM ──────────────────────────────────────────────
    work = edge_adaptive_usm(work, amount=1.1, sigma=1.4, edge_boost=1.8)

    # ── Step 8: Percentile stretch ─────────────────────────────────────────────
    p_lo, p_hi = np.percentile(work, 1), np.percentile(work, 99)
    if p_hi - p_lo > 10:
        work = np.clip(
            (work.astype(np.float64) - p_lo) / (p_hi - p_lo) * 245.0 + 5.0,
            0, 255).astype(np.uint8)

    # ── Step 9: Resize back to output dimensions ───────────────────────────────
    return cv2.resize(work, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────────────────
# Build the composite output image
# The output will be a grayscale image with enhanced bee regions in-place,
# smoothly blended with a feathered alpha mask at each crop boundary.
# ─────────────────────────────────────────────────────────────────────────────

# Start from grayscale original; convert to float32 for blending
composite = img_gray.astype(np.float32)
blend_acc = np.zeros_like(composite)          # accumulated enhanced signal
blend_wt  = np.zeros_like(composite)          # accumulated blend weights

print(f"\nProcessing {len(BOXES)} bee bounding boxes …")

for idx, box in enumerate(BOXES):
    cx = box["cx"]
    cy = box["cy"]
    bw = box["w"]
    bh = box["h"]

    # Padded crop boundaries (clamped to image)
    x1 = max(0, int(round(cx - bw / 2)) - PAD)
    y1 = max(0, int(round(cy - bh / 2)) - PAD)
    x2 = min(img_w, int(round(cx + bw / 2)) + PAD)
    y2 = min(img_h, int(round(cy + bh / 2)) + PAD)

    cw = x2 - x1
    ch = y2 - y1
    if cw < 4 or ch < 4:
        continue

    crop = img_gray[y1:y2, x1:x2]
    enh  = enhance(crop, out_h=ch, out_w=cw)

    # Feathered weight mask: cosine taper over PAD pixels at all edges
    feather = min(PAD, cw // 4, ch // 4, 6)
    mask = np.ones((ch, cw), dtype=np.float32)
    for i in range(feather):
        w_edge = 0.5 * (1.0 - np.cos(np.pi * i / feather))
        mask[i,  :]  = np.minimum(mask[i,  :],  w_edge)
        mask[-1-i,:] = np.minimum(mask[-1-i,:], w_edge)
        mask[:,  i]  = np.minimum(mask[:,  i],  w_edge)
        mask[:,-1-i] = np.minimum(mask[:,-1-i], w_edge)

    blend_acc[y1:y2, x1:x2] += enh.astype(np.float32) * mask
    blend_wt [y1:y2, x1:x2] += mask

    # Save individual closeup tile (before | after)
    target = TILE_SZ
    bef_t  = cv2.resize(crop, (target, target), interpolation=cv2.INTER_LANCZOS4)
    aft_t  = cv2.resize(enh,  (target, target), interpolation=cv2.INTER_LANCZOS4)
    divider = np.full((target, 3), 60, dtype=np.uint8)
    header  = np.full((22, target * 2 + 3), 18, dtype=np.uint8)
    tile_img = np.vstack([header,
                          np.hstack([bef_t, divider, aft_t])])
    tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tile_bgr, "BEFORE",   (4,  16),            font, 0.38, (120,120,120), 1)
    cv2.putText(tile_bgr, "AFTER",    (target+7, 16),      font, 0.38, (100,220,100), 1)
    cv2.putText(tile_bgr, f"#{box['id']}", (target-28, 16), font, 0.38, (220,180,60),  1)
    cv2.imwrite(f"{OUT_ROOT}/closeups/bee_{box['id']}.png", tile_bgr)

    if (idx + 1) % 20 == 0 or (idx + 1) == len(BOXES):
        print(f"  {idx+1:3d}/{len(BOXES)}  id={box['id']:<4s}  crop={cw}×{ch}px")

# ── Composite: where we have blend weights, mix enhanced in; elsewhere keep original ──
mask_any = blend_wt > 0
composite[mask_any] = (
    blend_acc[mask_any] / blend_wt[mask_any] * 0.85 +
    composite[mask_any] * 0.15
)
result = np.clip(composite, 0, 255).astype(np.uint8)

out_path = f"{OUT_ROOT}/full_enhanced.png"
cv2.imwrite(out_path, result)
h_out, w_out = result.shape
print(f"\nComposite → {out_path}  ({w_out}×{h_out} px)")
print(f"Closeups  → {OUT_ROOT}/closeups/  ({len(BOXES)} tiles)")
print("Done.")
