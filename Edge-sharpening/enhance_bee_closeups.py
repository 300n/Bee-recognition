"""
Enhance clarity on bee close-up crops from 0007.png  –  v2
==========================================================
Pipeline (based on snnV2.m / Bakker edge-preserving filtering):

  1. Adaptive gamma         – lift shadows / normalise brightness per crop
  2. Non-local means        – light sensor-noise removal (texture-preserving)
  3. CLAHE (tuned)          – smooth local contrast boost (no tile artifacts)
  4. SNN filter (win 3)     – gentle Bakker-style edge-preserving refinement
  5. Multi-scale detail     – Laplacian decomposition, boost mid-frequencies
  6. Adaptive sharpening    – edge-weighted unsharp mask (sharp edges, quiet flats)
  7. Percentile stretch     – full dynamic-range utilisation

Outputs in  output/0007/bee_closeups_v2/
"""

import cv2
import numpy as np
import os, json, time


# ── SNN filter – vectorised (port of snnV2.m) ───────────────────────────────
def snn_filter_fast(img, win_sz=3):
    """Symmetric Nearest Neighbor filter (numpy-vectorised)."""
    pad = win_sz // 2
    h, w = img.shape
    padded = np.pad(img, pad, mode='edge').astype(np.float64)

    offsets_top, offsets_bot = [], []
    for dy in range(-pad, 0):
        for dx in range(-pad, pad + 1):
            offsets_top.append((dy, dx))
            offsets_bot.append((-dy, -dx))
    for dx in range(-pad, 0):
        offsets_top.append((0, dx))
        offsets_bot.append((0, -dx))

    centre = padded[pad:pad+h, pad:pad+w]
    acc = centre.copy()
    count = np.ones_like(centre)

    for (dy_t, dx_t), (dy_b, dx_b) in zip(offsets_top, offsets_bot):
        v_top = padded[pad+dy_t:pad+dy_t+h, pad+dx_t:pad+dx_t+w]
        v_bot = padded[pad+dy_b:pad+dy_b+h, pad+dx_b:pad+dx_b+w]
        chosen = np.where(np.abs(v_top - centre) <= np.abs(v_bot - centre),
                          v_top, v_bot)
        acc += chosen
        count += 1

    return acc / count


# ── Enhancement pipeline v2 ─────────────────────────────────────────────────
def enhance_crop_v2(crop_gray):
    """
    Full 7-step enhancement pipeline for a single bee close-up.
    All parameters are adaptive (derived from the crop itself).
    """
    img = crop_gray.copy()
    h, w = img.shape

    # ── 1. Adaptive gamma correction ────────────────────────────────────────
    #    Estimate current brightness and warp towards a target so dark crops
    #    get lifted and already-bright ones are barely touched.
    mean_val = float(img.mean())
    target_mean = 115.0                         # ~45 % grey
    if mean_val > 5:
        gamma = np.log(target_mean / 255.0) / np.log(mean_val / 255.0)
        gamma = float(np.clip(gamma, 0.4, 2.5))
    else:
        gamma = 1.0
    lut = np.array([((i / 255.0) ** gamma) * 255.0
                     for i in range(256)], dtype=np.uint8)
    img = cv2.LUT(img, lut)

    # ── 2. Non-local means denoising (light) ────────────────────────────────
    #    Preserves texture much better than bilateral/Gaussian.
    #    h  = filter strength (higher → smoother)
    #    Adapt strength to crop noise level (std of Laplacian).
    noise_est = cv2.Laplacian(img, cv2.CV_64F).std()
    nlm_h = int(np.clip(noise_est * 0.6, 5, 14))
    img = cv2.fastNlMeansDenoising(img, h=nlm_h,
                                    templateWindowSize=7,
                                    searchWindowSize=21)

    # ── 3. CLAHE – smooth local contrast ────────────────────────────────────
    #    Low clip + large grid → no visible tile boundaries.
    #    Two-pass with different grids for smoother multi-scale boost.
    clahe1 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe1.apply(img)
    clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    img = clahe2.apply(img)

    # ── 4. SNN filter (win 3) – gentle edge-preserving refinement ───────────
    #    Window 3 is the lightest touch: smooths 1-pixel noise while
    #    leaving all real structure intact (bee hair, wing veins …).
    f = img.astype(np.float64) / 255.0
    f = snn_filter_fast(f, win_sz=3)
    img = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    # ── 5. Multi-scale detail boost (Laplacian decomposition) ───────────────
    #    base  = low-frequency illumination (Gaussian σ ≈ 3)
    #    detail = high-frequency structure (bee body, wings, legs)
    #    Amplify detail layer → sharper anatomical features.
    img_f = img.astype(np.float64)

    # Two-level decomposition for finer control
    base_coarse = cv2.GaussianBlur(img_f, (0, 0), 5.0)   # illumination
    base_fine   = cv2.GaussianBlur(img_f, (0, 0), 1.5)    # smooth structure
    detail_mid  = base_fine - base_coarse                   # medium detail
    detail_fine = img_f - base_fine                         # fine detail

    # Boost medium details (body segments, honeycomb edges) strongly,
    # boost fine details (hair, texture) more gently
    img_f = base_coarse + detail_mid * 1.8 + detail_fine * 1.3
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    # ── 6. Adaptive sharpening (edge-weighted unsharp mask) ─────────────────
    #    Compute per-pixel edge magnitude → sharpen strongly at edges,
    #    leave flat/noisy areas untouched.
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx * gx + gy * gy)
    edge_max = edge_mag.max()
    if edge_max > 0:
        edge_mag /= edge_max
    # Smooth the weight map so sharpening transitions are gradual
    edge_w = cv2.GaussianBlur(edge_mag, (0, 0), 2.0)
    edge_w = np.clip(edge_w * 2.0, 0, 1)           # boost low-magnitude edges

    blur_sharp = cv2.GaussianBlur(img.astype(np.float64), (0, 0), 1.2)
    hi_pass    = img.astype(np.float64) - blur_sharp
    sharpened  = img.astype(np.float64) + 0.9 * hi_pass
    # Blend: edges → sharpened, flats → original
    result = img.astype(np.float64) * (1.0 - edge_w) + sharpened * edge_w
    img = np.clip(result, 0, 255).astype(np.uint8)

    # ── 7. Percentile stretch (soft) ────────────────────────────────────────
    #    Map 1st–99th percentile to 5–250 so we use most of the dynamic
    #    range without hard-clipping outlier pixels.
    p_lo = np.percentile(img, 1)
    p_hi = np.percentile(img, 99)
    if p_hi - p_lo > 10:
        img = np.clip((img.astype(np.float64) - p_lo) / (p_hi - p_lo) * 245.0 + 5.0,
                       0, 255).astype(np.uint8)

    return img


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    img_path = '0007.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open {img_path}")

    img_h, img_w = img.shape
    print(f"Loaded {img_path}: {img_w}x{img_h}")

    boxes_json = json.loads(r'''
    [
        {"id":"2","x":898,"y":2137,"width":150,"height":100},
        {"id":"3","x":1860.5,"y":1766,"width":119,"height":132},
        {"id":"4","x":1520.5,"y":1961,"width":119,"height":94},
        {"id":"5","x":2119,"y":1686.5,"width":76,"height":111},
        {"id":"6","x":2163,"y":1542,"width":88,"height":140},
        {"id":"7","x":600,"y":798,"width":64,"height":142},
        {"id":"8","x":404.5,"y":943,"width":143,"height":96},
        {"id":"9","x":2083.5,"y":1579,"width":65,"height":150},
        {"id":"A","x":823.5,"y":465.5,"width":65,"height":151},
        {"id":"B","x":2468,"y":1827.5,"width":92,"height":97},
        {"id":"C","x":1812,"y":1553.5,"width":78,"height":127},
        {"id":"D","x":2941,"y":2237.5,"width":102,"height":129},
        {"id":"E","x":944,"y":286,"width":72,"height":114},
        {"id":"F","x":2090,"y":2236,"width":72,"height":134},
        {"id":"G","x":1354.5,"y":1342,"width":81,"height":196},
        {"id":"H","x":483,"y":1547,"width":116,"height":64},
        {"id":"I","x":1020.5,"y":339,"width":99,"height":90},
        {"id":"J","x":1455,"y":1653,"width":76,"height":152},
        {"id":"K","x":2300,"y":1853,"width":54,"height":144},
        {"id":"L","x":787,"y":610.5,"width":68,"height":149},
        {"id":"M","x":241.5,"y":1750.5,"width":63,"height":127},
        {"id":"N","x":831,"y":872.5,"width":64,"height":125},
        {"id":"O","x":126,"y":963,"width":54,"height":136},
        {"id":"P","x":3175.5,"y":1987,"width":113,"height":88},
        {"id":"Q","x":1251,"y":1893.5,"width":114,"height":131},
        {"id":"R","x":2019.5,"y":1851,"width":105,"height":122},
        {"id":"S","x":3026.5,"y":1978,"width":107,"height":116},
        {"id":"T","x":909.5,"y":468.5,"width":99,"height":117},
        {"id":"U","x":1021.5,"y":1529.5,"width":89,"height":81},
        {"id":"V","x":1968,"y":2276.5,"width":88,"height":149},
        {"id":"W","x":763,"y":1912.5,"width":60,"height":137},
        {"id":"X","x":1061,"y":1896,"width":138,"height":100},
        {"id":"Y","x":1453,"y":1335,"width":74,"height":144},
        {"id":"Z","x":731.5,"y":530,"width":59,"height":148},
        {"id":"a","x":1518.5,"y":1485.5,"width":63,"height":163},
        {"id":"b","x":51,"y":708.5,"width":102,"height":103},
        {"id":"c","x":633,"y":959.5,"width":62,"height":119},
        {"id":"d","x":755.5,"y":1767,"width":147,"height":84},
        {"id":"e","x":373,"y":1838.5,"width":74,"height":133},
        {"id":"f","x":236.5,"y":1608.5,"width":59,"height":159},
        {"id":"g","x":557.5,"y":1269.5,"width":111,"height":107},
        {"id":"h","x":1072.5,"y":1673.5,"width":85,"height":127},
        {"id":"i","x":2322.5,"y":1968.5,"width":125,"height":97},
        {"id":"j","x":3127,"y":2272,"width":106,"height":120},
        {"id":"k","x":1015,"y":528,"width":56,"height":142},
        {"id":"l","x":1190.5,"y":516.5,"width":65,"height":153},
        {"id":"m","x":1663.5,"y":1832,"width":117,"height":96},
        {"id":"n","x":665,"y":1163.5,"width":116,"height":103},
        {"id":"o","x":441,"y":1670.5,"width":70,"height":155},
        {"id":"p","x":166,"y":1934.5,"width":118,"height":101},
        {"id":"q","x":1925,"y":1648,"width":74,"height":148},
        {"id":"r","x":325,"y":1947.5,"width":112,"height":121},
        {"id":"s","x":2223.5,"y":1816,"width":65,"height":108},
        {"id":"t","x":1660,"y":1602,"width":84,"height":118},
        {"id":"u","x":2796,"y":2017.5,"width":106,"height":109},
        {"id":"v","x":2144,"y":1821,"width":86,"height":138},
        {"id":"w","x":2546,"y":2008.5,"width":64,"height":49},
        {"id":"x","x":607,"y":1830.5,"width":82,"height":99},
        {"id":"y","x":1360,"y":1229,"width":82,"height":162},
        {"id":"z","x":726,"y":682.5,"width":66,"height":93},
        {"id":"11","x":1674,"y":1319.5,"width":68,"height":123},
        {"id":"12","x":1134,"y":527.5,"width":58,"height":149},
        {"id":"13","x":1696,"y":1931.5,"width":124,"height":77},
        {"id":"14","x":159.5,"y":794.5,"width":71,"height":135},
        {"id":"15","x":2218.5,"y":2296,"width":111,"height":142},
        {"id":"16","x":1123.5,"y":1743.5,"width":93,"height":91},
        {"id":"17","x":992.5,"y":2244.5,"width":153,"height":101},
        {"id":"18","x":897,"y":1844,"width":90,"height":118}
    ]
    ''')

    out_dir   = os.path.join('..', 'output', '0007', 'bee_closeups_v2')
    os.makedirs(out_dir, exist_ok=True)

    PAD = 20

    crops_before = []
    crops_after  = []
    total = len(boxes_json)
    t0 = time.time()

    for idx, box in enumerate(boxes_json):
        bid = box['id']
        cx, cy = float(box['x']), float(box['y'])
        bw, bh = int(box['width']), int(box['height'])

        x1 = max(0, int(cx - bw / 2) - PAD)
        y1 = max(0, int(cy - bh / 2) - PAD)
        x2 = min(img_w, int(cx + bw / 2) + PAD)
        y2 = min(img_h, int(cy + bh / 2) + PAD)

        crop = img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        enhanced = enhance_crop_v2(crop)

        # ── Save outputs ────────────────────────────────────────────────────
        # Side-by-side (original | enhanced) with thin white separator
        sep = np.full((crop.shape[0], 2), 200, dtype=np.uint8)
        comparison = np.hstack([crop, sep, enhanced])
        cv2.imwrite(os.path.join(out_dir, f'bee_{bid}_compare.png'), comparison)
        cv2.imwrite(os.path.join(out_dir, f'bee_{bid}_enhanced.png'), enhanced)

        crops_before.append((bid, crop))
        crops_after.append((bid, enhanced))

        elapsed = time.time() - t0
        print(f"  [{idx+1}/{total}] bee {bid:>3s}  ({bw}x{bh})  {elapsed:.1f}s")

    # ── Montage ──────────────────────────────────────────────────────────────
    # Select the 30 largest bees for a readable grid
    sized = [(bid, bf, af, bf.shape[0] * bf.shape[1])
             for (bid, bf), (_, af) in zip(crops_before, crops_after)]
    sized.sort(key=lambda t: t[3], reverse=True)
    selected = [(b, bf, af) for b, bf, af, _ in sized[:30]]

    if selected:
        CELL = 200
        cols = 6
        rows = (len(selected) + cols - 1) // cols
        gap  = 3                               # px between before/after
        cell_w = CELL * 2 + gap
        montage_w = cols * cell_w + (cols - 1) * 2
        montage_h = rows * CELL + (rows - 1) * 2

        montage = np.full((montage_h, montage_w), 30, dtype=np.uint8)

        for k, (bid, bf, af) in enumerate(selected):
            r, c = divmod(k, cols)
            scale = min(CELL / bf.shape[0], CELL / bf.shape[1])
            nh, nw = int(bf.shape[0] * scale), int(bf.shape[1] * scale)
            bf_r = cv2.resize(bf, (nw, nh), interpolation=cv2.INTER_AREA)
            af_r = cv2.resize(af, (nw, nh), interpolation=cv2.INTER_AREA)

            yy = r * (CELL + 2) + (CELL - nh) // 2
            xx_bf = c * (cell_w + 2) + (CELL - nw) // 2
            xx_af = xx_bf + CELL + gap

            montage[yy:yy+nh, xx_bf:xx_bf+nw] = bf_r
            montage[yy:yy+nh, xx_af:xx_af+nw] = af_r

            cv2.putText(montage, bid, (xx_bf, r * (CELL + 2) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, 220, 1)

        cv2.imwrite(os.path.join(out_dir, 'montage_before_after.png'), montage)
        print(f"\nMontage saved  ({len(selected)} bees)")

    elapsed = time.time() - t0
    print(f"\nDone.  {len(crops_after)} crops in {elapsed:.1f}s")
    print(f"Output → {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    main()
