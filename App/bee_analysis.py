"""
bee_analysis.py
---------------
Behavioural analyses computed from Keypoint R-CNN predictions.

Analyses:
  1. Heading map        – arrows showing which way each bee faces
  2. Rose diagram       – polar histogram of headings + Rayleigh r
  3. Local alignment    – how aligned each bee is with its neighbours
  4. Flow field         – interpolated streamlines + divergence + curl
  5. Following chains   – head-to-tail following sequences
  6. Density heatmap    – Gaussian KDE of bee positions
  7. Facing pairs       – bees pointing at each other (trophallaxis candidates)
  8. Crowding index     – local density around each bee

Convention (from COCO keypoints):
  kp index 0 = head,  kp index 1 = tail
  heading vector = head − tail
"""

import base64, io
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_b64_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _to_b64_bgr(img_bgr):
    _, enc = cv2.imencode(".png", img_bgr)
    return base64.b64encode(enc.tobytes()).decode()


def _extract_bees(detections):
    """Keep only detections with both head (kp0) and tail (kp1) visible."""
    bees = []
    for det in detections:
        kps = det.get("keypoints", [])
        if len(kps) < 2:
            continue
        k0, k1 = kps[0], kps[1]
        if not (k0.get("visible") and k1.get("visible")):
            continue
        bees.append({
            "head": (float(k0["x"]), float(k0["y"])),
            "tail": (float(k1["x"]), float(k1["y"])),
            "cls":  det.get("class_name", "bee"),
            "conf": float(det.get("confidence", 1.0)),
        })
    return bees


def _build_arrays(bees):
    """Return (heads, tails, centres, unit_vecs, angles_deg, valid_mask)."""
    heads   = np.array([b["head"] for b in bees], dtype=np.float32)
    tails   = np.array([b["tail"] for b in bees], dtype=np.float32)
    centres = (heads + tails) / 2.0
    diffs   = heads - tails
    lengths = np.linalg.norm(diffs, axis=1)
    valid   = lengths > 1.0
    safe_l  = np.where(valid, lengths, 1.0)[:, None]
    unit_vecs  = diffs / safe_l
    angles_deg = np.degrees(np.arctan2(diffs[:, 0], -diffs[:, 1])) % 360
    return heads, tails, centres, unit_vecs, angles_deg, valid


def _darkened_bgr(img_rgb, factor=0.55):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return (bgr.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def _fig_size(img_rgb, max_w=12, max_h=8):
    H, W = img_rgb.shape[:2]
    fw = min(max_w, W / 80)
    fh = min(max_h, H / 80)
    return (max(fw, 5), max(fh, 4))


# ── 1. Heading map ──────────────────────────────────────────────────────────

def draw_directions(img_rgb, bees):
    """Colour-coded heading arrows (orange=cleaning, green=normal)."""
    canvas = _darkened_bgr(img_rgb, 0.60)
    for bee in bees:
        tx, ty = int(bee["tail"][0]), int(bee["tail"][1])
        hx, hy = int(bee["head"][0]), int(bee["head"][1])
        dx, dy = hx - tx, hy - ty
        ln = (dx**2 + dy**2) ** 0.5
        if ln < 2:
            continue
        ex = int(hx + dx / ln * 10)
        ey = int(hy + dy / ln * 10)
        col = (0, 120, 255) if "clean" in bee["cls"] else (40, 200, 80)
        cv2.arrowedLine(canvas, (tx, ty), (ex, ey), (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.28)
        cv2.arrowedLine(canvas, (tx, ty), (ex, ey), col,       2, cv2.LINE_AA, tipLength=0.28)

    fig, ax = plt.subplots(figsize=_fig_size(img_rgb), facecolor="#111")
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=(0.16, 0.47, 1.0), label="Cleaning bee"),
                       Patch(facecolor=(0.16, 0.78, 0.31), label="Normal bee")],
              loc="upper right", fontsize=7, framealpha=0.3,
              labelcolor="white", facecolor="#222")
    ax.set_title(f"Heading map  ({len(bees)} bees)", color="white", fontsize=9, pad=6)
    plt.tight_layout(pad=0.3)
    return _to_b64_fig(fig)


# ── 2. Rose diagram ─────────────────────────────────────────────────────────

def rose_diagram(bees):
    """Polar histogram of heading directions + Rayleigh r statistic."""
    _, _, _, unit_vecs, angles_deg, valid = _build_arrays(bees)
    unit_vecs  = unit_vecs[valid]
    angles_deg = angles_deg[valid]
    N = len(angles_deg)
    if N == 0:
        return None

    N_BINS = 24
    bin_edges   = np.linspace(0, 2 * np.pi, N_BINS + 1)
    counts, _   = np.histogram(np.radians(angles_deg), bins=bin_edges)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    width       = 2 * np.pi / N_BINS

    mx, my = float(np.mean(unit_vecs[:, 0])), float(np.mean(unit_vecs[:, 1]))
    R      = float((mx**2 + my**2) ** 0.5)
    mean_a = float(np.degrees(np.arctan2(my, mx)))

    fig = plt.figure(figsize=(6, 6), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, polar=True, facecolor="#16213e")

    norm   = plt.Normalize(vmin=0, vmax=max(counts.max(), 1))
    colors = cm.plasma(norm(counts))
    ax.bar(bin_centres, counts, width=width * 0.92, bottom=0,
           color=colors, edgecolor="#1a1a2e", linewidth=0.5)

    ax.annotate("", xy=(np.arctan2(my, mx), R * max(counts.max(), 1)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#00f5d4", lw=2.5, mutation_scale=18))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["polar"].set_color("#444")
    for l in ax.get_xticklabels(): l.set_color("white")
    for l in ax.get_yticklabels(): l.set_color("#aaa")
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=10, color="white")

    # Interpret Rayleigh r
    if R > 0.7:    tendency = "strongly aligned"
    elif R > 0.4:  tendency = "moderately aligned"
    elif R > 0.2:  tendency = "weakly aligned"
    else:           tendency = "random / scattered"

    ax.set_title(f"Heading distribution  (n={N})\n"
                 f"Rayleigh r = {R:.3f}  ({tendency})\n"
                 f"Mean direction = {mean_a:.0f}°",
                 color="white", pad=16, fontsize=9)

    sm = cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.1, shrink=0.55)
    cbar.set_label("count", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    plt.tight_layout()
    return _to_b64_fig(fig)


# ── 3. Local alignment map ────────────────────────────────────────────────────

def alignment_map(img_rgb, bees):
    """Arrows coloured by alignment score with k nearest neighbours (green=aligned)."""
    _, _, centres, unit_vecs, _, valid = _build_arrays(bees)
    centres_v   = centres[valid]
    unit_vecs_v = unit_vecs[valid]
    bees_v      = [b for b, v in zip(bees, valid) if v]
    n = len(centres_v)
    K = min(5, n - 1)
    if K < 1:
        return None

    tree = KDTree(centres_v)
    _, idxs = tree.query(centres_v, k=K + 1)
    scores = np.array([float(np.mean(unit_vecs_v[idxs[i][1:]] @ unit_vecs_v[i]))
                       for i in range(n)])
    s_norm = (scores - scores.min()) / (np.ptp(scores) + 1e-9)

    canvas = _darkened_bgr(img_rgb, 0.50)
    cmap   = cm.RdYlGn

    for i, bee in enumerate(bees_v):
        tx, ty = int(bee["tail"][0]), int(bee["tail"][1])
        hx, hy = int(bee["head"][0]), int(bee["head"][1])
        dx, dy = hx - tx, hy - ty
        ln = (dx**2 + dy**2) ** 0.5
        if ln < 2:
            continue
        ex = int(hx + dx / ln * 10)
        ey = int(hy + dy / ln * 10)
        r, g, b, _ = cmap(s_norm[i])
        col_bgr = (int(b * 255), int(g * 255), int(r * 255))
        cv2.arrowedLine(canvas, (tx, ty), (ex, ey), (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.28)
        cv2.arrowedLine(canvas, (tx, ty), (ex, ey), col_bgr,   2, cv2.LINE_AA, tipLength=0.28)

    fig, ax = plt.subplots(figsize=_fig_size(img_rgb), facecolor="#111")
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    sm = cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=scores.min(), vmax=scores.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.01, shrink=0.55)
    cbar.set_label(f"Local alignment (k={K} neighbours)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
    ax.set_title("Local heading alignment  (green=coordinated, red=isolated)",
                 color="white", fontsize=9, pad=6)
    plt.tight_layout(pad=0.3)
    return _to_b64_fig(fig)


# ── 4. Spatial flow field ─────────────────────────────────────────────────────

def flow_field(img_rgb, bees):
    """Interpolated streamlines + divergence + curl heatmaps."""
    if len(bees) < 5:
        return None
    try:
        from scipy.interpolate import griddata
    except ImportError:
        return None

    _, _, centres, unit_vecs, _, valid = _build_arrays(bees)
    C  = centres[valid]
    UV = unit_vecs[valid]

    H, W = img_rgb.shape[:2]
    NX = min(40, max(8, W // 12))
    NY = min(25, max(6, H // 12))

    gx = np.linspace(0, W, NX)
    gy = np.linspace(0, H, NY)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])

    U = griddata(C, UV[:, 0], pts, method="linear").reshape(NY, NX)
    V = griddata(C, UV[:, 1], pts, method="linear").reshape(NY, NX)
    nan_m = np.isnan(U)
    if nan_m.any():
        Un = griddata(C, UV[:, 0], pts, method="nearest").reshape(NY, NX)
        Vn = griddata(C, UV[:, 1], pts, method="nearest").reshape(NY, NX)
        U[nan_m] = Un[nan_m]; V[nan_m] = Vn[nan_m]

    dx_g = gx[1] - gx[0] if NX > 1 else 1.0
    dy_g = gy[1] - gy[0] if NY > 1 else 1.0
    div  = np.gradient(U, dx_g, axis=1) + np.gradient(V, dy_g, axis=0)
    curl = np.gradient(V, dx_g, axis=1) - np.gradient(U, dy_g, axis=0)
    spd  = np.sqrt(U**2 + V**2)

    base_rgb = img_rgb.copy()
    fig = plt.figure(figsize=(14, 9), facecolor="#0d0d1a")

    # ── Panel A: streamlines
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.imshow(base_rgb, alpha=0.38, extent=[0, W, H, 0], aspect="auto")
    strm = ax1.streamplot(gx, gy, U, -V, density=1.5, color=spd,
                          cmap="plasma", linewidth=1.2, arrowsize=1.4,
                          broken_streamlines=True)
    cb1 = fig.colorbar(strm.lines, ax=ax1, fraction=0.018, pad=0.01)
    cb1.set_label("vector magnitude", color="white", fontsize=7)
    cb1.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax1.quiver(C[:, 0], C[:, 1], UV[:, 0], -UV[:, 1],
               color="#00f5d4", alpha=0.85, scale=22, width=0.003, headwidth=4)
    ax1.set_xlim(0, W); ax1.set_ylim(H, 0); ax1.axis("off")
    ax1.set_title("Flow field  (plasma = vector magnitude)", color="white", fontsize=9)

    # ── Panel B: divergence
    ax2 = fig.add_subplot(2, 2, 3)
    dlim = max(np.percentile(np.abs(div), 97), 1e-9)
    im2  = ax2.imshow(div, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                      extent=[0, W, H, 0], aspect="auto", origin="upper")
    ax2.scatter(C[:, 0], C[:, 1], s=6, c="white", alpha=0.55, zorder=3)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
    cb2.set_label("divergence", color="white", fontsize=7)
    cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax2.set_title("Divergence  (red=spreading, blue=converging)", color="white", fontsize=8)
    ax2.axis("off")

    # ── Panel C: curl
    ax3 = fig.add_subplot(2, 2, 4)
    clim = max(np.percentile(np.abs(curl), 97), 1e-9)
    im3  = ax3.imshow(curl, cmap="PuOr", vmin=-clim, vmax=clim,
                      extent=[0, W, H, 0], aspect="auto", origin="upper")
    ax3.scatter(C[:, 0], C[:, 1], s=6, c="white", alpha=0.55, zorder=3)
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.03, pad=0.02)
    cb3.set_label("curl", color="white", fontsize=7)
    cb3.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax3.set_title("Curl  (orange=clockwise,  purple=counter-clockwise)",
                  color="white", fontsize=8)
    ax3.axis("off")

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#0d0d1a")

    plt.tight_layout(pad=0.8)
    return _to_b64_fig(fig)


# ── 5. Following chains ───────────────────────────────────────────────────────

def following_chains(img_rgb, bees):
    """Detect head-to-tail following sequences and draw colour-coded chains."""
    heads, tails, centres, unit_vecs, _, valid = _build_arrays(bees)
    H, W = img_rgb.shape[:2]
    FOLLOW_DIST = max(W, H) * 0.05
    DOT_THRESH  = 0.25

    tail_tree = KDTree(tails)
    edges = []
    for i in range(len(bees)):
        for j in tail_tree.query_ball_point(heads[i], FOLLOW_DIST):
            if j == i: continue
            if float(unit_vecs[i] @ unit_vecs[j]) >= DOT_THRESH:
                edges.append((i, j))

    f2l = {}
    for fol, lea in edges:
        d = float(np.linalg.norm(heads[fol] - tails[lea]))
        if fol not in f2l or d < f2l[fol][1]:
            f2l[fol] = (lea, d)
    f2l = {f: v[0] for f, v in f2l.items()}

    l2f = {}
    for f, l in f2l.items():
        l2f.setdefault(l, []).append(f)

    def trace(start):
        ch, vis, cur = [start], {start}, start
        while cur in l2f:
            nxt = [x for x in l2f[cur] if x not in vis]
            if not nxt: break
            ch.append(nxt[0]); vis.add(nxt[0]); cur = nxt[0]
        return ch

    chains = sorted(
        [c for c in [trace(s) for s in l2f if s not in f2l] if len(c) >= 2],
        key=len, reverse=True)

    canvas = _darkened_bgr(img_rgb, 0.45)

    for i in range(len(bees)):
        t = (int(tails[i, 0]), int(tails[i, 1]))
        dx, dy = unit_vecs[i]
        tip = (int(heads[i, 0] + dx * 7), int(heads[i, 1] + dy * 7))
        cv2.arrowedLine(canvas, t, tip, (0, 0, 0),  3, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(canvas, t, tip, (70, 70, 70), 1, cv2.LINE_AA, tipLength=0.3)

    for ci, chain in enumerate(chains):
        hue = ci / max(len(chains), 1)
        r, g, b_ = mcolors.hsv_to_rgb([hue, 0.95, 1.0])
        col = (int(b_ * 255), int(g * 255), int(r * 255))

        for step, idx in enumerate(chain):
            t   = (int(tails[idx, 0]), int(tails[idx, 1]))
            dx, dy = unit_vecs[idx]
            tip = (int(heads[idx, 0] + dx * 9), int(heads[idx, 1] + dy * 9))
            cv2.arrowedLine(canvas, t, tip, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.3)
            cv2.arrowedLine(canvas, t, tip, col,       2, cv2.LINE_AA, tipLength=0.3)

            if step > 0:
                p1 = np.array([int(heads[chain[step-1], 0]), int(heads[chain[step-1], 1])])
                p2 = np.array(t)
                ln = np.linalg.norm(p2 - p1)
                if ln > 1:
                    for s in range(int(ln / 13)):
                        f0 = s * 13 / ln
                        f1 = min(f0 + 8 / ln, 1.0)
                        cv2.line(canvas,
                                 tuple((p1 + (p2 - p1) * f0).astype(int)),
                                 tuple((p1 + (p2 - p1) * f1).astype(int)),
                                 col, 1, cv2.LINE_AA)

            cx, cy = int(centres[idx, 0]), int(centres[idx, 1])
            cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), -1)
            cv2.circle(canvas, (cx, cy), 3, col, -1)
            if step == 0:
                cv2.putText(canvas, f"C{ci+1}", (cx+6, cy-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, f"C{ci+1}", (cx+6, cy-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    n_chains   = len(chains)
    longest    = len(chains[0]) if chains else 0
    n_chained  = sum(len(c) for c in chains)

    fig, ax = plt.subplots(figsize=_fig_size(img_rgb), facecolor="#111")
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(f"Following chains  ({n_chains} chains, longest={longest} bees,  "
                 f"{n_chained}/{len(bees)} bees in chains)",
                 color="white", fontsize=9, pad=6)
    plt.tight_layout(pad=0.3)
    return _to_b64_fig(fig), n_chains, longest


# ── 6. Density heatmap ────────────────────────────────────────────────────────

def density_heatmap(img_rgb, bees):
    """Gaussian KDE showing where bees cluster on the comb."""
    _, _, centres, _, _, _ = _build_arrays(bees)
    H, W = img_rgb.shape[:2]

    grid = np.zeros((H, W), dtype=np.float32)
    for cx, cy in centres:
        gx, gy = int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))
        grid[gy, gx] += 1.0

    sigma   = max(W, H) * 0.045
    heatmap = gaussian_filter(grid, sigma=sigma)
    heatmap /= (heatmap.max() + 1e-9)

    hot_rgb = (cm.hot(heatmap)[:, :, :3] * 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hot_bgr  = cv2.cvtColor(hot_rgb, cv2.COLOR_RGB2BGR)
    alpha    = (heatmap * 0.75)[:, :, np.newaxis]
    blended  = (base_bgr * (1 - alpha) + hot_bgr * alpha).clip(0, 255).astype(np.uint8)

    for cx, cy in centres:
        cv2.circle(blended, (int(cx), int(cy)), 5, (0, 0, 0),     -1)
        cv2.circle(blended, (int(cx), int(cy)), 3, (255, 255, 0),  -1)

    # Mark density peak
    peak_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
    cv2.drawMarker(blended, (peak_x, peak_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    fig, ax = plt.subplots(figsize=_fig_size(img_rgb), facecolor="#111")
    ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    sm = cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.01, shrink=0.55)
    cbar.set_label("relative density", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
    ax.set_title(f"Bee density heatmap  ({len(bees)} bees)  ✛ = peak density",
                 color="white", fontsize=9, pad=6)
    plt.tight_layout(pad=0.3)
    return _to_b64_fig(fig), (peak_x, peak_y)


# ── 7. Facing pairs (trophallaxis candidates) ─────────────────────────────────

def facing_pairs(img_rgb, bees):
    """
    Detect pairs of bees whose heads are close and headings oppose each other.
    Candidate behaviour: trophallaxis (food exchange) or social interaction.
    """
    heads, tails, _, unit_vecs, _, _ = _build_arrays(bees)
    H, W = img_rgb.shape[:2]
    MAX_DIST   = max(W, H) * 0.10   # within 10 % of image diagonal
    FACING_DOT = -0.50              # cos(120°) — substantially opposed

    head_tree = KDTree(heads)
    pairs = []
    for i in range(len(bees)):
        for j in head_tree.query_ball_point(heads[i], MAX_DIST):
            if j <= i:
                continue
            dot  = float(unit_vecs[i] @ unit_vecs[j])
            dist = float(np.linalg.norm(heads[i] - heads[j]))
            if dot <= FACING_DOT:
                # Extra check: bee i's head is roughly in the direction bee j faces
                d_ij = heads[i] - heads[j]
                dn   = np.linalg.norm(d_ij)
                if dn > 0:
                    d_ij /= dn
                    if float(unit_vecs[j] @ d_ij) > 0.3:
                        pairs.append((i, j, dist, dot))

    canvas = _darkened_bgr(img_rgb, 0.50)

    for i in range(len(bees)):
        t = (int(tails[i, 0]), int(tails[i, 1]))
        dx, dy = unit_vecs[i]
        tip = (int(heads[i, 0] + dx * 7), int(heads[i, 1] + dy * 7))
        cv2.arrowedLine(canvas, t, tip, (0, 0, 0),  3, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(canvas, t, tip, (70, 70, 70), 1, cv2.LINE_AA, tipLength=0.3)

    for k, (i, j, dist, dot) in enumerate(pairs):
        hi = (int(heads[i, 0]), int(heads[i, 1]))
        hj = (int(heads[j, 0]), int(heads[j, 1]))
        for idx in (i, j):
            t = (int(tails[idx, 0]), int(tails[idx, 1]))
            dx, dy = unit_vecs[idx]
            tip = (int(heads[idx, 0] + dx * 9), int(heads[idx, 1] + dy * 9))
            cv2.arrowedLine(canvas, t, tip, (0, 0, 0),     5, cv2.LINE_AA, tipLength=0.3)
            cv2.arrowedLine(canvas, t, tip, (30, 210, 255), 2, cv2.LINE_AA, tipLength=0.3)
        mid = ((hi[0] + hj[0]) // 2, (hi[1] + hj[1]) // 2)
        cv2.line(canvas, hi, hj, (30, 210, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, hi, 6, (30, 210, 255), -1)
        cv2.circle(canvas, hj, 6, (30, 210, 255), -1)
        cv2.putText(canvas, f"P{k+1}", (mid[0]+4, mid[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0),     3, cv2.LINE_AA)
        cv2.putText(canvas, f"P{k+1}", (mid[0]+4, mid[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 210, 255), 1, cv2.LINE_AA)

    fig, ax = plt.subplots(figsize=_fig_size(img_rgb), facecolor="#111")
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(f"Facing pairs — trophallaxis / interaction candidates  ({len(pairs)} pairs)",
                 color="white", fontsize=9, pad=6)
    plt.tight_layout(pad=0.3)
    return _to_b64_fig(fig), len(pairs)


# ── 8. Crowding index ─────────────────────────────────────────────────────────

def crowding_map(img_rgb, bees):
    """
    For each bee, count neighbours within a local radius.
    High crowding → resource competition / comb maintenance zones.
    """
    _, _, centres, _, _, valid = _build_arrays(bees)
    C      = centres[valid]
    bees_v = [b for b, v in zip(bees, valid) if v]
    if len(C) < 2:
        return None

    H, W   = img_rgb.shape[:2]
    RADIUS = max(W, H) * 0.09
    tree   = KDTree(C)
    crowd  = np.array([len(tree.query_ball_point(c, RADIUS)) - 1 for c in C])

    norm = plt.Normalize(0, max(crowd.max(), 1))
    cmap = cm.YlOrRd
    canvas = _darkened_bgr(img_rgb, 0.48)

    for i, bee in enumerate(bees_v):
        cx = int((bee["head"][0] + bee["tail"][0]) / 2)
        cy = int((bee["head"][1] + bee["tail"][1]) / 2)
        r, g, b_, _ = cmap(norm(crowd[i]))
        col = (int(b_ * 255), int(g * 255), int(r * 255))
        cv2.circle(canvas, (cx, cy), int(RADIUS), col, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 8, (0, 0, 0), -1)
        cv2.circle(canvas, (cx, cy), 6, col, -1)
        cv2.putText(canvas, str(crowd[i]), (cx + 7, cy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(crowd[i]), (cx + 7, cy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    fig, ax = plt.subplots(figsize=_fig_size(img_rgb), facecolor="#111")
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.01, shrink=0.55)
    cbar.set_label(f"neighbours within {RADIUS:.0f} px", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
    ax.set_title(f"Crowding index  (red = dense zone, number = neighbour count)",
                 color="white", fontsize=9, pad=6)
    plt.tight_layout(pad=0.3)
    return _to_b64_fig(fig)


# ── Main entry point ───────────────────────────────────────────────────────────

def analyse(img_rgb, detections):
    """
    Run all behavioural analyses on model predictions.

    Returns a dict:
      analyses : { name → base64_png | None }
      stats    : { n_bees, n_cleaning, n_normal, rayleigh_r, mean_heading,
                   n_chains, longest_chain, n_facing_pairs, density_peak }
      error    : str (only present when not enough valid bees)
    """
    bees = _extract_bees(detections)

    if len(bees) < 2:
        return {"error": f"Need ≥2 bees with head+tail visible (got {len(bees)})."}

    out = {"analyses": {}}
    a   = out["analyses"]

    a["directions"] = draw_directions(img_rgb, bees)

    if len(bees) >= 3:
        a["rose"]      = rose_diagram(bees)
        a["alignment"] = alignment_map(img_rgb, bees)

    if len(bees) >= 5:
        a["flow"] = flow_field(img_rgb, bees)

    chains_img, n_chains, longest = following_chains(img_rgb, bees)
    a["chains"] = chains_img

    heatmap_img, peak = density_heatmap(img_rgb, bees)
    a["heatmap"] = heatmap_img

    facing_img, n_facing = facing_pairs(img_rgb, bees)
    a["facing"] = facing_img

    if len(bees) >= 3:
        a["crowding"] = crowding_map(img_rgb, bees)

    # ── Stats ──────────────────────────────────────────────────────────────
    _, _, _, unit_vecs, angles_deg, valid = _build_arrays(bees)
    uv  = unit_vecs[valid]
    mx, my = float(np.mean(uv[:, 0])), float(np.mean(uv[:, 1]))
    R   = float((mx**2 + my**2) ** 0.5)
    mean_h = float(np.degrees(np.arctan2(my, mx)))
    cleaning = sum(1 for b in bees if "clean" in b["cls"])

    if R > 0.7:    order_label = "strongly ordered"
    elif R > 0.4:  order_label = "moderately ordered"
    elif R > 0.2:  order_label = "weakly ordered"
    else:           order_label = "disordered"

    out["stats"] = {
        "n_bees":        len(bees),
        "n_cleaning":    cleaning,
        "n_normal":      len(bees) - cleaning,
        "rayleigh_r":    round(R, 3),
        "order_label":   order_label,
        "mean_heading":  round(mean_h, 1),
        "n_chains":      n_chains,
        "longest_chain": longest,
        "n_facing_pairs": n_facing,
        "density_peak":  list(peak),
    }

    return out
