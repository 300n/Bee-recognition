#!/usr/bin/env python3
"""
bee_flow_chains.py
==================
Two spatial analyses on bee heading vectors:

  1. Spatial flow field — heading vectors interpolated onto a regular grid,
     rendered as streamlines + divergence / curl heatmaps.

  2. Head-to-tail following chains — edges where bee A's head is within
     FOLLOW_DIST px of bee B's tail (and both face roughly the same way),
     chains traced and colour-coded on the image.

Outputs → pipeline_output/bee_boxes/
  flow_field.png
  following_chains.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.spatial import KDTree

# ── Keypoints ─────────────────────────────────────────────────────────────────
BEES = [
    {"id": "6",  "kp0": (664.5361, 159.6585), "kp1": (722.9339, 147.5952)},
    {"id": "7",  "kp0": (501.9891, 162.7360),  "kp1": (499.8257, 216.2864)},
    {"id": "9",  "kp0": (407.8836, 147.0191),  "kp1": (426.5712, 197.9274)},
    {"id": "A",  "kp0": (433.0566, 217.1855),  "kp1": (448.4541, 269.4072)},
    {"id": "B",  "kp0": (424.8149, 316.9559),  "kp1": (469.7787, 296.3650)},
    {"id": "E",  "kp0": (380.6553, 308.3224),  "kp1": (426.9059, 287.2998)},
    {"id": "F",  "kp0": (349.9113, 231.8472),  "kp1": (327.4968, 191.6196)},
    {"id": "G",  "kp0": (317.8621, 238.0958),  "kp1": (279.6633, 212.2125)},
    {"id": "H",  "kp0": (241.0400, 303.3800),  "kp1": (192.3472, 307.8744)},
    {"id": "I",  "kp0": (206.3609, 131.6757),  "kp1": (210.2174, 181.3104)},
    {"id": "J",  "kp0": (113.8259, 408.4734),  "kp1": (150.0398, 380.2573)},
    {"id": "K",  "kp0": (212.0942, 495.9783),  "kp1": (196.0558, 450.2941)},
    {"id": "L",  "kp0": (158.9928, 644.7997),  "kp1": (208.0121, 662.3321)},
    {"id": "M",  "kp0": (263.8464, 698.5082),  "kp1": (265.8762, 649.6247)},
    {"id": "N",  "kp0": (166.1517, 799.2094),  "kp1": (210.4297, 821.3882)},
    {"id": "P",  "kp0": (160.7175, 849.0032),  "kp1": (162.9085, 897.6990)},
    {"id": "Q",  "kp0": (112.4942, 852.4056),  "kp1": (126.7497, 899.0104)},
    {"id": "R",  "kp0": (653.0913, 905.8529),  "kp1": (663.2393, 866.7908)},
    {"id": "S",  "kp0": (892.8200, 955.1559),  "kp1": (912.4944, 908.9247)},
    {"id": "T",  "kp0": (1082.7812, 873.0373), "kp1": (1091.8444, 921.9229)},
    {"id": "U",  "kp0": (1429.6106, 848.3797), "kp1": (1377.6441, 850.0931)},
    {"id": "V",  "kp0": (1480.8046, 1015.0386),"kp1": (1492.9304, 975.0059)},
    {"id": "W",  "kp0": (1516.2562, 875.4008), "kp1": (1547.4513, 843.8405)},
    {"id": "X",  "kp0": (1562.4783, 837.8470), "kp1": (1527.7503, 817.4898)},
    {"id": "Y",  "kp0": (1569.4066, 822.5087), "kp1": (1579.2892, 864.8957)},
    {"id": "Z",  "kp0": (1536.1981, 743.9057), "kp1": (1534.0101, 786.1356)},
    {"id": "a",  "kp0": (1459.9586, 746.2029), "kp1": (1446.1082, 700.5605)},
    {"id": "b",  "kp0": (1342.2762, 703.7567), "kp1": (1355.7163, 664.4405)},
    {"id": "c",  "kp0": (1119.2748, 749.9615), "kp1": (1130.5331, 793.4414)},
    {"id": "d",  "kp0": (1038.5134, 655.8988), "kp1": (1074.2569, 687.0111)},
    {"id": "e",  "kp0": (955.5743,  566.9784), "kp1": (976.7449,  609.5542)},
    {"id": "g",  "kp0": (1010.4289, 495.9161), "kp1": (1053.5170, 486.5447)},
    {"id": "h",  "kp0": (1025.4611, 567.8244), "kp1": (1011.8260, 528.1367)},
    {"id": "i",  "kp0": (1047.6787, 529.7879), "kp1": (1078.7380, 567.8959)},
    {"id": "j",  "kp0": (1073.3976, 529.0113), "kp1": (1081.6362, 480.7798)},
    {"id": "k",  "kp0": (1154.0498, 495.0639), "kp1": (1164.4243, 543.7383)},
    {"id": "l",  "kp0": (1200.0310, 606.5641), "kp1": (1153.4579, 611.8167)},
    {"id": "m",  "kp0": (1271.1454, 522.2575), "kp1": (1314.2631, 537.0550)},
    {"id": "n",  "kp0": (1323.4355, 559.1297), "kp1": (1280.9862, 566.5539)},
    {"id": "p",  "kp0": (1470.7781, 589.2889), "kp1": (1421.7782, 572.8260)},
    {"id": "q",  "kp0": (1527.6125, 480.2492), "kp1": (1481.5955, 502.7474)},
    {"id": "r",  "kp0": (1365.9174, 394.7149), "kp1": (1409.8688, 414.6914)},
    {"id": "s",  "kp0": (1327.1430, 382.9576), "kp1": (1279.8953, 392.5708)},
    {"id": "t",  "kp0": (1301.1169, 329.1725), "kp1": (1334.2937, 345.0761)},
    {"id": "u",  "kp0": (1194.0794, 230.2909), "kp1": (1223.5171, 266.3548)},
    {"id": "v",  "kp0": (1077.9481, 285.0771), "kp1": (1085.4458, 329.7286)},
    {"id": "w",  "kp0": (1087.5134, 349.7787), "kp1": (1043.7211, 346.4895)},
    {"id": "x",  "kp0": (936.0775,  345.9048), "kp1": (925.4804,  395.8201)},
    {"id": "y",  "kp0": (810.0248,  411.7034), "kp1": (833.7251,  450.7431)},
    {"id": "z",  "kp0": (872.6618,  353.5392), "kp1": (829.5349,  352.9290)},
    {"id": "11", "kp0": (737.4734,  522.5739), "kp1": (695.9835,  527.8163)},
    {"id": "12", "kp0": (647.6000,  545.4313), "kp1": (676.1580,  578.9746)},
    {"id": "13", "kp0": (760.5238,  644.8640), "kp1": (732.1015,  689.0042)},
    {"id": "14", "kp0": (524.9976,  782.1770), "kp1": (556.2212,  746.2101)},
    {"id": "15", "kp0": (482.7025,  418.6071), "kp1": (523.4771,  445.2201)},
    {"id": "16", "kp0": (466.7148,  369.3942), "kp1": (511.9996,  391.5180)},
    {"id": "17", "kp0": (181.2846,  69.5279),  "kp1": (213.0256,  45.0689)},
    {"id": "18", "kp0": (334.8862,  28.4359),  "kp1": (344.5720,  58.0690)},
    {"id": "19", "kp0": (413.9171,  35.7876),  "kp1": (396.2843,  83.8738)},
    {"id": "1A", "kp0": (502.1173,  54.9432),  "kp1": (484.8559,  97.9706)},
    {"id": "1B", "kp0": (451.6011,  6.7268),   "kp1": (452.6956,  42.8103)},
    {"id": "1C", "kp0": (564.9710,  74.2851),  "kp1": (607.1289,  97.6393)},
    {"id": "1D", "kp0": (532.0852,  51.3742),  "kp1": (582.6426,  57.5389)},
    {"id": "1E", "kp0": (800.2551,  60.0386),  "kp1": (762.5916,  56.3976)},
    {"id": "1F", "kp0": (881.2533,  57.3618),  "kp1": (841.5117,  37.9834)},
    {"id": "1G", "kp0": (1155.4108, 40.1963),  "kp1": (1151.6498, 83.8795)},
    {"id": "1H", "kp0": (1132.1777, 111.0118), "kp1": (1092.1745, 95.1236)},
    {"id": "1I", "kp0": (1135.4086, 159.5378), "kp1": (1175.8609, 160.1890)},
    {"id": "1L", "kp0": (1212.9566, 173.2325), "kp1": (1194.8828, 136.0683)},
    {"id": "1M", "kp0": (1227.0843, 90.3093),  "kp1": (1184.3756, 103.9342)},
    {"id": "1N", "kp0": (1273.1186, 115.2332), "kp1": (1227.1619, 121.9017)},
    {"id": "1O", "kp0": (1234.6919, 62.1602),  "kp1": (1271.5031, 95.6468)},
    {"id": "1P", "kp0": (1316.4396, 80.2556),  "kp1": (1306.5104, 45.3089)},
    {"id": "1Q", "kp0": (1339.0032, 175.9356), "kp1": (1320.7249, 132.3050)},
    {"id": "1R", "kp0": (1311.1671, 280.0892), "kp1": (1304.2015, 231.0079)},
    {"id": "1S", "kp0": (1376.5033, 303.4936), "kp1": (1361.3988, 346.5324)},
    {"id": "1U", "kp0": (1436.9829, 326.1955), "kp1": (1412.4417, 360.7321)},
    {"id": "1V", "kp0": (1438.4635, 379.5347), "kp1": (1479.9189, 373.4477)},
    {"id": "1W", "kp0": (1529.3406, 627.1574), "kp1": (1523.6706, 674.6758)},
    {"id": "1X", "kp0": (1516.0321, 586.0894), "kp1": (1542.8628, 555.6953)},
    {"id": "1Y", "kp0": (1594.0313, 724.4414), "kp1": (1556.0399, 739.7229)},
    {"id": "1Z", "kp0": (1584.9853, 706.4385), "kp1": (1553.1609, 711.3796)},
    {"id": "1a", "kp0": (1562.8775, 645.5430), "kp1": (1550.8577, 604.8200)},
    {"id": "1b", "kp0": (1461.9228, 664.0658), "kp1": (1451.6144, 624.7434)},
    {"id": "1c", "kp0": (1484.0135, 603.7107), "kp1": (1481.1002, 647.0808)},
    {"id": "1d", "kp0": (1499.4618, 637.4213), "kp1": (1507.3219, 596.0927)},
    {"id": "1e", "kp0": (1592.0172, 278.7479), "kp1": (1555.6857, 253.1202)},
    {"id": "1f", "kp0": (1386.6767, 88.9917),  "kp1": (1422.4950, 119.4697)},
    {"id": "1g", "kp0": (1464.7032, 78.8103),  "kp1": (1473.1786, 121.9179)},
    {"id": "1h", "kp0": (1372.1385, 137.7361), "kp1": (1383.8450, 102.3735)},
    {"id": "1i", "kp0": (1500.0599, 161.6304), "kp1": (1511.4382, 122.4062)},
    {"id": "1j", "kp0": (1555.1241, 34.2932),  "kp1": (1553.5219, 76.2790)},
    {"id": "1k", "kp0": (950.3494,  266.6014), "kp1": (905.1027,  281.0716)},
]

IMG_W, IMG_H = 1696, 1031

# ─────────────────────────────────────────────────────────────────────────────
# Build arrays
# ─────────────────────────────────────────────────────────────────────────────

heads   = np.array([b["kp0"] for b in BEES], dtype=np.float32)   # (N, 2) x,y
tails   = np.array([b["kp1"] for b in BEES], dtype=np.float32)
centres = (heads + tails) / 2

diffs   = heads - tails
lengths = np.linalg.norm(diffs, axis=1, keepdims=True)
unit_vecs = diffs / np.where(lengths > 0, lengths, 1)            # (N, 2)

N = len(BEES)
print(f"{N} bees loaded")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatial flow field
# ─────────────────────────────────────────────────────────────────────────────

GRID_NX, GRID_NY = 48, 30   # interpolation grid resolution

gx = np.linspace(0, IMG_W, GRID_NX)
gy = np.linspace(0, IMG_H, GRID_NY)
GX, GY = np.meshgrid(gx, gy)
pts = np.column_stack([GX.ravel(), GY.ravel()])

# Interpolate u (x-component) and v (y-component) separately
U_grid = griddata(centres, unit_vecs[:, 0], pts, method="linear").reshape(GRID_NY, GRID_NX)
V_grid = griddata(centres, unit_vecs[:, 1], pts, method="linear").reshape(GRID_NY, GRID_NX)

# Fill NaNs (outside convex hull) with nearest
nan_mask = np.isnan(U_grid)
if nan_mask.any():
    U_nn = griddata(centres, unit_vecs[:, 0], pts, method="nearest").reshape(GRID_NY, GRID_NX)
    V_nn = griddata(centres, unit_vecs[:, 1], pts, method="nearest").reshape(GRID_NY, GRID_NX)
    U_grid[nan_mask] = U_nn[nan_mask]
    V_grid[nan_mask] = V_nn[nan_mask]

# Divergence and curl via finite differences
# Note: image y-axis is flipped relative to maths, but we treat it consistently
dy_arr = gy[1] - gy[0]
dx_arr = gx[1] - gx[0]
dVdx  = np.gradient(V_grid, dx_arr, axis=1)
dUdy  = np.gradient(U_grid, dy_arr, axis=0)
dUdx  = np.gradient(U_grid, dx_arr, axis=1)
dVdy  = np.gradient(V_grid, dy_arr, axis=0)
divergence = dUdx + dVdy     # sources (+) and sinks (−)
curl       = dVdx - dUdy     # clockwise (+) vs counter-clockwise (−)

# Speed (magnitude) for streamline colouring
speed = np.sqrt(U_grid**2 + V_grid**2)

# Load base image
src = cv2.imread("pipeline_output/bee_boxes/full_enhanced.png")
if src is None:
    raise FileNotFoundError("pipeline_output/bee_boxes/full_enhanced.png not found")
base_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) if len(src.shape) == 3 \
           else cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)

fig = plt.figure(figsize=(18, 12), facecolor="#0d0d1a")

# ── Panel A: streamlines over image ──────────────────────────────────────────
ax1 = fig.add_subplot(2, 2, (1, 2))   # top full width
ax1.imshow(base_rgb, alpha=0.45, extent=[0, IMG_W, IMG_H, 0], aspect="auto")

strm = ax1.streamplot(
    gx, gy, U_grid, -V_grid,          # negate V: image y↓ → plot y↑
    density=1.8,
    color=speed,
    cmap="plasma",
    linewidth=1.2,
    arrowsize=1.4,
    broken_streamlines=True,
)
cbar1 = fig.colorbar(strm.lines, ax=ax1, fraction=0.018, pad=0.01)
cbar1.set_label("vector magnitude", color="white", fontsize=8)
cbar1.ax.yaxis.set_tick_params(color="white", labelcolor="white")

# Raw bee arrows on top
ax1.quiver(centres[:, 0], centres[:, 1],
           unit_vecs[:, 0], -unit_vecs[:, 1],
           color="#00f5d4", alpha=0.9, scale=18, width=0.003,
           headwidth=4, headlength=5)

ax1.set_xlim(0, IMG_W); ax1.set_ylim(IMG_H, 0)
ax1.set_title("Spatial flow field — streamlines (plasma = magnitude)",
              color="white", fontsize=10)
ax1.axis("off")

# ── Panel B: divergence heatmap ───────────────────────────────────────────────
ax2 = fig.add_subplot(2, 2, 3)
div_lim = np.percentile(np.abs(divergence), 97)
im2 = ax2.imshow(divergence, cmap="RdBu_r", vmin=-div_lim, vmax=div_lim,
                 extent=[0, IMG_W, IMG_H, 0], aspect="auto", origin="upper")
ax2.scatter(centres[:, 0], centres[:, 1], s=8, c="white", alpha=0.6, zorder=3)
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
cbar2.set_label("divergence", color="white", fontsize=8)
cbar2.ax.yaxis.set_tick_params(color="white", labelcolor="white")
ax2.set_title("Divergence  (red = spreading out,  blue = converging)",
              color="white", fontsize=9)
ax2.axis("off")
ax2.set_facecolor("#0d0d1a")

# ── Panel C: curl heatmap ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(2, 2, 4)
curl_lim = np.percentile(np.abs(curl), 97)
im3 = ax3.imshow(curl, cmap="PuOr", vmin=-curl_lim, vmax=curl_lim,
                 extent=[0, IMG_W, IMG_H, 0], aspect="auto", origin="upper")
ax3.scatter(centres[:, 0], centres[:, 1], s=8, c="white", alpha=0.6, zorder=3)
cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.03, pad=0.02)
cbar3.set_label("curl", color="white", fontsize=8)
cbar3.ax.yaxis.set_tick_params(color="white", labelcolor="white")
ax3.set_title("Curl  (orange = clockwise,  purple = counter-clockwise)",
              color="white", fontsize=9)
ax3.axis("off")
ax3.set_facecolor("#0d0d1a")

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor("#0d0d1a")

plt.tight_layout(pad=0.8)
flow_path = "pipeline_output/bee_boxes/flow_field.png"
plt.savefig(flow_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Flow field → {flow_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Head-to-tail following chains
# ─────────────────────────────────────────────────────────────────────────────
# An edge A → B exists when:
#   • bee A's head (kp0) is within FOLLOW_DIST px of bee B's tail (kp1)
#   • A and B are heading in roughly the same direction (dot > DOT_THRESH)
# ─────────────────────────────────────────────────────────────────────────────

FOLLOW_DIST = 60   # px  — generous to account for annotation imprecision
DOT_THRESH  = 0.25 # cos similarity threshold (~75° tolerance)

# KD-tree on tails to find whose tail each head is near
tail_tree = KDTree(tails)
head_pts  = heads

# Build adjacency: following[i] = j means bee i's head is near bee j's tail
following = {}   # leader index → follower index (multiple followers allowed)
edges = []       # list of (follower_idx, leader_idx)

for i in range(N):
    idxs_near = tail_tree.query_ball_point(head_pts[i], FOLLOW_DIST)
    for j in idxs_near:
        if j == i:
            continue
        dot = float(unit_vecs[i] @ unit_vecs[j])
        if dot >= DOT_THRESH:
            edges.append((i, j))   # bee i follows bee j

print(f"\nFollowing edges found: {len(edges)}")

# Build chains by tracing connected paths greedily
# Map follower → leader (take the closest match if multiple)
follower_to_leader = {}
for (fol, lea) in edges:
    hd = heads[fol]
    tl = tails[lea]
    d  = float(np.linalg.norm(hd - tl))
    if fol not in follower_to_leader or d < follower_to_leader[fol][1]:
        follower_to_leader[fol] = (lea, d)

follower_to_leader = {f: v[0] for f, v in follower_to_leader.items()}
leader_to_followers = {}
for f, l in follower_to_leader.items():
    leader_to_followers.setdefault(l, []).append(f)

# Find chain starts: bees that are leaders but not themselves followers
chain_starts = [l for l in leader_to_followers if l not in follower_to_leader]
# Also include singletons that are followers but whose leader is not a follower
# (already covered above — chain_start is the ultimate leader)

def trace_chain(start):
    chain = [start]
    visited = {start}
    cur = start
    while cur in leader_to_followers:
        nexts = [f for f in leader_to_followers[cur] if f not in visited]
        if not nexts:
            break
        nxt = nexts[0]
        chain.append(nxt)
        visited.add(nxt)
        cur = nxt
    return chain

chains = [trace_chain(s) for s in chain_starts]
chains = [c for c in chains if len(c) >= 2]   # keep chains of 2+
chains.sort(key=len, reverse=True)
print(f"Chains (≥2 bees): {len(chains)}  |  "
      f"longest: {len(chains[0]) if chains else 0} bees")

# ── Draw on image ──────────────────────────────────────────────────────────────
canvas = base_rgb.copy()
# Darken base
canvas = (canvas.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)

# Distinct chain colours (HSV rainbow)
def chain_color(idx, total):
    hue = idx / max(total, 1)
    rgb = mcolors.hsv_to_rgb([hue, 0.95, 1.0])
    return tuple(int(v * 255) for v in rgb)   # RGB

# Draw all bees as dim grey arrows first
for i in range(N):
    t = (int(tails[i, 0]), int(tails[i, 1]))
    h = (int(heads[i, 0]), int(heads[i, 1]))
    dx, dy = unit_vecs[i]
    length_px = float(np.linalg.norm(heads[i] - tails[i]))
    tip = (int(heads[i, 0] + dx * 6), int(heads[i, 1] + dy * 6))
    bgr = (80, 80, 80)
    cv2.arrowedLine(canvas, t, tip, (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.35)
    cv2.arrowedLine(canvas, t, tip, bgr,        1, cv2.LINE_AA, tipLength=0.35)

# Draw chains with vibrant colours + connecting lines between head→tail
for ci, chain in enumerate(chains):
    col_rgb = chain_color(ci, len(chains))
    col_bgr = (col_rgb[2], col_rgb[1], col_rgb[0])
    col_bgr_dark = tuple(c // 3 for c in col_bgr)

    for step, idx in enumerate(chain):
        t = (int(tails[idx, 0]), int(tails[idx, 1]))
        h = (int(heads[idx, 0]), int(heads[idx, 1]))
        dx, dy = unit_vecs[idx]
        tip = (int(heads[idx, 0] + dx * 8), int(heads[idx, 1] + dy * 8))

        # Arrow
        cv2.arrowedLine(canvas, t, tip, (0, 0, 0), 5,  cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(canvas, t, tip, col_bgr,    2,  cv2.LINE_AA, tipLength=0.35)

        # Dashed connector: previous bee's head → this bee's tail
        if step > 0:
            prev_h = (int(heads[chain[step - 1], 0]),
                      int(heads[chain[step - 1], 1]))
            # Draw dashed line manually
            p1, p2 = np.array(prev_h), np.array(t)
            total_len = np.linalg.norm(p2 - p1)
            if total_len > 1:
                seg_len, gap_len = 8, 5
                n_segs = int(total_len / (seg_len + gap_len))
                for s in range(n_segs):
                    frac0 = s * (seg_len + gap_len) / total_len
                    frac1 = frac0 + seg_len / total_len
                    pt0 = tuple((p1 + (p2 - p1) * frac0).astype(int))
                    pt1 = tuple((p1 + (p2 - p1) * min(frac1, 1.0)).astype(int))
                    cv2.line(canvas, pt0, pt1, col_bgr, 1, cv2.LINE_AA)

        # Dot at bee centre + chain ID label on first bee
        cx, cy = int(centres[idx, 0]), int(centres[idx, 1])
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), -1)
        cv2.circle(canvas, (cx, cy), 3, col_bgr, -1)

        if step == 0:
            cv2.putText(canvas, f"C{ci+1}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"C{ci+1}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_bgr, 1, cv2.LINE_AA)

# Legend box
legend_y = 18
for ci, chain in enumerate(chains[:8]):   # show up to 8
    col_rgb = chain_color(ci, len(chains))
    col_bgr = (col_rgb[2], col_rgb[1], col_rgb[0])
    cv2.rectangle(canvas, (8, legend_y - 10), (22, legend_y + 2), col_bgr, -1)
    cv2.putText(canvas, f"C{ci+1}: {len(chain)} bees",
                (26, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (220, 220, 220), 1, cv2.LINE_AA)
    legend_y += 18

chain_path = "pipeline_output/bee_boxes/following_chains.png"
cv2.imwrite(chain_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print(f"Chains      → {chain_path}")
print("Done.")
