#!/usr/bin/env python3
"""
bee_behaviour_analysis.py
=========================
Two behavioural analyses from bee heading vectors:

  1. Rose diagram  — polar histogram of all headings
  2. Local alignment map — arrows coloured by how aligned each bee is
                           with its k nearest neighbours (0=random, 1=perfect)

Outputs → pipeline_output/bee_boxes/
  rose_diagram.png
  local_alignment.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import KDTree

# ── Keypoints (tail=kp1, head=kp0) ───────────────────────────────────────────
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

K_NEIGHBOURS = 5   # nearest neighbours for alignment score

# ─────────────────────────────────────────────────────────────────────────────
# Derive heading vectors and centres
# ─────────────────────────────────────────────────────────────────────────────

centres, unit_vecs, angles_deg = [], [], []

for bee in BEES:
    hx, hy = bee["kp0"]
    tx, ty = bee["kp1"]
    dx, dy = hx - tx, hy - ty
    length = (dx**2 + dy**2) ** 0.5
    if length < 1e-3:
        continue
    cx = (hx + tx) / 2
    cy = (hy + ty) / 2
    centres.append([cx, cy])
    unit_vecs.append([dx / length, dy / length])
    # convert image-space angle (y-down) to compass bearing (N=up=0°, CW)
    # atan2 gives angle from +x axis; we want angle from -y (up) axis, CW
    ang = np.degrees(np.arctan2(dx, -dy)) % 360
    angles_deg.append(ang)

centres   = np.array(centres)
unit_vecs = np.array(unit_vecs)
angles_deg = np.array(angles_deg)
N = len(centres)
print(f"{N} valid bee vectors")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rose diagram
# ─────────────────────────────────────────────────────────────────────────────

N_BINS = 24   # 15° per bin
bin_edges = np.linspace(0, 2 * np.pi, N_BINS + 1)
angles_rad = np.radians(angles_deg)
counts, _ = np.histogram(angles_rad, bins=bin_edges)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
width = 2 * np.pi / N_BINS

# Mean resultant vector — length 0=uniform, 1=all same direction
mean_x = np.mean(unit_vecs[:, 0])
mean_y = np.mean(unit_vecs[:, 1])
R = (mean_x**2 + mean_y**2) ** 0.5          # Rayleigh r
mean_ang_deg = np.degrees(np.arctan2(mean_y, mean_x))

fig = plt.figure(figsize=(7, 7), facecolor="#1a1a2e")
ax  = fig.add_subplot(111, polar=True, facecolor="#16213e")

# Bars coloured by count
norm   = plt.Normalize(vmin=0, vmax=counts.max())
colors = cm.plasma(norm(counts))
bars   = ax.bar(bin_centres, counts, width=width * 0.92,
                bottom=0, color=colors, edgecolor="#1a1a2e", linewidth=0.5,
                align="center")

# Mean resultant vector arrow
mean_ang_rad = np.arctan2(mean_y, mean_x)
ax.annotate("", xy=(mean_ang_rad, R * counts.max()),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#00f5d4",
                            lw=2.5, mutation_scale=20))

# Styling
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)           # clockwise = compass convention
ax.set_rlabel_position(22.5)
ax.tick_params(colors="white", labelsize=9)
ax.spines["polar"].set_color("#444")
ax.set_facecolor("#16213e")
for label in ax.get_xticklabels():
    label.set_color("white")
for label in ax.get_yticklabels():
    label.set_color("#aaa")

cardinal = {0: "N", 90: "E", 180: "S", 270: "W"}
ax.set_xticks(np.radians(list(cardinal.keys())))
ax.set_xticklabels(list(cardinal.values()), fontsize=11, color="white")

ax.set_title(f"Bee heading distribution  (n={N})\n"
             f"Rayleigh r = {R:.3f}  |  mean direction = {mean_ang_deg:.1f}°",
             color="white", pad=18, fontsize=11)

sm = cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                    fraction=0.03, pad=0.1, shrink=0.6)
cbar.set_label("count", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

plt.tight_layout()
rose_path = "pipeline_output/bee_boxes/rose_diagram.png"
plt.savefig(rose_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Rose diagram  → {rose_path}  (Rayleigh r={R:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Local alignment map
# ─────────────────────────────────────────────────────────────────────────────

tree = KDTree(centres)
# k+1 because the query includes the point itself
dists, idxs = tree.query(centres, k=min(K_NEIGHBOURS + 1, N))

align_scores = np.zeros(N)
for i in range(N):
    neighbours = idxs[i][1:]   # exclude self
    # mean cosine similarity (dot product of unit vectors)
    sims = unit_vecs[neighbours] @ unit_vecs[i]
    align_scores[i] = float(np.mean(sims))   # -1 (opposite) … +1 (aligned)

# Normalise 0–1 for colourmap
align_norm = (align_scores - align_scores.min()) / (np.ptp(align_scores) + 1e-9)

print(f"Alignment scores  min={align_scores.min():.3f}  "
      f"mean={align_scores.mean():.3f}  max={align_scores.max():.3f}")

# Load enhanced image as colour base
src = cv2.imread("pipeline_output/bee_boxes/full_enhanced.png")
if src is None:
    raise FileNotFoundError("pipeline_output/bee_boxes/full_enhanced.png not found")
canvas = src.copy() if len(src.shape) == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

# Darken base slightly so arrows pop
canvas = (canvas.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)

# Colourmap: blue (misaligned) → green → yellow → red (well-aligned)
cmap = cm.RdYlGn   # reversed: green=aligned, red=misaligned — intuitive

OUTLINE_COLOR = (0, 0, 0)

for i, bee in enumerate(BEES):
    hx, hy = bee["kp0"]
    tx, ty = bee["kp1"]
    dx, dy = hx - tx, hy - ty
    length = (dx**2 + dy**2) ** 0.5
    if length < 1e-3:
        continue

    tail = (int(round(tx)), int(round(ty)))
    head = (int(round(hx)), int(round(hy)))

    # Extend tip slightly past head
    extend = 7
    tip = (int(round(hx + dx / length * extend)),
           int(round(hy + dy / length * extend)))

    # Colour from alignment score via RdYlGn
    r_f, g_f, b_f, _ = cmap(align_norm[i])
    color_bgr = (int(b_f * 255), int(g_f * 255), int(r_f * 255))

    cv2.arrowedLine(canvas, tail, tip, OUTLINE_COLOR, 4,
                    cv2.LINE_AA, tipLength=0.35)
    cv2.arrowedLine(canvas, tail, tip, color_bgr, 2,
                    cv2.LINE_AA, tipLength=0.35)

# ── Matplotlib overlay: add colourbar legend via figure ──────────────────────
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(figsize=(rgb.shape[1] / 150, rgb.shape[0] / 150),
                       facecolor="#111")
ax.imshow(rgb)
ax.axis("off")

# Colourbar
sm = cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(vmin=align_scores.min(), vmax=align_scores.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                    fraction=0.025, pad=0.01, shrink=0.55)
cbar.set_label(f"Local alignment score\n(k={K_NEIGHBOURS} neighbours)",
               color="white", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)

ax.set_title("Bee local heading alignment", color="white",
             fontsize=10, pad=6)

plt.tight_layout(pad=0.3)
align_path = "pipeline_output/bee_boxes/local_alignment.png"
plt.savefig(align_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Alignment map → {align_path}")
print("Done.")
