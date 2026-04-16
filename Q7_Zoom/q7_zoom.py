import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os

# (a) Nearest-Neighbour Interpolation
def zoom_nearest(image, factor):

    assert 0 < factor, "Factor must be positive"
    src_h, src_w = image.shape[:2]
    dst_h = int(round(src_h * factor))
    dst_w = int(round(src_w * factor))

    # Build output pixel coordinate grids
    i_out = np.arange(dst_h)          # rows  0 … dst_h-1
    j_out = np.arange(dst_w)          # cols  0 … dst_w-1

    # Map output → input coordinates (inverse mapping)
    i_in = np.clip(np.round(i_out / factor).astype(int), 0, src_h - 1)
    j_in = np.clip(np.round(j_out / factor).astype(int), 0, src_w - 1)

    # Index: first select rows, then columns (broadcasting via np.ix_)
    if image.ndim == 2:
        zoomed = image[np.ix_(i_in, j_in)]
    else:
        zoomed = image[np.ix_(i_in, j_in)]   # works for (H,W,C) too

    return zoomed.astype(np.uint8)


# (b) Bilinear Interpolation
def zoom_bilinear(image, factor):
   
    assert 0 < factor, "Factor must be positive"
    src_h, src_w = image.shape[:2]
    dst_h = int(round(src_h * factor))
    dst_w = int(round(src_w * factor))

    img_f = image.astype(np.float32)

    # Continuous source coordinates for every output pixel
    i_src = np.arange(dst_h) / factor           # shape (dst_h,)
    j_src = np.arange(dst_w) / factor           # shape (dst_w,)

    # Floor / ceil neighbours — clamped to valid range
    i0 = np.clip(np.floor(i_src).astype(int), 0, src_h - 1)
    i1 = np.clip(i0 + 1,                       0, src_h - 1)
    j0 = np.clip(np.floor(j_src).astype(int), 0, src_w - 1)
    j1 = np.clip(j0 + 1,                       0, src_w - 1)

    # Fractional offsets (how far between the two neighbours)
    di = (i_src - np.floor(i_src))[:, np.newaxis]   # shape (dst_h, 1)
    dj = (j_src - np.floor(j_src))[np.newaxis, :]   # shape (1, dst_w)

    if img_f.ndim == 2:
        # Scalar case — straightforward
        top    = img_f[np.ix_(i0, j0)] * (1 - dj) + img_f[np.ix_(i0, j1)] * dj
        bottom = img_f[np.ix_(i1, j0)] * (1 - dj) + img_f[np.ix_(i1, j1)] * dj
        zoomed = top * (1 - di) + bottom * di
    else:
        # Multi-channel: process all channels at once
        # img_f[i0][:, j0] etc. → use advanced indexing per channel
        C      = img_f.shape[2]
        zoomed = np.zeros((dst_h, dst_w, C), dtype=np.float32)
        for c in range(C):
            ch     = img_f[:, :, c]
            top    = ch[np.ix_(i0, j0)] * (1 - dj) + ch[np.ix_(i0, j1)] * dj
            bottom = ch[np.ix_(i1, j0)] * (1 - dj) + ch[np.ix_(i1, j1)] * dj
            zoomed[:, :, c] = top * (1 - di) + bottom * di

    return np.clip(zoomed, 0, 255).astype(np.uint8)


# Unified zoom function
def zoom(image, factor, method="bilinear"):

    if method == "nearest":
        return zoom_nearest(image, factor)
    elif method == "bilinear":
        return zoom_bilinear(image, factor)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'nearest' or 'bilinear'.")


# Normalised SSD metric
def normalised_ssd(img_a, img_b):

    assert img_a.shape == img_b.shape, \
        f"Shape mismatch: {img_a.shape} vs {img_b.shape}"
    diff = img_a.astype(np.float64) - img_b.astype(np.float64)
    return np.sum(diff ** 2) / img_a.size


# Test pairs: (small image path, large original path, scale factor)
BASE = "assets/a1images/a1q8images"
test_pairs = [
    (f"{BASE}/im01small.png",         f"{BASE}/im01.png",    4.0,  "im01"),
    (f"{BASE}/im02small.png",         f"{BASE}/im02.png",    4.0,  "im02"),
    (f"{BASE}/im03small.png",         f"{BASE}/im03.png",    4.0,  "im03"),
    (f"{BASE}/taylor_small.jpg",      f"{BASE}/taylor.jpg",  5.0,  "taylor"),
    (f"{BASE}/taylor_very_small.jpg", f"{BASE}/taylor.jpg",  20.0, "taylor_vsmall"),
]

print("=" * 70)
print(f"{'Image':<18} {'Factor':>7}  {'SSD Nearest':>14}  {'SSD Bilinear':>14}  {'Winner'}")
print("=" * 70)

results = []
for small_path, large_path, factor, name in test_pairs:
    small = np.array(Image.open(small_path))
    large = np.array(Image.open(large_path))

    # Zoom small → match large size
    nn  = zoom(small, factor, method="nearest")
    bl  = zoom(small, factor, method="bilinear")

    # Crop to exact original size (rounding may give ±1 pixel)
    H, W = large.shape[:2]
    nn  = nn[:H, :W]
    bl  = bl[:H, :W]

    ssd_nn = normalised_ssd(large, nn)
    ssd_bl = normalised_ssd(large, bl)
    winner = "Bilinear ✓" if ssd_bl < ssd_nn else "Nearest  ✓"

    print(f"{name:<18} {factor:>7.1f}  {ssd_nn:>14.2f}  {ssd_bl:>14.2f}  {winner}")
    results.append((name, factor, small, large, nn, bl, ssd_nn, ssd_bl))

print("=" * 70)


# FIGURE 1 — Visual comparison for im01 (4×)
name, factor, small, large, nn, bl, ssd_nn, ssd_bl = results[0]  # im01

fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"Q7 — Image Zooming  ({name}, factor={factor}×)",
             fontsize=13, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

# Row 0 — full images
H, W = large.shape[:2]
for col, (im, title) in enumerate([
    (small,  f"Input (small)\n{small.shape[1]}×{small.shape[0]}"),
    (nn,     f"(a) Nearest-neighbour\nSSD = {ssd_nn:.1f}"),
    (bl,     f"(b) Bilinear\nSSD = {ssd_bl:.1f}"),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(im)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

# Row 1 — zoomed crops showing interpolation artefacts clearly
r1, r2 = H//4, H//4 + H//6
c1, c2 = W//4, W//4 + W//6

for col, (im, title) in enumerate([
    (large[r1:r2, c1:c2],  "Original (crop)"),
    (nn[r1:r2, c1:c2],     "Nearest (crop)\n— blocky artefacts visible"),
    (bl[r1:r2, c1:c2],     "Bilinear (crop)\n— smoother transitions"),
]):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(im)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.savefig("Q7_Zoom/outputs/q7_zoom_im01.png", dpi=150, bbox_inches="tight")
print("Saved -> q7_zoom_im01.png")
plt.show()


# FIGURE 2 — SSD bar chart summary across all test pairs
labels   = [r[0] + f"\n×{r[1]:.0f}" for r in results]
ssd_nns  = [r[6] for r in results]
ssd_bls  = [r[7] for r in results]

x     = np.arange(len(results))
width = 0.35

fig2, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, ssd_nns, width, label="Nearest-neighbour",
               color="#185FA5", alpha=0.85)
bars2 = ax.bar(x + width/2, ssd_bls, width, label="Bilinear",
               color="#0F6E56", alpha=0.85)

ax.set_xlabel("Test image", fontsize=11)
ax.set_ylabel("Normalised SSD  (lower = better)", fontsize=11)
ax.set_title("Q7 — Reconstruction Quality: Nearest-neighbour vs Bilinear",
             fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# Annotate bars with values
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8,
            color="#185FA5")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8,
            color="#0F6E56")

plt.tight_layout()
plt.savefig("Q7_Zoom/outputs/q7_ssd_comparison.png", dpi=150, bbox_inches="tight")
print("Saved -> q7_ssd_comparison.png")
plt.show()


# FIGURE 3 — Extreme zoom (taylor_very_small ×20) to exaggerate differences
_, _, _, large_t, nn_t, bl_t, ssd_nn_t, ssd_bl_t = results[4]  # taylor_very_small

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle(f"Q7 — Extreme Zoom ×20 (taylor_very_small → taylor)\n"
              f"Nearest SSD={ssd_nn_t:.1f}   Bilinear SSD={ssd_bl_t:.1f}",
              fontsize=12, fontweight="bold")

for ax, (im, title) in zip(axes3, [
    (large_t,  "Original large"),
    (nn_t,     "Nearest-neighbour\n(blocky pixelation)"),
    (bl_t,     "Bilinear\n(smooth but blurry)"),
]):
    ax.imshow(im)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("Q7_Zoom/outputs/q7_zoom_extreme.png", dpi=150, bbox_inches="tight")
print("Saved -> q7_zoom_extreme.png")
plt.show()
