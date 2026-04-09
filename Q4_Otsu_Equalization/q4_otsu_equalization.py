import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# Load & convert to grayscale 
img_bgr  = cv2.imread("assets/a1images/emma.jpg")
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)   # uint8 [0, 255]

print(f"Image shape (color)     : {img_rgb.shape}")
print(f"Image shape (grayscale) : {img_gray.shape}")

# (a) Otsu Thresholding
thresh_val, binary_mask = cv2.threshold(
    img_gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(f"\nOtsu threshold value    : {thresh_val:.0f}  (out of 255)")

# foreground_mask: True where pixel is foreground
foreground_mask = binary_mask == 255
background_mask = binary_mask == 0

print(f"Foreground pixels       : {foreground_mask.sum():,}  "
      f"({100*foreground_mask.mean():.1f}%)")
print(f"Background pixels       : {background_mask.sum():,}  "
      f"({100*background_mask.mean():.1f}%)")


# (b) Histogram Equalization on Foreground ONLY
def equalize_histogram(image):
    """Same custom implementation from Q3."""
    hist   = np.zeros(256, dtype=np.int64)
    for v in image.ravel():
        hist[v] += 1
    pdf        = hist / image.size
    cdf        = np.cumsum(pdf)
    cdf_mapped = np.round(cdf * 255).astype(np.uint8)
    return cdf_mapped[image], hist, cdf_mapped


# Extract foreground pixels, equalize them, put back
fg_pixels        = img_gray[foreground_mask]          # 1-D array of fg pixels
fg_equalized, fg_hist_orig, fg_lut = equalize_histogram(fg_pixels.reshape(-1, 1))
fg_equalized     = fg_equalized.ravel()               # back to 1-D

# Build output image: equalized foreground, original background
result = img_gray.copy()
result[foreground_mask] = fg_equalized

print(f"\nForeground pixel range (before): [{fg_pixels.min()}, {fg_pixels.max()}]")
print(f"Foreground pixel range (after) : [{fg_equalized.min()}, {fg_equalized.max()}]")

# FIGURE 1 — Main results
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Q4 — Otsu Thresholding & Foreground Histogram Equalization",
             fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

# Row 0: Images 
panels = [
    (img_gray,     "Grayscale original",             "gray"),
    (binary_mask,  f"Otsu binary mask\nthreshold = {thresh_val:.0f}", "gray"),
    (result,       "Foreground equalized\n(background unchanged)", "gray"),
]
for col, (im, title, cmap) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

# Overlay: show mask boundary on original
overlay = img_rgb.copy()
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (0, 200, 100), 2)
ax_ov = fig.add_subplot(gs[0, 3])
ax_ov.imshow(overlay)
ax_ov.set_title("Foreground boundary\n(green contour)", fontsize=9)
ax_ov.axis("off")

# Row 1: Histograms of full grayscale image 
hist_gray, _   = np.histogram(img_gray.ravel(),  bins=256, range=(0, 255))
hist_result, _ = np.histogram(result.ravel(),    bins=256, range=(0, 255))
hist_fg_eq, _  = np.histogram(fg_equalized,      bins=256, range=(0, 255))

for col, (h, title, color) in enumerate([
    (hist_gray,   "Histogram — grayscale original",   "#444444"),
    (hist_fg_eq,  "Histogram — foreground pixels\n(after equalization)", "#185FA5"),
    (hist_result, "Histogram — full result image",    "#0F6E56"),
]):
    ax = fig.add_subplot(gs[1, col])
    ax.bar(range(256), h, width=1.0, color=color, alpha=0.85)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Intensity", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=6)

# Otsu threshold line on original histogram
ax_ot = fig.add_subplot(gs[1, 3])
ax_ot.bar(range(256), hist_gray, width=1.0, color="#444444", alpha=0.7)
ax_ot.axvline(thresh_val, color="#E24B4A", linewidth=2,
              label=f"Otsu threshold = {thresh_val:.0f}")
ax_ot.set_title("Otsu threshold on histogram", fontsize=8)
ax_ot.set_xlabel("Intensity", fontsize=7)
ax_ot.set_ylabel("Count", fontsize=7)
ax_ot.legend(fontsize=7)
ax_ot.tick_params(labelsize=6)

# Row 2: Side-by-side zoom crop to reveal hidden features
# Crop the interior room / window area where detail is hidden
H, W    = img_gray.shape
r1, r2  = int(H * 0.15), int(H * 0.75)
c1, c2  = int(W * 0.25), int(W * 0.85)

crop_orig   = img_gray[r1:r2, c1:c2]
crop_result = result[r1:r2, c1:c2]

ax_co = fig.add_subplot(gs[2, 0:2])
ax_co.imshow(crop_orig, cmap="gray", vmin=0, vmax=255)
ax_co.set_title("Cropped region — original\n(detail hidden in shadows/highlights)", fontsize=9)
ax_co.axis("off")

ax_cr = fig.add_subplot(gs[2, 2:4])
ax_cr.imshow(crop_result, cmap="gray", vmin=0, vmax=255)
ax_cr.set_title("Cropped region — after foreground equalization\n(hidden detail revealed)", fontsize=9)
ax_cr.axis("off")

plt.savefig("Q4_Otsu_Equalization/outputs/q4_output.png", dpi=150, bbox_inches="tight")
print("Saved -> q4_output.png")
plt.show()


# FIGURE 2 — Before / After comparison (clean, report-ready)
fig2, axes = plt.subplots(1, 2, figsize=(12, 6))
fig2.suptitle("Q4 — Before vs After Foreground Equalization", fontsize=12, fontweight="bold")

axes[0].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original grayscale", fontsize=11)
axes[0].axis("off")

axes[1].imshow(result, cmap="gray", vmin=0, vmax=255)
axes[1].set_title(f"Foreground equalized\n(Otsu threshold = {thresh_val:.0f})", fontsize=11)
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Q4_Otsu_Equalization/outputs/q4_before_after.png", dpi=150, bbox_inches="tight")
print("Saved -> q4_before_after.png")
plt.show()

