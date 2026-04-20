import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# Load the noisy image (taylor.jpg — has salt & pepper noise)
BASE     = "assets"
img_bgr  = cv2.imread(f"{BASE}/corrupted.jpg")
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

print(f"Image shape : {img_gray.shape}")
print(f"Pixel range : [{img_gray.min()}, {img_gray.max()}]")

# Estimate noise level — count extreme pixels (salt=255, pepper=0)
n_salt   = np.sum(img_gray == 255)
n_pepper = np.sum(img_gray == 0)
total    = img_gray.size
print(f"\nSalt   pixels (=255) : {n_salt:,}  ({100*n_salt/total:.1f}%)")
print(f"Pepper pixels (=0)   : {n_pepper:,}  ({100*n_pepper/total:.1f}%)")
print(f"Total noise pixels   : {n_salt+n_pepper:,}  ({100*(n_salt+n_pepper)/total:.1f}%)")

# (a) Gaussian Smoothing — multiple kernel sizes
gauss_3 = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=1)
gauss_5 = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=1)
gauss_7 = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=1)


# (b) Median Filtering — multiple kernel sizes
median_3 = cv2.medianBlur(img_gray, ksize=3)
median_5 = cv2.medianBlur(img_gray, ksize=5)
median_7 = cv2.medianBlur(img_gray, ksize=7)


# Quality metric: PSNR (Peak Signal-to-Noise Ratio)
# Higher PSNR = less noise remaining
def psnr(original, filtered):
    """PSNR in dB — higher is better."""
    mse = np.mean((original.astype(np.float64) - filtered.astype(np.float64))**2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

# Remaining noise: count extreme pixels after filtering
def noise_remaining(img):
    return np.sum(img == 255) + np.sum(img == 0)

print("\nFilter Quality Comparison (PSNR — higher is better):")
print("=" * 55)
print(f"{'Filter':<22}  {'PSNR (dB)':>10}  {'Noise pixels left':>18}")
print("=" * 55)
for name, img_f in [
    ("Gaussian  3×3",  gauss_3),
    ("Gaussian  5×5",  gauss_5),
    ("Gaussian  7×7",  gauss_7),
    ("Median    3×3",  median_3),
    ("Median    5×5",  median_5),
    ("Median    7×7",  median_7),
]:
    p = psnr(img_gray, img_f)
    n = noise_remaining(img_f)
    print(f"{name:<22}  {p:>10.2f}  {n:>18,}")
print("=" * 55)


# FIGURE 1 — Main 3×3 grid comparison
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Q8 — Salt & Pepper Noise Removal: Gaussian vs Median Filtering",
             fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

panels = [
    (img_gray,  "Original (noisy)",        0, 0),
    (gauss_3,   "(a) Gaussian  3×3",       0, 1),
    (gauss_5,   "(a) Gaussian  5×5",       0, 2),
    (median_3,  "(b) Median    3×3",       1, 0),
    (median_5,  "(b) Median    5×5",       1, 1),
    (median_7,  "(b) Median    7×7",       1, 2),
    (gauss_7,   "(a) Gaussian  7×7",       2, 0),
    # Show absolute difference maps
    (np.abs(img_gray.astype(np.int32) - median_5.astype(np.int32)).astype(np.uint8),
             "Median 5×5 — change map",   2, 1),
    (np.abs(img_gray.astype(np.int32) - gauss_5.astype(np.int32)).astype(np.uint8),
             "Gaussian 5×5 — change map", 2, 2),
]

for (im, title, row, col) in panels:
    ax = fig.add_subplot(gs[row, col])
    cmap = "hot" if "change" in title.lower() else "gray"
    ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.savefig("Q8_Salt_Pepper/outputs/q8_output.png", dpi=150, bbox_inches="tight")
print("Saved -> q8_output.png")
plt.show()

# FIGURE 2 — Zoomed crop: visually reveal the key differences
H, W  = img_gray.shape
r1, r2 = int(H * 0.15), int(H * 0.55)
c1, c2 = int(W * 0.30), int(W * 0.75)

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 5))
fig2.suptitle("Q8 — Zoomed Crop: Gaussian vs Median (5×5 kernel)",
              fontsize=12, fontweight="bold")

crops = [
    (img_gray, "Noisy original"),
    (gauss_5,  f"Gaussian 5×5\nPSNR={psnr(img_gray, gauss_5):.1f} dB\n(blurs edges, smears noise)"),
    (median_5, f"Median 5×5\nPSNR={psnr(img_gray, median_5):.1f} dB\n(sharp edges preserved)"),
    (median_3, f"Median 3×3\nPSNR={psnr(img_gray, median_3):.1f} dB\n(less smoothing)"),
]
for ax, (im, title) in zip(axes2, crops):
    ax.imshow(im[r1:r2, c1:c2], cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("Q8_Salt_Pepper/outputs/q8_zoom_crop.png", dpi=150, bbox_inches="tight")
print("Saved -> q8_zoom_crop.png")
plt.show()


# FIGURE 3 — 1D signal slice to explain WHY median wins
# Take a single row through the image to show the filtering effect on a 1D signal
row_idx = H // 3
row_orig   = img_gray[row_idx, :].astype(np.float32)
row_gauss  = gauss_5[row_idx,  :].astype(np.float32)
row_median = median_5[row_idx, :].astype(np.float32)

# Find noise spike positions in this row
spike_pos = np.where((row_orig == 0) | (row_orig == 255))[0]

fig3, axes3 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig3.suptitle("Q8 — 1D Signal Slice Showing Filter Behaviour on S&P Noise",
              fontsize=12, fontweight="bold")

col_range = slice(100, 400)  # zoom into an interesting section

for ax, (sig, title, color) in zip(axes3, [
    (row_orig,   "Original row (with S&P spikes)",      "#444444"),
    (row_gauss,  "After Gaussian 5×5 (spikes spread into neighbours)", "#185FA5"),
    (row_median, "After Median 5×5   (spikes completely removed)",     "#0F6E56"),
]):
    ax.plot(sig[col_range], color=color, linewidth=1.2)
    # Mark spike positions
    spikes_in_range = spike_pos[(spike_pos >= 100) & (spike_pos < 400)] - 100
    if len(spikes_in_range):
        ax.scatter(spikes_in_range, sig[col_range][spikes_in_range],
                   color="#E24B4A", s=20, zorder=5, label="S&P noise spike")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Intensity", fontsize=9)
    ax.grid(True, alpha=0.3)
    if len(spikes_in_range):
        ax.legend(fontsize=8)

axes3[-1].set_xlabel("Column index", fontsize=10)
plt.tight_layout()
plt.savefig("Q8_Salt_Pepper/outputs/q8_1d_signal.png", dpi=150, bbox_inches="tight")
print("Saved -> q8_1d_signal.png")
plt.show()
