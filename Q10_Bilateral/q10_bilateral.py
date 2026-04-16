import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import time

# Load image — use shells.tif (good mix of smooth + sharp-edged regions)
img_bgr  = cv2.imread("assets/a1images/shells.tif")
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

print(f"Image : shells.tif  {img_gray.shape}")


# (a) Manual Bilateral Filter
def bilateral_filter_manual(image, diameter, sigma_s, sigma_r):
    
    H, W   = image.shape
    radius = diameter // 2
    output = np.zeros_like(image)

    # Pre-build spatial Gaussian weights for the kernel window
    ky, kx   = np.mgrid[-radius:radius+1, -radius:radius+1]   # both (d,d)
    spatial_w = np.exp(-(kx**2 + ky**2) / (2 * sigma_s**2))   # (d,d)

    # Pad image so border pixels have a full neighbourhood
    pad   = radius
    img_p = np.pad(image, pad, mode="reflect")

    for i in range(H):
        for j in range(W):
            # Extract the neighbourhood patch
            patch = img_p[i:i+diameter, j:j+diameter]   # (d,d)

            # Range weight: Gaussian of intensity difference from centre pixel
            centre   = image[i, j]
            range_w  = np.exp(-((patch - centre)**2) / (2 * sigma_r**2))

            # Combined weight
            combined = spatial_w * range_w                # (d,d)

            # Normalised weighted average
            output[i, j] = np.sum(combined * patch) / np.sum(combined)

    return output


# Run on a smaller crop first for speed, then full image 
# Parameters: diameter=9, σ_s=10 (spatial), σ_r=25 (range)
DIAMETER = 9
SIGMA_S  = 10    # larger → smoother (more spatial averaging)
SIGMA_R  = 25    # larger → less edge-preserving (approaches Gaussian blur)

print(f"\nRunning manual bilateral filter (d={DIAMETER}, σs={SIGMA_S}, σr={SIGMA_R})...")
print("(This uses nested loops — may take ~30–60 seconds for a full image)")

# Work on a downscaled version for the manual filter to keep it fast
H, W    = img_gray.shape
scale   = 0.4
small   = cv2.resize(img_gray, (int(W*scale), int(H*scale)))

t0 = time.time()
bf_manual_small = bilateral_filter_manual(small, DIAMETER, SIGMA_S, SIGMA_R)
elapsed = time.time() - t0
print(f"Manual filter done in {elapsed:.1f}s  (on {small.shape} image)")


# (b) Gaussian smoothing via cv2.GaussianBlur
gauss_full = cv2.GaussianBlur(img_gray, ksize=(9, 9), sigmaX=SIGMA_S)
gauss_small = cv2.GaussianBlur(small,   ksize=(9, 9), sigmaX=SIGMA_S)


# (c) OpenCV bilateral filter (on full image — fast C++ implementation)
bf_opencv_full  = cv2.bilateralFilter(img_gray, d=DIAMETER,
                                      sigmaColor=SIGMA_R, sigmaSpace=SIGMA_S)
bf_opencv_small = cv2.bilateralFilter(small, d=DIAMETER,
                                      sigmaColor=SIGMA_R, sigmaSpace=SIGMA_S)

print(f"\nManual vs OpenCV bilateral (on small image):")
diff = np.abs(bf_manual_small - bf_opencv_small)
print(f"  Max difference  : {diff.max():.4f}")
print(f"  Mean difference : {diff.mean():.4f}")


# FIGURE 1 — Full image: all three methods on full-resolution image
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Q10 — Bilateral Filtering vs Gaussian Smoothing",
             fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

def to_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)

# Row 0: full-image panels
full_panels = [
    (img_gray,      "Original grayscale"),
    (gauss_full,    f"(b) cv2.GaussianBlur\n(σ={SIGMA_S}, 9×9)\nSmooths edges"),
    (bf_opencv_full,f"(c) cv2.bilateralFilter\n(d={DIAMETER}, σs={SIGMA_S}, σr={SIGMA_R})\nEdges preserved"),
    (np.abs(gauss_full - bf_opencv_full).astype(np.uint8),
                    "Difference map\nGaussian − Bilateral\n(edge regions differ most)"),
]
for col, (im, title) in enumerate(full_panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(to_uint8(im), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

# Row 1: small image panels (manual filter result)
small_panels = [
    (small,             "Original (40% scale)"),
    (gauss_small,       "(b) Gaussian (small)"),
    (bf_opencv_small,   "(c) OpenCV bilateral (small)"),
    (bf_manual_small,   f"(d) Manual bilateral (small)\n{elapsed:.1f}s runtime"),
]
for col, (im, title) in enumerate(small_panels):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(to_uint8(im), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.savefig("Q10_Bilateral/outputs/q10_output.png", dpi=150, bbox_inches="tight")
print("\nSaved → q10_output.png")
plt.show()


# FIGURE 2 — Zoomed crop: edge preservation is the key comparison
Hs, Ws = small.shape
r1, r2 = int(Hs * 0.15), int(Hs * 0.65)
c1, c2 = int(Ws * 0.10), int(Ws * 0.70)

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 5))
fig2.suptitle("Q10 — Zoomed Crop: Edge Preservation Comparison",
              fontsize=12, fontweight="bold")

crop_panels = [
    (small,            "Original"),
    (gauss_small,      f"Gaussian\n(edges blurred)"),
    (bf_opencv_small,  f"OpenCV bilateral\n(edges sharp)"),
    (bf_manual_small,  f"Manual bilateral\n(edges sharp)"),
]
for ax, (im, title) in zip(axes2, crop_panels):
    ax.imshow(to_uint8(im)[r1:r2, c1:c2], cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("Q10_Bilateral/outputs/q10_zoom_crop.png", dpi=150, bbox_inches="tight")
print("Saved -> q10_zoom_crop.png")
plt.show()


# FIGURE 3 — σ_r effect: show how range sigma controls edge preservation
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 5))
fig3.suptitle("Q10 — Effect of σ_r (range sigma) on Edge Preservation\n"
              f"(OpenCV bilateral, d={DIAMETER}, σ_s={SIGMA_S})",
              fontsize=11, fontweight="bold")

for ax, sigma_r in zip(axes3, [5, 25, 75, 150]):
    result = cv2.bilateralFilter(small, d=DIAMETER,
                                 sigmaColor=sigma_r, sigmaSpace=SIGMA_S)
    ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    label = "edge-preserving" if sigma_r <= 25 else ("Gaussian-like" if sigma_r >= 75 else "")
    ax.set_title(f"σ_r = {sigma_r}\n{label}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("Q10_Bilateral/outputs/q10_sigma_r_effect.png", dpi=150, bbox_inches="tight")
print("Saved -> q10_sigma_r_effect.png")
plt.show()


# FIGURE 4 — 1D profile through an edge: the key analytical plot
row_idx = Hs // 2
x = np.arange(Ws)

fig4, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, small[row_idx, :],            color="#444444", lw=1.5, label="Original",           alpha=0.8)
ax.plot(x, gauss_small[row_idx, :],      color="#185FA5", lw=2,   label=f"Gaussian (σ={SIGMA_S})")
ax.plot(x, bf_opencv_small[row_idx, :],  color="#0F6E56", lw=2,   label=f"OpenCV bilateral (σr={SIGMA_R})")
ax.plot(x, bf_manual_small[row_idx, :],  color="#A32D2D", lw=1.5, label="Manual bilateral",   ls="--")

ax.set_xlabel("Column index", fontsize=11)
ax.set_ylabel("Intensity", fontsize=11)
ax.set_title("Q10 — 1D Profile: Gaussian blurs edges, Bilateral preserves them",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Q10_Bilateral/outputs/q10_1d_profile.png", dpi=150, bbox_inches="tight")
print("Saved -> q10_1d_profile.png")
plt.show()
