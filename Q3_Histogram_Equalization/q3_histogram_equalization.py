import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# ── Load image ─────────────────────────────────────────────────────────────────
img_bgr = cv2.imread("assets/runway.png", cv2.IMREAD_GRAYSCALE)
img = img_bgr 

print(f"Image shape : {img.shape}")
print(f"Pixel range : [{img.min()}, {img.max()}]")
print(f"Total pixels: {img.size}")

# CUSTOM HISTOGRAM EQUALIZATION 
def equalize_histogram(image):

    # Step 1: Histogram 
    hist = np.zeros(256, dtype=np.int64)
    for pixel_value in image.ravel():         
        hist[pixel_value] += 1

    # Step 2: PDF (Probability Density Function) 
    total_pixels = image.size               
    pdf = hist / total_pixels                  

    # Step 3: CDF (Cumulative Distribution Function) 
    cdf = np.cumsum(pdf)                    

    # Step 4: Scale CDF to [0, 255] — this IS the intensity mapping 
    cdf_mapped = np.round(cdf * 255).astype(np.uint8)   # shape (256,)

    # Step 5: Apply the mapping as a lookup table 
    equalized = cdf_mapped[image]             

    return equalized, hist, pdf, cdf, cdf_mapped


# Run custom equalization
eq_custom, hist_orig, pdf_orig, cdf_orig, lut = equalize_histogram(img)

# OpenCV reference for validation 
eq_opencv = cv2.equalizeHist(img)

# Compute difference
diff = np.abs(eq_custom.astype(np.int32) - eq_opencv.astype(np.int32))
print("\nValidation vs OpenCV cv2.equalizeHist():")
print(f"  Max pixel difference : {diff.max()}")
print(f"  Mean difference      : {diff.mean():.4f}")
print(f"  Identical pixels     : {(diff == 0).sum()} / {diff.size}  "
      f"({100*(diff==0).mean():.1f}%)")


# FIGURE 1 — Main results
fig = plt.figure(figsize=(16, 11))
fig.suptitle("Q3 — Custom Histogram Equalization on Runway Image",
             fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

# Row 0: Images
for col, (im, title) in enumerate([
    (img,       "Original"),
    (eq_custom, "Custom equalization\n(our implementation)"),
    (eq_opencv, "OpenCV cv2.equalizeHist()\n(reference)"),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

# Row 1: Histograms 
hist_eq_custom, _ = np.histogram(eq_custom.ravel(), bins=256, range=(0, 255))
hist_eq_opencv, _ = np.histogram(eq_opencv.ravel(), bins=256, range=(0, 255))

for col, (h, title, color) in enumerate([
    (hist_orig,      "Histogram — Original",        "#444444"),
    (hist_eq_custom, "Histogram — Custom equalized", "#185FA5"),
    (hist_eq_opencv, "Histogram — OpenCV equalized", "#0F6E56"),
]):
    ax = fig.add_subplot(gs[1, col])
    ax.bar(range(256), h, color=color, alpha=0.85, width=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Intensity", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=7)

# Row 2: CDF + LUT 
ax_cdf = fig.add_subplot(gs[2, 0])
ax_cdf.plot(range(256), cdf_orig * 255, color="#185FA5", linewidth=2)
ax_cdf.set_title("CDF (scaled to [0,255])\n= the intensity mapping", fontsize=9)
ax_cdf.set_xlabel("Input intensity", fontsize=8)
ax_cdf.set_ylabel("Output intensity", fontsize=8)
ax_cdf.tick_params(labelsize=7)
ax_cdf.grid(True, alpha=0.3)

ax_lut = fig.add_subplot(gs[2, 1])
ax_lut.plot(range(256), lut, color="#A32D2D", linewidth=2)
ax_lut.plot([0, 255], [0, 255], "--", color="#888780", linewidth=1, label="Identity")
ax_lut.set_title("Lookup table T(r)\n(rounded CDF mapping)", fontsize=9)
ax_lut.set_xlabel("Input intensity  r", fontsize=8)
ax_lut.set_ylabel("Output intensity  T(r)", fontsize=8)
ax_lut.legend(fontsize=8)
ax_lut.tick_params(labelsize=7)
ax_lut.grid(True, alpha=0.3)

ax_diff = fig.add_subplot(gs[2, 2])
ax_diff.imshow(diff, cmap="hot", vmin=0, vmax=5)
ax_diff.set_title(f"Difference map\n(custom vs OpenCV)\nmax={diff.max()}", fontsize=9)
ax_diff.axis("off")

plt.savefig("Q3_Histogram_Equalization/outputs/q3_output.png", dpi=150, bbox_inches="tight")
print("Saved -> q3_output.png")
plt.show()


# FIGURE 2 — Step-by-step algorithm walkthrough

fig2, axes = plt.subplots(1, 3, figsize=(13, 4))
fig2.suptitle("Q3 — Equalization Algorithm: PDF → CDF → Mapping",
              fontsize=12, fontweight="bold")

axes[0].bar(range(256), pdf_orig, width=1.0, color="#185FA5", alpha=0.85)
axes[0].set_title("Step 2: PDF\n(normalized histogram)", fontsize=10)
axes[0].set_xlabel("Intensity", fontsize=9)
axes[0].set_ylabel("Probability", fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(256), cdf_orig, color="#0F6E56", linewidth=2)
axes[1].set_title("Step 3: CDF\n(cumulative sum of PDF)", fontsize=10)
axes[1].set_xlabel("Intensity", fontsize=9)
axes[1].set_ylabel("Cumulative probability", fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(range(256), lut, color="#A32D2D", linewidth=2)
axes[2].plot([0, 255], [0, 255], "--", color="#888780", linewidth=1, label="Identity")
axes[2].set_title("Step 4: Intensity Mapping T(r)\n= CDF × 255", fontsize=10)
axes[2].set_xlabel("Input intensity  r", fontsize=9)
axes[2].set_ylabel("Output intensity  T(r)", fontsize=9)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Q3_Histogram_Equalization/outputs/q3_algorithm_steps.png", dpi=150, bbox_inches="tight")
print("Saved -> q3_algorithm_steps.png")
plt.show()


