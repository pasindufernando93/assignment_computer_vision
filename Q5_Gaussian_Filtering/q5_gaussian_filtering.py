import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3D projection
import cv2

# Helper — build a Gaussian kernel of any size
def gaussian_kernel(size, sigma):

    assert size % 2 == 1, "Kernel size must be odd"
    half = size // 2
    # Create coordinate grids centred at zero
    # e.g. for size=5: coords = [-2, -1, 0, 1, 2]
    coords = np.arange(-half, half + 1)          # shape (size,)
    x, y   = np.meshgrid(coords, coords)         # both shape (size, size)

    # Evaluate the (unnormalised) Gaussian
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalise so coefficients sum to 1
    kernel /= kernel.sum()
    return kernel


# (a) 5×5 kernel, σ = 2
SIGMA = 2
kernel_5x5 = gaussian_kernel(size=5, sigma=SIGMA)

print("(a) 5×5 Gaussian kernel  (σ = 2):")
print("-" * 55)
# Print with 6 decimal places so the symmetry is clear
with np.printoptions(precision=6, suppress=True):
    print(kernel_5x5)
print(f"\nKernel sum (should be 1.0) : {kernel_5x5.sum():.10f}")
print(f"Centre value               : {kernel_5x5[2,2]:.6f}  (largest)")
print(f"Corner value               : {kernel_5x5[0,0]:.6f}  (smallest)")

# (b) 51×51 kernel — 3D surface plot
kernel_51x51 = gaussian_kernel(size=51, sigma=SIGMA)

half = 25
coords = np.arange(-half, half + 1)
X, Y   = np.meshgrid(coords, coords)

fig_3d = plt.figure(figsize=(10, 7))
ax3d   = fig_3d.add_subplot(111, projection="3d")

surf = ax3d.plot_surface(
    X, Y, kernel_51x51,
    cmap="viridis",
    edgecolor="none",
    alpha=0.92
)

ax3d.set_xlabel("x  (pixels)", fontsize=10, labelpad=8)
ax3d.set_ylabel("y  (pixels)", fontsize=10, labelpad=8)
ax3d.set_zlabel("Kernel coefficient", fontsize=10, labelpad=8)
ax3d.set_title(f"(b) 51×51 Gaussian Kernel  (σ = {SIGMA})", fontsize=12)
fig_3d.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1, label="Coefficient value")

plt.tight_layout()
plt.savefig("q5_3d_surface.png", dpi=150, bbox_inches="tight")
print("\nSaved → q5_3d_surface.png")
plt.show()


# Load image for filtering  (daisy — good texture + edges for comparison)
img_bgr  = cv2.imread("Assignment/a1images/a1images/daisy.jpg")
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

print(f"\nImage : daisy.jpg  {img_gray.shape}")

# (c) Apply our manual kernel using cv2.filter2D
kernel_f32     = kernel_5x5.astype(np.float32)
smoothed_manual = cv2.filter2D(img_gray, ddepth=-1, kernel=kernel_f32)

# (d) OpenCV built-in cv2.GaussianBlur
smoothed_opencv = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=SIGMA, sigmaY=SIGMA)

# Difference map
diff = np.abs(smoothed_manual.astype(np.float32) - smoothed_opencv.astype(np.float32))
print("Comparison — manual vs cv2.GaussianBlur():")
print(f"  Max pixel difference  : {diff.max():.4f}")
print(f"  Mean pixel difference : {diff.mean():.6f}")


# FIGURE — Filtering results
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Q5 — Gaussian Filtering  (σ = 2, 5×5 kernel)", fontsize=13, fontweight="bold")

# Row 0: images
panels = [
    (img_gray,        "Original grayscale"),
    (smoothed_manual, "(c) Manual kernel\nvia cv2.filter2D()"),
    (smoothed_opencv, "(d) cv2.GaussianBlur()\n(built-in)"),
]
for ax, (im, title) in zip(axes[0], panels):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

# Row 1: zoomed crop (top-left petal — rich texture)
H, W = img_gray.shape
r1, r2 = int(H * 0.05), int(H * 0.45)
c1, c2 = int(W * 0.05), int(W * 0.55)

zoom_orig   = img_gray[r1:r2, c1:c2]
zoom_manual = smoothed_manual[r1:r2, c1:c2]
zoom_opencv = smoothed_opencv[r1:r2, c1:c2]

for ax, (im, title) in zip(axes[1], [
    (zoom_orig,   "Zoomed — original"),
    (zoom_manual, "Zoomed — manual kernel"),
    (zoom_opencv, "Zoomed — OpenCV"),
]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("q5_filtering.png", dpi=150, bbox_inches="tight")
print("Saved → q5_filtering.png")
plt.show()


# FIGURE — 5×5 kernel visualised as heatmap + values
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
fig2.suptitle("(a) 5×5 Gaussian Kernel  (σ = 2)", fontsize=12, fontweight="bold")

# Heatmap
im = axes2[0].imshow(kernel_5x5, cmap="Blues", aspect="equal")
axes2[0].set_title("Heatmap", fontsize=10)
axes2[0].set_xticks(range(5))
axes2[0].set_yticks(range(5))
plt.colorbar(im, ax=axes2[0], label="Coefficient")

# Annotate each cell with its value
for i in range(5):
    for j in range(5):
        axes2[0].text(j, i, f"{kernel_5x5[i,j]:.4f}",
                      ha="center", va="center", fontsize=8,
                      color="white" if kernel_5x5[i,j] > 0.06 else "#1a1a1a")

# 2D curve through centre row
axes2[1].plot(range(5), kernel_5x5[2, :], "o-", color="#185FA5",
              linewidth=2, markersize=7, label="Centre row  (y=0)")
axes2[1].plot(range(5), kernel_5x5[0, :], "s--", color="#888780",
              linewidth=1.5, markersize=5, label="Corner row  (y=-2)")
axes2[1].set_xticks(range(5))
axes2[1].set_xticklabels(["-2", "-1", "0", "+1", "+2"])
axes2[1].set_xlabel("x offset", fontsize=10)
axes2[1].set_ylabel("Coefficient", fontsize=10)
axes2[1].set_title("Kernel profile", fontsize=10)
axes2[1].legend(fontsize=9)
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("q5_kernel_5x5.png", dpi=150, bbox_inches="tight")
print("Saved → q5_kernel_5x5.png")
plt.show()

# Difference heatmap 
fig3, ax = plt.subplots(figsize=(6, 5))
im3 = ax.imshow(diff, cmap="hot", vmin=0, vmax=2)
ax.set_title(f"Difference map: manual vs OpenCV\n"
             f"max={diff.max():.3f}  mean={diff.mean():.4f}", fontsize=10)
ax.axis("off")
plt.colorbar(im3, ax=ax, label="Absolute pixel difference")
plt.tight_layout()
plt.savefig("q5_diff.png", dpi=150, bbox_inches="tight")
print("Saved → q5_diff.png")
plt.show()

