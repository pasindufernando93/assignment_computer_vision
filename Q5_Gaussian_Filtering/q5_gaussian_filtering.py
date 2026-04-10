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



