import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import cv2

# (a) Mathematical Derivation 
def gaussian_kernel(size, sigma):
    half   = size // 2
    coords = np.arange(-half, half + 1)
    x, y   = np.meshgrid(coords, coords)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def dog_kernel_x(size, sigma):
    half   = size // 2
    coords = np.arange(-half, half + 1)
    x, y   = np.meshgrid(coords, coords)
    G      = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = -(x / sigma**2) * G
    # Normalise by sum of absolute values (not sum, since kernel sums to ~0)
    kernel /= np.abs(kernel).sum()
    return kernel


def dog_kernel_y(size, sigma):
    half   = size // 2
    coords = np.arange(-half, half + 1)
    x, y   = np.meshgrid(coords, coords)
    G      = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = -(y / sigma**2) * G
    kernel /= np.abs(kernel).sum()
    return kernel

# (b) 5×5 DoG kernels, σ = 2
SIGMA = 2

dog_x_5 = dog_kernel_x(size=5, sigma=SIGMA)
dog_y_5 = dog_kernel_y(size=5, sigma=SIGMA)

print("(b) 5×5 DoG kernel — x-direction  (σ=2):")
print("-" * 60)
with np.printoptions(precision=6, suppress=True):
    print(dog_x_5)

print(f"\nKernel sum (should be ~0): {dog_x_5.sum():.2e}  (antisymmetric)")
print(f"Sum of absolute values   : {np.abs(dog_x_5).sum():.6f}  (normalisation)")

print("\n5×5 DoG kernel — y-direction  (σ=2):")
print("-" * 60)
with np.printoptions(precision=6, suppress=True):
    print(dog_y_5)

# (c) 51×51 DoG kernel — 3D surface plot (x-direction)
dog_x_51 = dog_kernel_x(size=51, sigma=SIGMA)

half   = 25
coords = np.arange(-half, half + 1)
X, Y   = np.meshgrid(coords, coords)

fig_3d = plt.figure(figsize=(11, 7))
ax3d   = fig_3d.add_subplot(111, projection="3d")

surf = ax3d.plot_surface(X, Y, dog_x_51, cmap="RdBu_r",
                          edgecolor="none", alpha=0.92)

ax3d.set_xlabel("x  (pixels)", fontsize=10, labelpad=8)
ax3d.set_ylabel("y  (pixels)", fontsize=10, labelpad=8)
ax3d.set_zlabel("Kernel coefficient", fontsize=10, labelpad=8)
ax3d.set_title(f"(c) 51×51 Derivative-of-Gaussian — x-direction  (σ={SIGMA})\n"
               f"Antisymmetric saddle shape: negative left, positive right",
               fontsize=11)
fig_3d.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1, label="Coefficient")

plt.tight_layout()
plt.savefig("Q6_Derivative_of_Gaussian/outputs/q6_3d_surface.png", dpi=150, bbox_inches="tight")
print("Saved -> q6_3d_surface.png")
plt.show()


# Load image
img_bgr  = cv2.imread("assets/a1images/einstein.png")
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

print(f"\nImage : einstein.png  {img_gray.shape}")


