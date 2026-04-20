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


# (d) Apply DoG kernels → image gradients
Gx_dog = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=dog_x_5.astype(np.float32))
Gy_dog = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=dog_y_5.astype(np.float32))

# Gradient magnitude
mag_dog = np.sqrt(Gx_dog**2 + Gy_dog**2)

# Gradient direction (in degrees)
angle_dog = np.degrees(np.arctan2(Gy_dog, Gx_dog))

print(f"\nDoG gradient magnitude range : [{mag_dog.min():.2f}, {mag_dog.max():.2f}]")

# (e) cv2.Sobel() comparison
Gx_sobel = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
Gy_sobel = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
mag_sobel = np.sqrt(Gx_sobel**2 + Gy_sobel**2)

print(f"Sobel gradient magnitude range: [{mag_sobel.min():.2f}, {mag_sobel.max():.2f}]")

# Normalise both magnitudes to [0,255] for fair visual comparison
def norm255(arr):
    mn, mx = arr.min(), arr.max()
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

mag_dog_vis   = norm255(mag_dog)
mag_sobel_vis = norm255(mag_sobel)
Gx_dog_vis    = norm255(Gx_dog)
Gy_dog_vis    = norm255(Gy_dog)
Gx_sobel_vis  = norm255(Gx_sobel)
Gy_sobel_vis  = norm255(Gy_sobel)

# FIGURE — Gradient results
fig, axes = plt.subplots(3, 3, figsize=(15, 13))
fig.suptitle("Q6 — Derivative of Gaussian vs Sobel Gradients  (σ=2, 5×5 kernel)",
             fontsize=13, fontweight="bold")

panels = [
    # Row 0
    (img_gray,      "Original grayscale",         "gray"),
    (Gx_dog_vis,    "(d) DoG — Gx\n(horizontal edges)", "gray"),
    (Gy_dog_vis,    "(d) DoG — Gy\n(vertical edges)",   "gray"),
    # Row 1
    (mag_dog_vis,   "(d) DoG gradient magnitude\n|∇I| = √(Gx²+Gy²)", "hot"),
    (Gx_sobel_vis,  "(e) Sobel — Gx",             "gray"),
    (Gy_sobel_vis,  "(e) Sobel — Gy",             "gray"),
    # Row 2
    (mag_sobel_vis, "(e) Sobel gradient magnitude", "hot"),
    (norm255(np.abs(mag_dog - mag_sobel * (mag_dog.max()/mag_sobel.max()))),
                    "Scaled difference map\n(DoG vs Sobel magnitude)", "hot"),
    (norm255(angle_dog + 180),
                    "Gradient direction (DoG)\n(colour = angle)", "hsv"),
]

for ax, (im, title, cmap) in zip(axes.ravel(), panels):
    ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("Q6_Derivative_of_Gaussian/outputs/q6_gradients.png", dpi=150, bbox_inches="tight")
print("Saved -> q6_gradients.png")
plt.show()

# FIGURE — Kernel comparison (DoG x vs Sobel x)
sobel_x_5 = cv2.getDerivKernels(dx=1, dy=0, ksize=5)
# getDerivKernels returns two 1D kernels; outer product = 2D kernel
sobel_x_2d = np.outer(sobel_x_5[1], sobel_x_5[0]).astype(np.float64)
sobel_x_2d /= np.abs(sobel_x_2d).sum()   # normalise same way as DoG

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle("Kernel Comparison: DoG (x) vs Sobel (x)  —  5×5", fontsize=12, fontweight="bold")

vmax = max(np.abs(dog_x_5).max(), np.abs(sobel_x_2d).max())

im0 = axes2[0].imshow(dog_x_5, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes2[0].set_title("DoG — x-direction", fontsize=10)
plt.colorbar(im0, ax=axes2[0])
for i in range(5):
    for j in range(5):
        axes2[0].text(j, i, f"{dog_x_5[i,j]:.4f}", ha="center", va="center",
                      fontsize=7, color="black")

im1 = axes2[1].imshow(sobel_x_2d, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes2[1].set_title("Sobel — x-direction (normalised)", fontsize=10)
plt.colorbar(im1, ax=axes2[1])
for i in range(5):
    for j in range(5):
        axes2[1].text(j, i, f"{sobel_x_2d[i,j]:.4f}", ha="center", va="center",
                      fontsize=7, color="black")

# Overlay centre-row profiles
axes2[2].plot(range(5), dog_x_5[2, :],    "o-", color="#185FA5", lw=2, ms=7,
              label="DoG centre row")
axes2[2].plot(range(5), sobel_x_2d[2, :], "s--", color="#A32D2D", lw=2, ms=7,
              label="Sobel centre row")
axes2[2].axhline(0, color="#888780", lw=1, ls=":")
axes2[2].set_xticks(range(5))
axes2[2].set_xticklabels(["-2", "-1", "0", "+1", "+2"])
axes2[2].set_xlabel("x offset", fontsize=10)
axes2[2].set_ylabel("Coefficient", fontsize=10)
axes2[2].set_title("Centre row profile", fontsize=10)
axes2[2].legend(fontsize=9)
axes2[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Q6_Derivative_of_Gaussian/outputs/q6_kernel_comparison.png", dpi=150, bbox_inches="tight")
print("Saved -> q6_kernel_comparison.png")
plt.show()
