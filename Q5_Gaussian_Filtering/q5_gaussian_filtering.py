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


