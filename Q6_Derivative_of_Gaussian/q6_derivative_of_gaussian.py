import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import cv2

# (a) Mathematical Derivation 
def gaussian_kernel(size, sigma):
    """Normalised 2D Gaussian kernel (reused from Q5)."""
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

