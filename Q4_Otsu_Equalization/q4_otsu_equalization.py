import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# Load & convert to grayscale 
img_bgr  = cv2.imread("Q4_Otsu_Equalization/assets/a1images/emma.jpg")
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


# (b) Histogram Equalization on Foreground only
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

