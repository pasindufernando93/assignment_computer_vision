import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# Load the noisy image (taylor.jpg — has salt & pepper noise)
BASE     = "assets/a1images/a1q8images"
img_bgr  = cv2.imread(f"{BASE}/taylor.jpg")
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


