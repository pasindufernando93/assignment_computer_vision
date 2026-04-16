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


