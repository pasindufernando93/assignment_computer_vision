
import numpy as np

import cv2

# Load image 
img_bgr = cv2.imread("Q2_lab_Gamma/assets/a1images/highlights_and_shadows.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # for matplotlib display

print(f"Image shape : {img_rgb.shape}")
print(f"Dtype       : {img_rgb.dtype}")

# Convert to L*a*b* 
# OpenCV L*a*b* encoding:
#   L : [0, 255]  (maps to real L* range [0, 100])
#   a : [0, 255]  (maps to real a* range [-128, +127])
#   b : [0, 255]  (maps to real b* range [-128, +127])
# Work in float32 to avoid quantization during gamma.
img_float = img_rgb.astype(np.float32) / 255.0
img_bgr_float = cv2.cvtColor(img_float, cv2.COLOR_RGB2BGR)
img_lab = cv2.cvtColor(img_bgr_float, cv2.COLOR_BGR2Lab)

L, a, b = cv2.split(img_lab)   # L in [0, 100], a/b in [-128, +127]

print(f"\nL channel range (before) : [{L.min():.2f}, {L.max():.2f}]")


