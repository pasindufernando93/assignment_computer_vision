import numpy as np
import cv2

# ── Load image ─────────────────────────────────────────────────────────────────
img_bgr = cv2.imread("Assignment/runway.png", cv2.IMREAD_GRAYSCALE)
img = img_bgr   # already grayscale uint8, range [0, 255]

print(f"Image shape : {img.shape}")
print(f"Pixel range : [{img.min()}, {img.max()}]")
print(f"Total pixels: {img.size}")


