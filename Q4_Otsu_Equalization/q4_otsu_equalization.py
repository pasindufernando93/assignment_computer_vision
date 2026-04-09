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



