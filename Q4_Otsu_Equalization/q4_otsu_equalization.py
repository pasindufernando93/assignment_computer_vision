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



