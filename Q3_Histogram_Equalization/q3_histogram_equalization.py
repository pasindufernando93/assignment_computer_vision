import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# ── Load image ─────────────────────────────────────────────────────────────────
img_bgr = cv2.imread("Q3_Histogram_Equalization/assets/runway.png", cv2.IMREAD_GRAYSCALE)
img = img_bgr 

print(f"Image shape : {img.shape}")
print(f"Pixel range : [{img.min()}, {img.max()}]")
print(f"Total pixels: {img.size}")

# CUSTOM HISTOGRAM EQUALIZATION 
def equalize_histogram(image):

    # Step 1: Histogram 
    hist = np.zeros(256, dtype=np.int64)
    for pixel_value in image.ravel():         
        hist[pixel_value] += 1

    # Step 2: PDF (Probability Density Function) 
    total_pixels = image.size               
    pdf = hist / total_pixels                  

    # Step 3: CDF (Cumulative Distribution Function) 
    cdf = np.cumsum(pdf)                    

    # Step 4: Scale CDF to [0, 255] — this IS the intensity mapping 
    cdf_mapped = np.round(cdf * 255).astype(np.uint8)   # shape (256,)

    # Step 5: Apply the mapping as a lookup table 
    equalized = cdf_mapped[image]             

    return equalized, hist, pdf, cdf, cdf_mapped


# Run custom equalization
eq_custom, hist_orig, pdf_orig, cdf_orig, lut = equalize_histogram(img)

# OpenCV reference for validation 
eq_opencv = cv2.equalizeHist(img)

# Compute difference
diff = np.abs(eq_custom.astype(np.int32) - eq_opencv.astype(np.int32))
print("\nValidation vs OpenCV cv2.equalizeHist():")
print(f"  Max pixel difference : {diff.max()}")
print(f"  Mean difference      : {diff.mean():.4f}")
print(f"  Identical pixels     : {(diff == 0).sum()} / {diff.size}  "
      f"({100*(diff==0).mean():.1f}%)")




