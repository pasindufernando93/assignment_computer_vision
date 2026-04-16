
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# Load image 
img_bgr = cv2.imread("assets/a1images/highlights_and_shadows.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print(f"Image shape : {img_rgb.shape}")
print(f"Dtype       : {img_rgb.dtype}")

# Convert to L*a*b* 
img_float = img_rgb.astype(np.float32) / 255.0
img_bgr_float = cv2.cvtColor(img_float, cv2.COLOR_RGB2BGR)
img_lab = cv2.cvtColor(img_bgr_float, cv2.COLOR_BGR2Lab)

L, a, b = cv2.split(img_lab)   # L in [0, 100], a/b in [-128, +127]

print(f"\nL channel range (before) : [{L.min():.2f}, {L.max():.2f}]")

# Choose gamma
GAMMA = 0.5

# Normalize L to [0,1], apply gamma, scale back to [0,100]
L_norm     = L / 100.0
L_corrected = np.power(L_norm, GAMMA) * 100.0

print(f"L channel range (after)  : [{L_corrected.min():.2f}, {L_corrected.max():.2f}]")
print(f"Chosen gamma             : {GAMMA}")


# Merge corrected L back with original a*, b* 
img_lab_corrected = cv2.merge([L_corrected.astype(np.float32), a, b])
img_bgr_corrected = cv2.cvtColor(img_lab_corrected, cv2.COLOR_Lab2BGR)
img_rgb_corrected = cv2.cvtColor(img_bgr_corrected, cv2.COLOR_BGR2RGB)

# Clip to valid float range [0, 1] and convert to uint8 for display
img_rgb_corrected = np.clip(img_rgb_corrected, 0, 1)
img_rgb_uint8     = (img_rgb_corrected * 255).astype(np.uint8)


# Main figure: images + histograms 
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Q2 — Gamma Correction on L* Channel in L*a*b* Color Space",
             fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

# Row 0: RGB images 
ax_orig = fig.add_subplot(gs[0, 0])
ax_orig.imshow(img_rgb)
ax_orig.set_title("Original image", fontsize=10)
ax_orig.axis("off")

ax_corr = fig.add_subplot(gs[0, 1])
ax_corr.imshow(img_rgb_uint8)
ax_corr.set_title(f"After gamma correction  γ = {GAMMA}\n(L* channel only)", fontsize=10)
ax_corr.axis("off")

# Row 1: L* channel images
ax_L_orig = fig.add_subplot(gs[1, 0])
ax_L_orig.imshow(L, cmap="gray", vmin=0, vmax=100)
ax_L_orig.set_title("L* channel — original", fontsize=10)
ax_L_orig.axis("off")

ax_L_corr = fig.add_subplot(gs[1, 1])
ax_L_corr.imshow(L_corrected, cmap="gray", vmin=0, vmax=100)
ax_L_corr.set_title(f"L* channel — after γ = {GAMMA}", fontsize=10)
ax_L_corr.axis("off")

# Row 2: Histograms of L* channel
ax_h_orig = fig.add_subplot(gs[2, 0])
ax_h_orig.hist(L.ravel(), bins=100, range=(0, 100),
               color="#185FA5", alpha=0.85)
ax_h_orig.set_title("Histogram — L* original", fontsize=10)
ax_h_orig.set_xlabel("L* value  (0=black, 100=white)", fontsize=8)
ax_h_orig.set_ylabel("Pixel count", fontsize=8)
ax_h_orig.tick_params(labelsize=7)

ax_h_corr = fig.add_subplot(gs[2, 1])
ax_h_corr.hist(L_corrected.ravel(), bins=100, range=(0, 100),
               color="#0F6E56", alpha=0.85)
ax_h_corr.set_title(f"Histogram — L* after γ = {GAMMA}", fontsize=10)
ax_h_corr.set_xlabel("L* value  (0=black, 100=white)", fontsize=8)
ax_h_corr.set_ylabel("Pixel count", fontsize=8)
ax_h_corr.tick_params(labelsize=7)

plt.savefig("Q2_Lab_Gamma/outputs/q2_output.png", dpi=150, bbox_inches="tight")
print("Saved -> q2_output.png")
plt.show()


# Overlay histogram comparison 
fig2, ax = plt.subplots(figsize=(8, 4))
ax.hist(L.ravel(), bins=100, range=(0, 100),
        color="#185FA5", alpha=0.6, label="L* original")
ax.hist(L_corrected.ravel(), bins=100, range=(0, 100),
        color="#0F6E56", alpha=0.6, label=f"L* after γ = {GAMMA}")

# Mark mean values
ax.axvline(L.mean(), color="#185FA5", linestyle="--", linewidth=1.5,
           label=f"Mean original = {L.mean():.1f}")
ax.axvline(L_corrected.mean(), color="#0F6E56", linestyle="--", linewidth=1.5,
           label=f"Mean corrected = {L_corrected.mean():.1f}")

ax.set_xlabel("L* value  (0 = black, 100 = white)", fontsize=11)
ax.set_ylabel("Pixel count", fontsize=11)
ax.set_title("L* Histogram Overlay — Before vs After Gamma Correction", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Q2_Lab_Gamma/outputs/q2_histogram_overlay.png", dpi=150, bbox_inches="tight")
print("Saved -> q2_histogram_overlay.png")
plt.show()



