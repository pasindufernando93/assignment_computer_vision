import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import sys
sys.stdout.reconfigure(encoding='utf-8') 

# Load image 
img_raw = Image.open("assets/runway.png").convert("L")
# normalize to [0, 1]
img = np.array(img_raw, dtype=np.float64) / 255.0  

print(f"Image shape : {img.shape}")
print(f"Pixel range : [{img.min():.3f}, {img.max():.3f}]")


# Gamma correction
def gamma_correction(image, gamma):

    # Apply power-law (gamma) transformation: s = r^γ
    return np.clip(np.power(image, gamma), 0, 1)


# (a) Gamma correction with r = 0.5
gamma_05 = gamma_correction(img, gamma=0.5)
# (b) Gamma correction with r = 2
gamma_2  = gamma_correction(img, gamma=2.0)


# (c) Contrast Stretching 
def contrast_stretch(image, r1=0.2, r2=0.8):
 
    # Piecewise linear contrast stretching
    if r2 == r1:
        return image.copy()

    output = np.zeros_like(image)

    # Mid region
    mid_mask = (image >= r1) & (image <= r2)
    output[mid_mask] = (image[mid_mask] - r1) / (r2 - r1)

    # High region
    output[image > r2] = 1.0

    return np.clip(output, 0, 1)


contrast_stretched = contrast_stretch(img, r1=0.2, r2=0.8)

# Plotting
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Q1 — Intensity Transformations on Runway Image", fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.3)

images = [img, gamma_05, gamma_2, contrast_stretched]
titles = [
    "Original",
    "Gamma Correction  γ = 0.5\n(brightens dark regions)",
    "Gamma Correction  γ = 2.0\n(darkens bright regions)",
    "Contrast Stretching\nr₁=0.2, r₂=0.8",
]

# Top row: images
for col, (im, title) in enumerate(zip(images, titles)):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

# Bottom row: histograms
hist_colors = ["#444444", "#185FA5", "#A32D2D", "#0F6E56"]

for col, (im, title, color) in enumerate(zip(images, titles, hist_colors)):
    ax = fig.add_subplot(gs[1, col])
    ax.hist(im.ravel(), bins=256, range=(0, 1), color=color, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pixel intensity", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(f"Histogram — {title.split(chr(10))[0]}", fontsize=8)
    ax.tick_params(labelsize=7)

plt.savefig("Q1_Inensity_Transforms/outputs/q1_output.png", dpi=150, bbox_inches="tight")
print("Saved -> q1_output.png")
plt.show()

# Transform curve plot
r = np.linspace(0, 1, 500)

fig2, ax = plt.subplots(figsize=(7, 5))
ax.plot(r, r, linestyle="--", label="Identity (no change)", linewidth=1.5)
ax.plot(r, gamma_correction(r, 0.5), label="Gamma γ = 0.5", linewidth=2)
ax.plot(r, gamma_correction(r, 2.0), label="Gamma γ = 2.0", linewidth=2)
ax.plot(r, contrast_stretch(r, 0.2, 0.8), label="Contrast Stretch r₁=0.2, r₂=0.8", linewidth=2)

ax.set_xlabel("Input intensity  r", fontsize=11)
ax.set_ylabel("Output intensity  s", fontsize=11)
ax.set_title("Intensity Transform Curves", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("Q1_Inensity_Transforms/outputs/q1_transform_curves.png", dpi=150, bbox_inches="tight")
print("Saved -> q1_transform_curves.png")
plt.show()