import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# Load image
img_bgr  = cv2.imread("assets/a1images/spider.png")
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

print(f"Image : spider.png  {img_gray.shape}")


# Method 1 — Unsharp Masking
def unsharp_mask(image, sigma=2, alpha=1.5):

    blurred = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma)
    mask    = image - blurred                       # high-frequency detail
    sharp   = image + alpha * mask                  # add it back amplified
    return np.clip(sharp, 0, 255)


usm_a05 = unsharp_mask(img_gray, sigma=2, alpha=0.5)   # gentle
usm_a15 = unsharp_mask(img_gray, sigma=2, alpha=1.5)   # moderate
usm_a30 = unsharp_mask(img_gray, sigma=2, alpha=3.0)   # aggressive


# Method 2 — Laplacian Sharpening
laplacian_kernel = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
], dtype=np.float32)

lap = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=laplacian_kernel)

def laplacian_sharpen(image, lap_response, lam=0.5):

    sharp = image - lam * lap_response
    return np.clip(sharp, 0, 255)

lap_sharp_05 = laplacian_sharpen(img_gray, lap, lam=0.5)
lap_sharp_10 = laplacian_sharpen(img_gray, lap, lam=1.0)


# Helper: convert float image to uint8 for display
def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# FIGURE 1 — Full image comparison
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Q9 — Image Sharpening: Unsharp Masking & Laplacian",
             fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

panels_row0 = [
    (img_gray,   "Original"),
    (usm_a05,    "Unsharp Mask  α=0.5\n(gentle)"),
    (usm_a15,    "Unsharp Mask  α=1.5\n(moderate)"),
    (usm_a30,    "Unsharp Mask  α=3.0\n(aggressive)"),
]
panels_row1 = [
    (lap,        "Laplacian response\n(edge map)"),
    (lap_sharp_05, "Laplacian sharp  λ=0.5"),
    (lap_sharp_10, "Laplacian sharp  λ=1.0"),
    (img_gray,   "Original (repeat for ref)"),
]

for col, (im, title) in enumerate(panels_row0):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(to_uint8(im), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

for col, (im, title) in enumerate(panels_row1):
    ax = fig.add_subplot(gs[1, col])
    if "Laplacian response" in title:
        # Centre around zero for display
        lap_disp = lap - lap.min()
        lap_disp = (lap_disp / lap_disp.max() * 255).astype(np.uint8)
        ax.imshow(lap_disp, cmap="gray")
    else:
        ax.imshow(to_uint8(im), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.savefig("Q9_Sharpening/outputs/q9_output.png", dpi=150, bbox_inches="tight")
print("Saved -> q9_output.png")
plt.show()


# FIGURE 2 — Zoomed crop comparison (shows fine detail clearly)
H, W  = img_gray.shape
# Crop around the spider's body/legs — rich fine texture
r1, r2 = int(H * 0.25), int(H * 0.65)
c1, c2 = int(W * 0.20), int(W * 0.65)

fig2, axes2 = plt.subplots(2, 3, figsize=(14, 9))
fig2.suptitle("Q9 — Zoomed Crop: Sharpening Detail Comparison",
              fontsize=12, fontweight="bold")

crop_panels = [
    (img_gray,    "Original"),
    (usm_a15,     "Unsharp Mask  α=1.5\n(edges enhanced, smooth areas clean)"),
    (usm_a30,     "Unsharp Mask  α=3.0\n(halos visible at strong edges)"),
    (lap,         "Laplacian response\n(second-derivative edge detector)"),
    (lap_sharp_05, "Laplacian sharp  λ=0.5\n(subtle enhancement)"),
    (lap_sharp_10, "Laplacian sharp  λ=1.0\n(strong, may over-sharpen)"),
]

for ax, (im, title) in zip(axes2.ravel(), crop_panels):
    if "Laplacian response" in title:
        lap_disp = lap[r1:r2, c1:c2]
        lap_disp = lap_disp - lap_disp.min()
        lap_disp = (lap_disp / (lap_disp.max() + 1e-6) * 255).astype(np.uint8)
        ax.imshow(lap_disp, cmap="gray")
    else:
        ax.imshow(to_uint8(im)[r1:r2, c1:c2], cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("Q9_Sharpening/outputs/q9_zoom_crop.png", dpi=150, bbox_inches="tight")
print("Saved -> q9_zoom_crop.png")
plt.show()


# FIGURE 3 — 1D profile: show edge enhancement numerically
row_idx = H // 2
col_slice = slice(int(W * 0.2), int(W * 0.6))

fig3, axes3 = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig3.suptitle("Q9 — 1D Profile: Edge Enhancement by Sharpening",
              fontsize=12, fontweight="bold")

x = np.arange(col_slice.start, col_slice.stop)

axes3[0].plot(x, img_gray[row_idx, col_slice],   color="#444444", lw=1.5, label="Original")
axes3[0].plot(x, usm_a15[row_idx, col_slice],    color="#185FA5", lw=1.5, label="USM α=1.5",    alpha=0.9)
axes3[0].plot(x, usm_a30[row_idx, col_slice],    color="#A32D2D", lw=1.5, label="USM α=3.0",    alpha=0.9)
axes3[0].set_title("Unsharp Masking — 1D intensity profile", fontsize=10)
axes3[0].set_ylabel("Intensity", fontsize=9)
axes3[0].legend(fontsize=9)
axes3[0].grid(True, alpha=0.3)

axes3[1].plot(x, img_gray[row_idx, col_slice],     color="#444444", lw=1.5, label="Original")
axes3[1].plot(x, lap_sharp_05[row_idx, col_slice], color="#0F6E56", lw=1.5, label="Laplacian λ=0.5", alpha=0.9)
axes3[1].plot(x, lap_sharp_10[row_idx, col_slice], color="#BA7517", lw=1.5, label="Laplacian λ=1.0", alpha=0.9)
axes3[1].set_title("Laplacian Sharpening — 1D intensity profile", fontsize=10)
axes3[1].set_xlabel("Column index", fontsize=9)
axes3[1].set_ylabel("Intensity", fontsize=9)
axes3[1].legend(fontsize=9)
axes3[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Q9_Sharpening/outputs/q9_1d_profile.png", dpi=150, bbox_inches="tight")
print("Saved → q9_1d_profile.png")
plt.show()

