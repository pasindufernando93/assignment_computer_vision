
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2


# Build filters and compute their 2D frequency responses via FFT
def freq_response_2d(kernel, size=256):
    pad = np.zeros((size, size))
    kh, kw = kernel.shape
    pad[:kh, :kw] = kernel
    F = np.fft.fft2(pad)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)
    return mag

def radial_profile(mag):
    cy, cx = np.array(mag.shape) // 2
    Y, X   = np.ogrid[:mag.shape[0], :mag.shape[1]]
    R      = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    radial = np.bincount(R.ravel(), weights=mag.ravel()) / np.bincount(R.ravel())
    return radial


SIZE = 256

# Box filter 9×9
box_k     = np.ones((9, 9), dtype=np.float32) / 81
box_mag   = freq_response_2d(box_k, SIZE)
box_rad   = radial_profile(box_mag)

# Gaussian filter σ=2, 9×9 
coords    = np.arange(-4, 5)
x, y      = np.meshgrid(coords, coords)
gauss_k   = np.exp(-(x**2 + y**2) / (2 * 2**2))
gauss_k  /= gauss_k.sum()
gauss_mag = freq_response_2d(gauss_k, SIZE)
gauss_rad = radial_profile(gauss_mag)

# Laplacian filter 
lap_k     = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
lap_mag   = freq_response_2d(lap_k, SIZE)
lap_rad   = radial_profile(lap_mag)

# Normalise for comparable display
def norm(r): return r / r.max()

box_rad_n   = norm(box_rad)
gauss_rad_n = norm(gauss_rad)
lap_rad_n   = norm(lap_rad)
freq_axis   = np.arange(len(box_rad_n)) / len(box_rad_n)   # normalised freq

# FIGURE 1 — 2D magnitude spectra of each filter
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Q11 — Frequency Responses of Spatial Filters (2D spectra + radial profiles)",
             fontsize=13, fontweight="bold")

spectra = [
    (box_mag,   box_rad_n,   "Box filter 9×9",             "#185FA5"),
    (gauss_mag, gauss_rad_n, "Gaussian filter σ=2, 9×9",   "#0F6E56"),
    (lap_mag,   lap_rad_n,   "Laplacian filter 3×3",       "#A32D2D"),
]

for col, (mag, rad, title, color) in enumerate(spectra):
    # 2D spectrum (log scale for visibility)
    ax0 = axes[0, col]
    crop = SIZE // 4   # show central quarter (interesting region)
    c    = SIZE // 2
    mag_crop = np.log1p(mag[c-crop:c+crop, c-crop:c+crop])
    ax0.imshow(mag_crop, cmap="inferno", origin="upper")
    ax0.set_title(f"{title}\n2D magnitude spectrum (log scale)", fontsize=9)
    ax0.axis("off")

    # Radial profile
    ax1 = axes[1, col]
    max_freq_show = SIZE // 4
    ax1.plot(freq_axis[:max_freq_show], rad[:max_freq_show],
             color=color, linewidth=2)
    ax1.set_xlabel("Normalised frequency", fontsize=9)
    ax1.set_ylabel("|H(f)|", fontsize=9)
    ax1.set_title(f"Radial profile — {title.split(' ')[0]}", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig("Q11_Frequency_Theory/outputs/q11_spectra.png", dpi=150, bbox_inches="tight")
print("Saved -> q11_spectra.png")
plt.show()

# FIGURE 2 — Ringing comparison: ideal LPF vs Gaussian on a step edge
# Create a synthetic step-edge image
step = np.zeros((128, 256), dtype=np.float32)
step[:, 128:] = 255.0

# Apply ideal LPF (brick-wall in frequency domain)
F_step = np.fft.fft2(step)
F_shift = np.fft.fftshift(F_step)
cy, cx = 64, 128
Y, X   = np.ogrid[:128, :256]
R      = np.sqrt((X - cx)**2 + (Y - cy)**2)
cutoff = 15
ideal_mask  = (R <= cutoff).astype(np.float32)
gauss_mask  = np.exp(-(R**2) / (2 * (cutoff/1.5)**2))

step_ideal = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift * ideal_mask)))
step_gauss = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift * gauss_mask)))

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8))
fig2.suptitle("Q11(c) — Ringing: Ideal Low-Pass Filter vs Gaussian Filter",
              fontsize=12, fontweight="bold")

axes2[0,0].imshow(step_ideal, cmap="gray", vmin=0, vmax=255)
axes2[0,0].set_title("Ideal LPF applied\n(ringing artefacts at step edge)", fontsize=10)
axes2[0,0].axis("off")

axes2[0,1].imshow(step_gauss, cmap="gray", vmin=0, vmax=255)
axes2[0,1].set_title("Gaussian LPF applied\n(smooth, no ringing)", fontsize=10)
axes2[0,1].axis("off")

# 1D profiles through the middle row
row = 64
axes2[1,0].plot(step[row], color="#444444", lw=1.5, label="Original step")
axes2[1,0].plot(step_ideal[row], color="#A32D2D", lw=2, label="After ideal LPF")
axes2[1,0].set_title("1D profile — Ideal LPF\n(Gibbs / ringing oscillation visible)", fontsize=10)
axes2[1,0].legend(fontsize=9)
axes2[1,0].grid(True, alpha=0.3)
axes2[1,0].set_xlabel("Column", fontsize=9)
axes2[1,0].set_ylabel("Intensity", fontsize=9)

axes2[1,1].plot(step[row], color="#444444", lw=1.5, label="Original step")
axes2[1,1].plot(step_gauss[row], color="#0F6E56", lw=2, label="After Gaussian LPF")
axes2[1,1].set_title("1D profile — Gaussian LPF\n(smooth monotonic transition)", fontsize=10)
axes2[1,1].legend(fontsize=9)
axes2[1,1].grid(True, alpha=0.3)
axes2[1,1].set_xlabel("Column", fontsize=9)
axes2[1,1].set_ylabel("Intensity", fontsize=9)

plt.tight_layout()
plt.savefig("Q11_Frequency_Theory/outputs/q11_ringing.png", dpi=150, bbox_inches="tight")
print("Saved -> q11_ringing.png")
plt.show()


# FIGURE 3 — All three filter responses overlaid + noise analysis
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Q11 — Filter Frequency Responses & Noise Reduction Comparison",
              fontsize=12, fontweight="bold")

max_f = SIZE // 4
f     = freq_axis[:max_f]

# Overlay all three
axes3[0].plot(f, box_rad_n[:max_f],   color="#185FA5", lw=2,   label="Box filter (sinc)")
axes3[0].plot(f, gauss_rad_n[:max_f], color="#0F6E56", lw=2,   label="Gaussian filter")
axes3[0].plot(f, lap_rad_n[:max_f],   color="#A32D2D", lw=2,   label="Laplacian filter")
axes3[0].axvspan(0.15, max_f/SIZE, alpha=0.08, color="#888780", label="High-freq noise region")
axes3[0].set_xlabel("Normalised frequency", fontsize=10)
axes3[0].set_ylabel("|H(f)| — normalised", fontsize=10)
axes3[0].set_title("All three filter frequency responses", fontsize=10)
axes3[0].legend(fontsize=9)
axes3[0].grid(True, alpha=0.3)

# Apply to a noisy image to show noise reduction
img_bgr  = cv2.imread("assets/a1images/einstein.png")
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Add Gaussian noise
np.random.seed(42)
noisy = np.clip(img_gray + np.random.normal(0, 30, img_gray.shape), 0, 255)

filtered_box   = cv2.blur(noisy, (9, 9))
filtered_gauss = cv2.GaussianBlur(noisy, (9, 9), sigmaX=2)
filtered_lap   = cv2.filter2D(noisy, -1, lap_k)   # edge-sharpened (not denoised)

def mse(a, b): return np.mean((a.astype(np.float64) - b.astype(np.float64))**2)

mse_noisy = mse(img_gray, noisy)
mse_box   = mse(img_gray, filtered_box)
mse_gauss = mse(img_gray, filtered_gauss)

bars = axes3[1].bar(
    ["Noisy\n(original)", "Box\n9×9", "Gaussian\n9×9, σ=2"],
    [mse_noisy, mse_box, mse_gauss],
    color=["#888780", "#185FA5", "#0F6E56"],
    alpha=0.85, width=0.5
)
for bar, val in zip(bars, [mse_noisy, mse_box, mse_gauss]):
    axes3[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                  f"MSE={val:.0f}", ha="center", fontsize=9)
axes3[1].set_ylabel("MSE vs clean image\n(lower = better)", fontsize=10)
axes3[1].set_title("Q11(d) — Noise reduction quality\n(Laplacian excluded — amplifies noise)", fontsize=10)
axes3[1].grid(True, alpha=0.3, axis="y")
axes3[1].set_ylim(0, mse_noisy * 1.25)

plt.tight_layout()
plt.savefig("Q11_Frequency_Theory/outputs/q11_noise_reduction.png", dpi=150, bbox_inches="tight")
print("Saved -> q11_noise_reduction.png")
plt.show()
