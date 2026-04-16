import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os

# (a) Nearest-Neighbour Interpolation
def zoom_nearest(image, factor):

    assert 0 < factor, "Factor must be positive"
    src_h, src_w = image.shape[:2]
    dst_h = int(round(src_h * factor))
    dst_w = int(round(src_w * factor))

    # Build output pixel coordinate grids
    i_out = np.arange(dst_h)          # rows  0 … dst_h-1
    j_out = np.arange(dst_w)          # cols  0 … dst_w-1

    # Map output → input coordinates (inverse mapping)
    i_in = np.clip(np.round(i_out / factor).astype(int), 0, src_h - 1)
    j_in = np.clip(np.round(j_out / factor).astype(int), 0, src_w - 1)

    # Index: first select rows, then columns (broadcasting via np.ix_)
    if image.ndim == 2:
        zoomed = image[np.ix_(i_in, j_in)]
    else:
        zoomed = image[np.ix_(i_in, j_in)]   # works for (H,W,C) too

    return zoomed.astype(np.uint8)


