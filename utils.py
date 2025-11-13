import numpy as np

def normalize_volume(vol):
    vmin = np.nanmin(vol)
    vmax = np.nanmax(vol)
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return (vol - vmin) / (vmax - vmin)

def ensure_3d(volume):
    if volume.ndim == 2:
        return volume[np.newaxis, ...]
    return volume
