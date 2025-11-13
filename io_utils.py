import os, glob
import numpy as np
from PIL import Image
import pydicom
import nibabel as nib

def load_slices_from_folder(folder, glob_pattern="*.png", sort_numerically=True):
    paths = sorted(glob.glob(os.path.join(folder, glob_pattern)))
    if sort_numerically:
        try:
            paths = sorted(paths, key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or 0))
        except Exception:
            pass
    slices = [np.array(Image.open(p).convert("L")) for p in paths]
    vol = np.stack(slices, axis=0)  # z, y, x
    return vol.astype(np.float32)

def load_dicom_folder(folder):
    files = [p for p in glob.glob(os.path.join(folder, "**"), recursive=False) if os.path.isfile(p)]
    ds = []
    for f in files:
        try:
            d = pydicom.dcmread(f)
            ds.append((getattr(d, "InstanceNumber", 0), d))
        except Exception:
            continue
    ds = sorted(ds, key=lambda x: x[0])
    imgs = [d.pixel_array for _, d in ds]
    vol = np.stack(imgs, axis=0).astype(np.float32)
    return vol

def load_nifti(path):
    nii = nib.load(path)
    arr = nii.get_fdata().astype(np.float32)
    if arr.ndim == 3:
        return np.transpose(arr, (2,1,0))
    return arr

def save_mesh_ply(verts, faces, out_path, normals=None, colors=None):
    import numpy as np
    with open(out_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if normals is not None:
            f.write("property float nx\nproperty float ny\nproperty nz\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i, v in enumerate(verts):
            line = f"{v[0]} {v[1]} {v[2]}"
            if colors is not None:
                c = (np.clip(colors[i], 0, 1) * 255).astype(int)
                line += f" {c[0]} {c[1]} {c[2]}"
            if normals is not None:
                n = normals[i]
                line += f" {n[0]} {n[1]} {n[2]}"
            f.write(line + "\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
