import argparse, yaml
import numpy as np
from skimage import measure
from utils import normalize_volume
from io_utils import load_slices_from_folder, load_dicom_folder, load_nifti, save_mesh_ply

def build_volume(source, source_type='folder', glob_pattern="*.png"):
    if source_type == 'folder':
        vol = load_slices_from_folder(source, glob_pattern=glob_pattern)
    elif source_type == 'dicom':
        vol = load_dicom_folder(source)
    elif source_type == 'nifti':
        vol = load_nifti(source)
    else:
        raise ValueError("Unsupported source_type")
    return vol

def mesh_from_volume(vol, iso=0.5, spacing=(1.0,1.0,1.0)):
    vol_n = normalize_volume(vol)
    verts, faces, normals, values = measure.marching_cubes(vol_n, level=iso, spacing=spacing)
    return verts, faces, normals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source_type", choices=['folder','dicom','nifti'], default='folder')
    parser.add_argument("--glob", default="*.png")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="mesh.ply")
    parser.add_argument("--iso", type=float, default=None)
    args = parser.parse_args()

    try:
        cfg = yaml.safe_load(open(args.config))
    except Exception:
        cfg = {}
    iso = args.iso or (cfg.get('reconstruct',{}).get('iso_value', 0.5))
    spacing = cfg.get('reconstruct',{}).get('spacing', [1.0,1.0,1.0])
    vol = build_volume(args.source, source_type=args.source_type, glob_pattern=args.glob)
    verts, faces, normals = mesh_from_volume(vol, iso=iso, spacing=tuple(spacing))
    save_mesh_ply(verts, faces, args.out, normals=normals)
    print(f"Saved mesh to {args.out} (verts={len(verts)}, faces={len(faces)})")

if __name__ == '__main__':
    main()
