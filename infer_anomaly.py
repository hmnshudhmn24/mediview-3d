import argparse, yaml, numpy as np
from utils import normalize_volume
from io_utils import load_slices_from_folder, load_nifti, load_dicom_folder, save_mesh_ply
from skimage import measure
import os

def simple_threshold_mask(volume, threshold=0.6):
    vol_n = normalize_volume(volume)
    return (vol_n > threshold).astype(np.uint8)

def mask_to_vertex_colors(verts, faces, mask_vol, spacing=(1.0,1.0,1.0)):
    coords = verts / np.array(spacing)
    coords_idx = np.round(coords).astype(int)
    colors = np.zeros((len(verts), 3), dtype=float)
    zmax, ymax, xmax = mask_vol.shape
    for i, (z,y,x) in enumerate(coords_idx):
        if 0 <= z < zmax and 0 <= y < ymax and 0 <= x < xmax:
            if mask_vol[z,y,x]:
                colors[i] = np.array([1.0, 0.0, 0.0])
            else:
                colors[i] = np.array([0.7, 0.7, 0.7])
        else:
            colors[i] = np.array([0.7, 0.7, 0.7])
    return colors

def mesh_and_mask_from_volume(vol, iso=0.5, spacing=(1.0,1.0,1.0), mask=None):
    vol_n = normalize_volume(vol)
    verts, faces, normals, values = measure.marching_cubes(vol_n, level=iso, spacing=spacing)
    colors = None
    if mask is not None:
        colors = mask_to_vertex_colors(verts, faces, mask, spacing=spacing)
    return verts, faces, normals, colors

def analyze_regions(mask, min_voxels=50):
    # find connected components and simple stats
    from scipy import ndimage as ndi
    labeled, n = ndi.label(mask)
    regions = []
    for lab in range(1, n+1):
        coords = np.argwhere(labeled==lab)
        voxels = coords.shape[0]
        z_mean, y_mean, x_mean = coords.mean(axis=0).tolist()
        regions.append({'label': lab, 'voxels': int(voxels), 'center': [float(z_mean), float(y_mean), float(x_mean)]})
    # filter small
    regions = [r for r in regions if r['voxels'] >= min_voxels]
    return regions

def generate_text_explanation(regions, vol_shape, spacing=(1.0,1.0,1.0)):
    if not regions:
        return "No anomalous regions detected above threshold."

    texts = []

    zdim, ydim, xdim = vol_shape
    for r in regions:
        zc, yc, xc = r['center']
        # approximate location as top/middle/bottom and left/center/right
        zpos = 'top' if zc < zdim*0.33 else ('bottom' if zc > zdim*0.66 else 'middle')
        ypos = 'left' if xc < xdim*0.33 else ('right' if xc > xdim*0.66 else 'center')
        texts.append(f"Region {r['label']}: approx {r['voxels']} voxels, located near the {zpos} (z~{zc:.1f}), {ypos} (x~{xc:.1f}). Suggest clinical review and consider high-resolution imaging or segmentation.")
    return "\n".join(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source_type", choices=['folder','dicom','nifti'], default='folder')
    parser.add_argument("--glob", default="*.png")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", choices=['threshold','model'], default='threshold')
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--out", default="mesh_colored.ply")
    parser.add_argument("--explain_out", default="explanation.txt")
    args = parser.parse_args()

    try:
        cfg = yaml.safe_load(open(args.config))
    except Exception:
        cfg = {}
    cfg_thresh = cfg.get('anomaly', {}).get('threshold', 0.6)
    threshold = args.threshold or cfg_thresh
    spacing = tuple(cfg.get('reconstruct', {}).get('spacing', [1.0,1.0,1.0]))
    iso = cfg.get('reconstruct',{}).get('iso_value', 0.5)
    min_vox = cfg.get('text_explainer',{}).get('min_region_voxels', 50)

    # load volume
    if args.source_type == 'folder':
        vol = load_slices_from_folder(args.source, glob_pattern=args.glob)
    elif args.source_type == 'dicom':
        vol = load_dicom_folder(args.source)
    else:
        vol = load_nifti(args.source)

    if args.method == 'threshold':
        mask = simple_threshold_mask(vol, threshold=threshold)
    else:
        # model-based per-slice segmentation (if model provided)
        try:
            import torch
            from models.unet import UNet
            model = UNet(in_channels=1, out_channels=1)
            model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
            model.eval()
            vol_n = normalize_volume(vol)
            mask = np.zeros_like(vol_n, dtype=np.uint8)
            for i in range(vol_n.shape[0]):
                s = vol_n[i]
                x = (s - s.min())/(s.max()-s.min()+1e-8)
                import torch
                inp = torch.tensor(x[np.newaxis, np.newaxis, ...], dtype=torch.float32)
                with torch.no_grad():
                    out = model(inp).numpy()[0,0]
                mask[i] = (out > 0.5).astype(np.uint8)
        except Exception as e:
            print("Model-based method failed:", e)
            mask = simple_threshold_mask(vol, threshold=threshold)

    regions = analyze_regions(mask, min_voxels=min_vox)
    explanation = generate_text_explanation(regions, vol.shape, spacing=spacing)
    verts, faces, normals, colors = mesh_and_mask_from_volume(vol, iso=iso, spacing=spacing, mask=mask)
    save_mesh_ply(verts, faces, args.out, normals=normals, colors=colors)
    with open(args.explain_out, 'w') as f:
        f.write(explanation)
    print("Saved colored mesh to", args.out)
    print("Saved textual explanation to", args.explain_out)

if __name__ == '__main__':
    main()
