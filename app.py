import streamlit as st
from PIL import Image
import numpy as np
import tempfile, os, zipfile
from io_utils import load_slices_from_folder
from infer_anomaly import simple_threshold_mask, mesh_and_mask_from_volume
from io_utils import save_mesh_ply

st.set_page_config(page_title="MEDIVIEW-3D", layout="wide")
st.title("MEDIVIEW-3D — 2D → 3D Reconstruction + Explanation")

uploaded = st.file_uploader("Upload a zip of PNG slices (or skip to use example)", type=["zip"])
use_example = st.button("Use example synthetic slices")
thresh = st.slider("Threshold (for simple anomaly)", 0.0, 1.0, 0.6)
out_mesh = st.text_input("Output mesh path", value="mesh_demo.ply")

def unpack_and_read(zipf):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(zipf) as z:
        z.extractall(tmpdir)
    vol = load_slices_from_folder(tmpdir, glob_pattern="*.png")
    return vol, tmpdir

vol = None
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        vol, tmpdir = unpack_and_read(tmp.name)
elif use_example:
    if os.path.isdir("examples/synthetic_phantom"):
        vol = load_slices_from_folder("examples/synthetic_phantom", glob_pattern="*.png")
    else:
        st.error("Example slices not found in examples/synthetic_phantom")

if vol is not None:
    st.write("Volume shape (z,y,x):", vol.shape)
    mid = vol.shape[0]//2
    st.image(Image.fromarray(vol[mid].astype('uint8')), caption=f"Middle slice (index {mid})")
    if st.button("Run reconstruction & anomaly detection"):
        mask = simple_threshold_mask(vol, threshold=thresh)
        verts, faces, normals, colors = mesh_and_mask_from_volume(vol, iso=0.5, spacing=(1.0,1.0,1.0), mask=mask)
        save_mesh_ply(verts, faces, out_mesh, normals=normals, colors=colors)
        st.success(f"Saved colored mesh to {out_mesh}")
        # Generate simple textual explanation
        from infer_anomaly import analyze_regions, generate_text_explanation
        regions = analyze_regions(mask, min_voxels=20)
        expl = generate_text_explanation(regions, vol.shape)
        st.subheader('Detected regions explanation')
        st.text(expl)
else:
    st.info("Upload a slice zip or click 'Use example synthetic slices' to proceed.")
