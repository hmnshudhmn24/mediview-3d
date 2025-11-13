---
language:
  - en
license: apache-2.0
tags:
  - image-to-3d
  - medical-imaging
  - reconstruction
  - segmentation
  - explainability
pipeline_tag: image-to-3d
library_name: python
---

# MEDIVIEW-3D (Advanced)

**MEDIVIEW-3D** converts 2D medical image slices into a 3D reconstruction, localizes anomalous regions, and generates **textual explanations** describing the detected regions (size, approximate location, and suggested next steps).

**Important:** This is a research/demo tool and **not** a medical device. Do not use for clinical decisions.

## Quickstart (demo)

1. Install:
```bash
pip install -r requirements.txt
```

2. Generate synthetic phantom slices:
```bash
python examples/generate_synthetic_phantom.py
```

3. Run inference with thresholding and get a mesh + explanation:
```bash
python infer_anomaly.py --source examples/synthetic_phantom --method threshold --out demo_mesh_threshold.ply --explain_out explanation.txt
```

4. (Optional) Train small UNet and run model-based inference:
```bash
python train_unet.py --data examples/synthetic_phantom --epochs 3 --out models/unet_demo.pt
python infer_anomaly.py --source examples/synthetic_phantom --method model --model_path models/unet_demo.pt --out demo_mesh_model.ply --explain_out explanation_model.txt
```

5. Run Streamlit demo:
```bash
streamlit run app.py
```

## What you get
- 3D mesh `.ply` with anomaly regions colored red
- `explanation.txt` with human-friendly descriptions of detected regions
- Example synthetic phantom (no patient data)
- Small UNet implementation for demo training

## Safety & Limitations
- Demo-only; not clinically validated.
- Do not upload identifiable patient data to public repos.
- For real medical use, integrate robust preprocessing and obtain regulatory approvals.
