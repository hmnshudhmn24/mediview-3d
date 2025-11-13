# 🧠 MEDIVIEW-3D  
### *2D Medical Slices → 3D Reconstruction + Anomaly Localization + Text Explanations*

**MEDIVIEW-3D** transforms stacks of **2D medical image slices** into an interactive **3D reconstruction**, highlights **anomalous regions**, and generates **human-readable explanations** describing the findings.

🚨 **Disclaimer:**  
This project is a *research/demo tool only* — **not** a certified medical device.  
Do **NOT** use it for clinical diagnoses or treatment decisions.

## ✨ Features
- 🧱 **3D Reconstruction** from 2D slices  
- 🔴 **Anomaly Detection & Highlighting** (colored mesh output)  
- 📝 **Auto-Generated Explanations** (size, rough location, next-step suggestions)  
- 🤖 **Two Inference Modes**  
  - Threshold-based  
  - Model-based (UNet)  
- 🧪 Includes a **synthetic phantom dataset** (safe & non-patient)  
- 🖥️ **Streamlit UI** for interactive demo  

## 🚀 Quickstart (Demo)

### 1️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 2️⃣ Generate synthetic phantom slices  
```bash
python examples/generate_synthetic_phantom.py
```

### 3️⃣ Run inference (threshold method)  
Outputs a **3D mesh** + **text explanation**.
```bash
python infer_anomaly.py   --source examples/synthetic_phantom   --method threshold   --out demo_mesh_threshold.ply   --explain_out explanation.txt
```

### 4️⃣ (Optional) Train UNet + Run model-based inference  
```bash
python train_unet.py   --data examples/synthetic_phantom   --epochs 3   --out models/unet_demo.pt
```

```bash
python infer_anomaly.py   --source examples/synthetic_phantom   --method model   --model_path models/unet_demo.pt   --out demo_mesh_model.ply   --explain_out explanation_model.txt
```

### 5️⃣ Launch the Streamlit demo  
```bash
streamlit run app.py
```

## 📦 Output Files
| File | Description |
|------|-------------|
| `*.ply` | 3D reconstructed mesh with anomalous regions highlighted in **red** |
| `explanation.txt` | Human-friendly description of detected anomalies |
| Synthetic Phantom | Example dataset for fully offline experimentation |

## 🧩 Project Structure
```
MEDIVIEW-3D/
│── app.py                     
│── infer_anomaly.py           
│── train_unet.py              
│── examples/
│     └── synthetic_phantom/   
│── models/                    
│── utils/                     
└── requirements.txt
```

## 🧠 How It Works (High-Level)
1. **Load 2D slices** → preprocess  
2. **3D reconstruction** via volume stacking  
3. **Anomaly detection**  
   - Thresholding OR  
   - UNet segmentation  
4. **Mesh generation** (PLY)  
5. **Natural language explanation** summarizing findings  
