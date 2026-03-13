# RadAI – AI-Assisted Radiology Image Analysis

A modular Python prototype for detecting abnormal patterns in chest X-ray images using deep learning and visual explainability.

## Pipeline

```
Patient input → Upload CXR → Preprocessing → Autoencoder Anomaly Detection
→ DenseNet121 CNN Analysis → Artifact/Implant Detection → Grad-CAM Heatmap
→ Streamlit Radiologist Interface
```

## Project Structure

```
RadAI/
├── app.py                # Streamlit entry point
├── preprocess.py         # Image loading & normalization (DICOM/PNG/JPG)
├── autoencoder.py        # Conv Autoencoder anomaly detector
├── cnn_analyzer.py       # DenseNet121 anatomical feature extractor
├── artifact_detector.py  # External object / implant filter (OpenCV)
├── heatmap.py            # Grad-CAM heatmap generator & overlay utilities
├── requirements.txt
├── models/               # Place pretrained .pth weight files here
└── data/                 # Place input chest X-ray images here
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## Notes

- **Autoencoder**: Initialized with random weights for the prototype.  
  Fine-tune on normal chest X-rays (e.g., NIH ChestX-ray14) for production.
- **DenseNet121**: Uses ImageNet-pretrained weights from torchvision.  
  Drop CheXNet `.pth` weights into `models/` and set `DENSENET_WEIGHTS` in `app.py` for pathology-specific inference.
- **DICOM support**: Upload `.dcm` files directly — `preprocess.py` handles pixel extraction automatically.

## Supported Pathology Labels (DenseNet121)

Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema,
Fibrosis, Pleural Thickening, Hernia
