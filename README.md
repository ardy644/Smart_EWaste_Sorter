# ♻️ Smart E-Waste & Recyclable Vision Sorter

A complete **Data Science & Deep Learning CNN Lab Project** that classifies 21 categories of e-waste and recyclable materials using Convolutional Neural Networks, with an interactive Streamlit dashboard featuring real-time inference, Grad-CAM explainability, and comprehensive model evaluation.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Overview

This project tackles the growing global e-waste crisis by building an **AI-powered vision sorting system** that can:

- **Classify** images into 21 waste categories (e-waste, dry recyclables, biodegradable)
- **Explain** predictions using Grad-CAM activation heatmaps
- **Compare** a custom 4-layer CNN baseline against EfficientNetV2-S transfer learning
- **Generate** automated sorting manifests for batch processing

### 📊 Key Results

| Model | Val Accuracy | Macro F1 | Parameters |
|-------|:----------:|:-------:|:----------:|
| Baseline 4-Layer CNN | 50.4% | 0.442 | 0.39M |
| **EfficientNetV2-S** | **97.2%** | **0.973** | 20.2M |

---

## 🗂️ Project Structure

```
Smart_EWaste_Sorter/
├── app.py                          # Streamlit dashboard (4 tabs)
├── model_utils.py                  # Model architectures, inference, Grad-CAM
├── requirements.txt                # Python dependencies
├── .gitignore
└── ewaste_deploy_bundle/
    ├── efficientnet_ewaste.pth     # EfficientNetV2-S weights (21-class)
    ├── baseline_cnn.pth            # Custom 4-layer CNN weights
    ├── eda_and_metrics.json        # Training metrics, class metadata, reports
    └── sample_images/              # 42 sample test images (2 per class)
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Smart_EWaste_Sorter.git
cd Smart_EWaste_Sorter
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# CPU-only PyTorch (lightweight ~126 MB)
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Other dependencies
pip install --no-cache-dir streamlit opencv-python-headless pillow numpy pandas plotly scikit-learn
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🖥️ Dashboard Tabs

### 🔬 Tab 1: Live CNN Sorter & Grad-CAM XAI
- Upload any image or select from 42 sample images
- Toggle between **EfficientNetV2-S** and **Baseline CNN** (or run both side-by-side)
- Real-time inference with confidence scores and latency
- **Grad-CAM heatmap overlay** showing what the CNN "sees"
- Color-coded disposal card with hazard warnings and handling instructions

### 📊 Tab 2: Exploratory Data Analysis
- Raw vs. Balanced class distribution (18,515 → 9,900 images)
- Super-category breakdown: Hazardous E-Waste vs. Dry Recyclable vs. Biodegradable
- Filterable sample image gallery with metadata

### 📈 Tab 3: Model Evaluation & Comparison
- KPI summary cards (Accuracy, F1, Parameter Count)
- Interactive learning curves (loss & accuracy across epochs)
- 21×21 confusion matrix heatmaps
- Per-class Precision, Recall, F1-Score table with visual bar chart

### 📦 Tab 4: Batch Inspection & Sorting Manifest
- Upload multiple images for batch classification
- One-click demo on 10 random sample images
- Downloadable CSV sorting manifest (Filename, Class, Confidence, Hazard, Bin)

---

## 🏗️ Model Architectures

### Baseline CNN (Custom 4-Layer)
```
Conv2d(3→32) → BN → ReLU → MaxPool
Conv2d(32→64) → BN → ReLU → MaxPool
Conv2d(64→128) → BN → ReLU → MaxPool
Conv2d(128→256) → BN → ReLU → MaxPool
→ AdaptiveAvgPool2d → Flatten → Linear(256→21)
```

### EfficientNetV2-S (Transfer Learning)
- Pretrained on ImageNet, fine-tuned on balanced e-waste dataset
- Classifier head replaced: `Linear(1280→21)`

---

## 📦 21 Waste Classes

| Hazardous E-Waste 🔴 | Dry Recyclable 🔵 | Biodegradable 🟢 |
|---|---|---|
| Battery | Brown-Glass | Biological |
| Keyboard | Cardboard | |
| Microwave | Clothes | |
| Mobile | Green-Glass | |
| Mouse | Metal | |
| PCB | Paper | |
| Player | Plastic | |
| Printer | Shoes | |
| Television | Trash | |
| | Washing Machine | |
| | White-Glass | |

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, TorchVision
- **Explainability**: Grad-CAM (custom implementation)
- **Dashboard**: Streamlit
- **Visualization**: Plotly
- **Image Processing**: OpenCV, Pillow
- **Data Science**: Pandas, NumPy, Scikit-learn

---

## 📄 License

This project is licensed under the MIT License.
