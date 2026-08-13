# 🌲 Automated Deforestation Detection Using ResNet50 & Sentinel-2 Imagery

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Satellite%20Data-4285F4.svg)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning and remote sensing framework that leverages **transfer learning with a ResNet50** Convolutional Neural Network (CNN) to classify land cover types and detect tropical deforestation across multi-temporal **Sentinel-2 satellite imagery** (2018 vs. 2024).

This repository applies the pipeline to **Rondônia, Brazil**, a major deforestation hotspot in the Amazon basin characterized by distinct "fishbone" agricultural land clearing patterns.

---

## 📌 Project Summary & Key Findings

* **Model Architecture**: ResNet50 pretrained on ImageNet, adapted for 10 land cover classes using PyTorch.
* **Training Dataset**: EuroSAT RGB Dataset (27,000 Sentinel-2 images at 10m spatial resolution).
* **Model Performance**: **~98% overall accuracy** on the held-out test set after two-stage fine-tuning.
* **Target Study Region**: Rondônia, Brazil (Amazon Basin) `[-63.05, -9.90, -62.75, -9.60]`.
* **Detection Results**: Pinpointed **22 distinct clear-cut deforestation events** between 2018 and 2024.
* **Benchmark Validation**: Results correlate directly with **Global Forest Watch (GFW)** reference data, which recorded **25.3 kha (253 km²)** of cumulative tree cover loss in this area over the same timeframe.

---

## 📸 Visualizations & Key Results

### 1. Model Evaluation (EuroSAT ResNet50 Confusion Matrix)
The fine-tuned model achieved exceptional classification performance across all 10 EuroSAT classes, demonstrating near-perfect discrimination for `Forest` (306/312) and `SeaLake` (297/300).


<img width="1000" height="800" alt="confusion_matrix" src="https://github.com/user-attachments/assets/f7b99430-9362-4978-b283-bb052ca13306" />



---

### 2. Spatial Map of Detected Deforestation Events
Grid patches ($64 \times 64$ pixels) transitioning from `Forest` in 2018 to cleared land types (`Pasture`, `AnnualCrop`, `Industrial`, or `Residential`) in 2024 are highlighted with red bounding boxes.


<img width="790" height="812" alt="download" src="https://github.com/user-attachments/assets/df9cb8b4-7bf3-4566-9ff3-3f7cee3e743c" />

---

## 📊 Global Forest Watch (GFW) Reference Data

The detected deforestation events were validated against independent tree cover loss metrics published by Global Forest Watch for the target area:

| Year | GFW Tree Cover Loss (kha) | % Tree Cover Loss |
| :--- | :--- | :--- |
| **2018** | 5.7 kha | 2.00% |
| **2019** | 4.4 kha | 2.00% |
| **2020** | 4.1 kha | 1.00% |
| **2021** | 4.0 kha | 1.00% |
| **2022** | 4.2 kha | 1.00% |
| **2023** | 1.8 kha | 0.63% |
| **2024** | 1.1 kha | 0.38% |
| **Total (2018–2024)** | **25.3 kha (253 km²)** | **~8.01% Cumulative** |

---

## ⚙️ Methodology & Pipeline Overview
+-------------------------------------------------------------------------+
|                           METHODOLOGY PIPELINE                          |
+-------------------------------------------------------------------------+
| 1. EuroSAT Dataset (27,000 images, 10 classes, 64x64 RGB)               |
|    └─> Split: 80% Train | 10% Validation | 10% Test                     |
+-------------------------------------------------------------------------+
│
▼
| 2. Model Training & Fine-Tuning (ResNet50 + PyTorch)                    |
|    ├─> Stage 1: Freeze base, train FC layer (10 epochs, lr=0.001)       |
|    └─> Stage 2: Unfreeze all, fine-tune model (5 epochs, lr=0.0001)     |
+-------------------------------------------------------------------------+
│
▼
| 3. Satellite Data Acquisition (Google Earth Engine)                     |
|    ├─> Location: Rondônia, Brazil [-63.05, -9.90, -62.75, -9.60]        |
|    └─> Sentinel-2 RGB composites (10m resolution): 2018 vs. 2024        |
+-------------------------------------------------------------------------+
│
▼
| 4. Grid Patching & Model Inference                                      |
|    ├─> Tile satellite composites into 64x64 pixel patches               |
|    └─> Predict land cover class for map_2018 and map_2024               |
+-------------------------------------------------------------------------+
│
▼
| 5. Change Detection & Validation                                        |
|    ├─> Flag transitions: 'Forest' (2018) -> 'Crop/Pasture' (2024)       |
|    └─> Benchmark against Global Forest Watch (GFW) reference data       |
+-------------------------------------------------------------------------+


---

## 📂 Repository Structure

.
├── Deforestation_Research_Paper.pdf  # Complete formatted research paper (PDF)
├── Deforestation_Research_Paper.docx # Complete editable research paper (Word)
├── confusion_matrix.png              # EuroSAT model evaluation confusion matrix
├── download.jpg                      # Satellite map highlighting 22 deforestation events
├── notebook.ipynb                    # Google Colab notebook containing all executable code
└── README.md                         # Project documentation


---

## 🚀 Getting Started

### Prerequisites & Dependencies
Ensure you have the following tools and libraries available:
* Python 3.10+
* PyTorch & Torchvision
* Google Earth Engine Account
* Google Colab (with GPU Hardware Accelerator enabled)

### Running the Notebook in Google Colab
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Amazon-Deforestation-Detection-ResNet50.git](https://github.com/YOUR_USERNAME/Amazon-Deforestation-Detection-ResNet50.git)
Upload and open notebook.ipynb in Google Colab.

Enable GPU acceleration (Runtime > Change runtime type > T4 GPU).

Execute cells sequentially to download EuroSAT, train/fine-tune the ResNet50 model, extract satellite patch grids, and visualize deforestation events.

📜 References & Acknowledgments
EuroSAT Dataset: Helber et al., 2019 (IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing).

Satellite Imagery: European Space Agency (ESA) Sentinel-2 via Google Earth Engine.

Reference Forest Loss Data: Global Forest Watch (GFW).

📄 License
This project is open-source and licensed under the MIT License - see the LICENSE file for details.


---

### One Final Detail:
Before you save, double-check that the image filenames in your repository match what's in the text:
* Make sure your confusion matrix image is named **`confusion_matrix.png`**.
* Make sure your satellite map with red boxes is named **`download.jpg`** (or rename `download.jpg` in the text above to `deforestation_map.png` if you renamed it during upload).
