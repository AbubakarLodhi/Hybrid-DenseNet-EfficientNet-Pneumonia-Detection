# 🫁 Hybrid DenseNet121–EfficientNet-B2 for Pneumonia Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-99.3%25-brightgreen)

## 📌 Overview
This repository contains the official PyTorch implementation of the research paper: **"An Integrated Hybrid DenseNet121–EfficientNet-B2 Framework for Robust and High-Accuracy Pneumonia Detection from Chest X-Ray Images."**

Pneumonia remains a leading cause of childhood mortality globally. This project proposes a **dual-branch Convolutional Neural Network (CNN)** that fuses:
1.  **DenseNet121:** For capturing low-level textural features via feature reuse.
2.  **EfficientNet-B2:** For capturing high-level semantic features via compound scaling.

By concatenating features from both backbones, our model achieves state-of-the-art performance, outperforming individual networks and complex ensembles.

---

## 🚀 Key Features
* **Hybrid Architecture:** Combines the strengths of Dense Connections (DenseNet) and MBConv blocks (EfficientNet).
* **High Accuracy:** Achieved **99.3% Validation Accuracy** and **99.1% Recall**.
* **Robustness:** Tested on the challenging Guangzhou Women and Children’s Medical Center dataset.
* **Lightweight:** ~13.3M parameters (more efficient than VGG16-based models).
* **Data Augmentation:** Includes rotation, scaling, and color jittering for regularization.

---

## 📊 Dataset
The model is trained on the **Chest X-Ray Images (Pneumonia)** dataset.
* **Source:** [Kaggle - Paul Mooney](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* **Classes:** `NORMAL` vs `PNEUMONIA`
* **Total Images:** 5,863
* **Preprocessing:** All images resized to **260x260** and normalized using ImageNet statistics.

---

## 🛠️ Methodology
The system processes an input image through two parallel backbones. The features are extracted after the final convolutional blocks, flattened via Global Average Pooling (GAP), and concatenated into a single embedding vector before passing through a custom classification head.

| Backbone | Feature Vector Size | Role |
| :--- | :--- | :--- |
| **DenseNet121** | 1024 | Propagates signal, captures fine textures. |
| **EfficientNet-B2** | 1408 | Optimize parameter efficiency, captures shapes. |
| **Fused Vector** | **2432** | Combined representation for final classification. |

---

## ⚙️ Installation & Requirements

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Hybrid-DenseNet-EfficientNet-Pneumonia-Detection.git](https://github.com/your-username/Hybrid-DenseNet-EfficientNet-Pneumonia-Detection.git)
   cd Hybrid-DenseNet-EfficientNet-Pneumonia-Detection
