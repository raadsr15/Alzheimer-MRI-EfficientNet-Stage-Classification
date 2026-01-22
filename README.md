# 🧠 Alzheimer MRI EfficientNet Stage Classification

![banner-placeholder](https://github.com/user-attachments/assets/your-banner-image)

---

## 🧬 Introduction

This repository presents a deep learning–based framework for **Alzheimer’s disease stage classification** from **structural MRI brain images** using **EfficientNet** with transfer learning. The project demonstrates an end-to-end PyTorch pipeline designed for reproducible research and practical experimentation in medical image analysis.

Alzheimer’s disease (AD) is a progressive neurodegenerative disorder and one of the leading causes of dementia worldwide. Early and accurate identification of disease stages is crucial for timely clinical intervention and disease monitoring. Structural MRI plays a key role in capturing brain atrophy patterns associated with cognitive decline, but manual interpretation remains time-consuming and subject to inter-observer variability.

To address this challenge, this project leverages **EfficientNet**, a convolutional neural network architecture known for its compound scaling of depth, width, and resolution. By using ImageNet-pretrained weights and fine-tuning the classification head, the model learns discriminative neuroanatomical patterns corresponding to different Alzheimer’s stages.

The classification task involves four clinically meaningful categories:

- **Non Demented**
- **Very Mild Demented**
- **Mild Demented**
- **Moderate Demented**

The codebase is modular, GPU-accelerated, and notebook-friendly, making it suitable for academic projects, research prototyping, and future extensions such as explainable AI and transformer-based architectures.

---

## 📚 Dataset

### Overview

This project uses a **preprocessed Alzheimer MRI dataset** prepared for research and educational purposes, with a focus on machine learning and deep learning–based analysis of Alzheimer’s disease.

### Data Source

The original raw neuroimaging data were obtained from the:

**Alzheimer’s Disease Neuroimaging Initiative (ADNI)**

ADNI is a large, multi-center, longitudinal study designed to develop clinical, imaging, genetic, and biochemical biomarkers for the early detection and tracking of Alzheimer’s disease.

🔗 Official website:  
https://adni.loni.usc.edu/

---

### ⚠️ Important Note on Data Access

This dataset **does not provide direct access to original ADNI raw data**.

Access to raw ADNI data requires:
- Formal application
- Acceptance of ADNI Data Use Agreement
- Approval through the official ADNI portal

This repository uses **derived and preprocessed data only**.

---

### 🧪 Preprocessing and Modifications

All data included in this dataset are **preprocessed versions** of the original ADNI scans. Preprocessing steps include:

- Data cleaning  
- Image normalization  
- Resizing for deep learning compatibility  
- Formatting into folder-based class labels  
- Removal of personal identifiers  

No protected health information (PHI) is included.

---

### 📊 Dataset Statistics

**Alzheimer MRI Preprocessed Dataset (128 × 128)**

- Total images: **6,400**
- Image resolution: **128 × 128 pixels**
- Number of classes: **4**

| Class | Number of Images |
|------|------------------|
| Non Demented | 3,200 |
| Very Mild Demented | 2,240 |
| Mild Demented | 896 |
| Moderate Demented | 64 |

> ⚠️ The dataset is highly imbalanced, particularly for the Moderate Demented class.  
> This project addresses imbalance using **stratified sampling and class-weighted loss functions**.

---

### 📌 Attribution

Data used in the preparation of this dataset were obtained from:

**Alzheimer’s Disease Neuroimaging Initiative (ADNI)**

If you use this dataset, please cite the original data source accordingly.

---

## 📁 Dataset Folder Structure
