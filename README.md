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



## 🧠 Project Architecture

This repository follows a clean, modular, and reproducible research-oriented architecture.

### 🔍  Model Architecture
- EfficientNet (B0 / B3) pretrained on ImageNet
- Transfer learning–based fine-tuning
- Custom classification head for four Alzheimer stages
- Dropout regularization for improved generalization

---

### 🔄  Training Workflow
- Stratified train–validation–test split
- Weighted CrossEntropy loss for class imbalance handling
- AdamW optimizer
- Mixed precision training (AMP)
- Best model checkpointing based on validation accuracy

---

### 🧾  Data Pipeline
- Dataset loading via `torchvision.datasets.ImageFolder`
- MRI normalization using ImageNet statistics
- Data augmentation:
  - Random rotation
  - Horizontal flipping
  - Brightness and contrast jitter
- GPU-accelerated batching with PyTorch DataLoader

---

### 📊  Evaluation & Visualization
Model evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix visualization
- Classification report heatmaps
- Training loss and accuracy curves

---

## 📈 Results

The EfficientNet-based model was trained for **100 epochs** using GPU acceleration with stratified sampling and class-weighted loss. The training process demonstrated strong convergence and stable generalization performance.

### 🧪 Final Training Summary

| Metric | Value |
|------|------|
| Final Training Loss | **0.0160** |
| Final Validation Loss | **0.0175** |
| Best Validation Accuracy | **99.27%** |
| Total Epochs | **100** |

The close alignment between training and validation loss indicates **minimal overfitting** and strong generalization capability.

---

## 📊 Confusion Matrix

The confusion matrix below illustrates the classification performance across all four Alzheimer’s stages.

![Confusion Matrix](results/confusion_matrix.png)

### Interpretation

- **Non Demented** samples are classified with very high accuracy, with minimal confusion.
- **Very Mild Demented** cases show strong discrimination, with only minor overlap with Non Demented samples.
- **Mild Demented** samples exhibit small misclassification toward adjacent cognitive stages, which is expected due to overlapping structural patterns.
- **Moderate Demented** class achieves perfect classification despite limited sample size.

Overall, the confusion matrix demonstrates **robust inter-class separation**.

---

## 📑 Classification Report (Per Class)

The following heatmap summarizes precision, recall, and F1-score for each class.

![Classification Report](results/classification_report.png)

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| Mild Demented | 0.94 | 0.87 | 0.91 |
| Moderate Demented | 1.00 | 1.00 | 1.00 |
| Non Demented | 0.93 | 0.96 | 0.94 |
| Very Mild Demented | 0.95 | 0.93 | 0.94 |

### Observations

- All classes achieve **F1-scores above 0.91**.
- The **Moderate Demented** class reaches perfect performance, despite severe class imbalance.
- High recall values indicate strong sensitivity in detecting Alzheimer’s stages.

---

## 📉 Training Loss Curve

The loss curve below shows the evolution of training and validation loss across epochs.

![Loss Curve](results/loss_curve.png)

### Interpretation

- Rapid loss reduction occurs during early epochs, showing effective feature learning.
- Training and validation losses remain closely aligned.
- No divergence is observed, indicating **stable optimization and regularization**.

---

## 📈 Validation Accuracy Curve

The validation accuracy progression during training is shown below.

![Validation Accuracy](results/val_accuracy_curve.png)

### Interpretation

- Validation accuracy increases sharply within the first few epochs.
- Performance stabilizes above **99%** for the majority of training.
- Occasional small fluctuations are expected due to stochastic optimization.

This confirms **excellent convergence and strong generalization**.

---

## 📌 Overall Performance Summary

- The model achieves **near-perfect validation accuracy (99.27%)**.
- Class imbalance was effectively handled using weighted loss.
- EfficientNet demonstrates strong capability in extracting discriminative MRI features.
- The training behavior shows **no signs of overfitting**.

These results validate the suitability of EfficientNet for Alzheimer’s disease stage classification using structural MRI data.
