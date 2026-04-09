# ProACT: Prototype-guided Adaptive Clinical Test-time Learning

## 📌 Overview

This repository provides the official (or unofficial) implementation of **ProACT**, a clinically grounded test-time adaptation (TTA) framework designed for stable multi-center pneumoconiosis diagnosis under severe domain shifts.

Traditional TTA methods often fail in clinical scenarios due to semantic drift and lack of domain-specific constraints. ProACT addresses this issue by introducing:

- **Prototype-guided anchor-based stabilization**
- **Sensitivity-aware parameter adaptation**

These components enable robust deployment across different hospitals without requiring labeled target data.

---

## 🧠 Method

### 🔹 Key Components

1. **Prototype-guided Stabilization**
   - Constructs class-wise prototypes using reliable target samples
   - Models intra-class heterogeneity across different domains

2. **Anchor-based Reliability Estimation**
   - Uses clinically validated reference samples as anchors
   - Filters unreliable target samples to prevent drift

3. **Sensitivity-aware Parameter Update**
   - Selectively updates parameters based on sensitivity
   - Preserves clinically meaningful diagnostic features

---

## 📁 Project Structure

```code
proj4_feiqu_baseline/
├── model_cls/ # Backbone models (ResNet, ViT, etc.)
├── data/ # Dataset loading and preprocessing
├── loss/ # Loss functions
├── utils/ # Utility functions
├── test_adaptation/ # Test-time adaptation (TTA) pipeline
├── train/ # Training scripts
├── eval/ # Evaluation scripts
├── kernels/ # Custom CUDA ops (if required)
├── configs/ # Configurations (optional)
├── requirements.txt
└── README.md
```


---

## ⚙️ Installation

### 1. Create environment

```bash
conda create -n proact python=3.10 -y
conda activate proact
```

### 2. Install dependencies

```code
pip install -r requirements.txt
```
