# Classical ML vs. Neural Network from Scratch: CKD Prediction

An Introduction to Machine Learning assignment comparing classical models and a from-scratch neural network for binary classification of Chronic Kidney Disease (CKD) using the UCI CKD dataset.

## Overview

This notebook implements and compares three model families on the task of predicting CKD from 24 routine clinical features (400 patients):

| Model Family | Variants | Key Hyperparameters |
|---|---|---|
| Logistic Regression | LR-1, LR-2, LR-3 | Regularisation strength C = {1.0, 0.01, 100.0} |
| Random Forest | RF-1, RF-2, RF-3 | n_estimators = {10, 100}, max_depth = {3, None, 5} |
| Neural Network (NumPy) | Exp1, Exp2, Exp3 | lr = {0.01, 0.001}, hidden = {[64,32], [128,64]} |

The neural network is a 3-layer feedforward network implemented entirely in NumPy (no PyTorch/TensorFlow/JAX) with:

- ReLU hidden activations, sigmoid output
- Binary cross-entropy loss with epsilon clipping
- Analytically derived backpropagation gradients
- He (Kaiming) weight initialisation
- Mini-batch gradient descent

## Dataset

**Chronic Kidney Disease** — UCI Machine Learning Repository (ID: 336)

Soundarapandian, P., & Rubini, L. J. (2015). *Chronic kidney disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

- 400 instances, 24 features (continuous + nominal), binary target
- ~62.5% CKD / 37.5% not-CKD (mild imbalance)
- Significant missingness across multiple columns

## How to Run

### Google Colab (recommended)

1. Upload `Assignment1_ElvinCyubahiro.ipynb` to Google Colab
2. Runtime → Run all

### Local Jupyter

```bash
pip install ucimlrepo scikit-learn matplotlib seaborn pandas numpy
jupyter notebook Assignment1_ElvinCyubahiro.ipynb
```

Then use Kernel → Restart & Run All.

## Notebook Structure

| Section | Content |
|---|---|
| 1 | Library imports and environment setup |
| 2 | Dataset selection and justification |
| 3 | Data loading, EDA (6-panel figure), preprocessing pipeline |
| 4 | Classical ML: Logistic Regression (3 configs) + Random Forest (3 configs) with confusion matrices, ROC curves, PCA decision boundaries, and feature importance |
| 5 | Neural network from scratch: 8 NumPy functions, 3 experiments with learning curves, confusion matrices, ROC curves, and PCA decision boundary |
| 6 | Comparative analysis: results table, F1 bar chart, ROC overlay, discussion |
| 7 | Conclusion |
| 8 | APA references (17 sources) |
| 9 | Academic integrity statement |

## Results

All models achieve strong performance on the CKD dataset, reflecting the high discriminative power of routine clinical biomarkers:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| LR-1 (C=1.0) | 0.9875 | 1.0000 | 0.9800 | 0.9899 |
| LR-2 (C=0.01) | 0.9875 | 1.0000 | 0.9800 | 0.9899 |
| LR-3 (C=100) | 0.9500 | 0.9792 | 0.9400 | 0.9592 |
| RF-1 (10 trees, depth=3) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RF-2 (100 trees, full) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RF-3 (100 trees, depth=5) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| NN-Exp1 (lr=0.01, 64-32) | 0.9875 | 1.0000 | 0.9800 | 0.9899 |
| NN-Exp2 (lr=0.001, 64-32) | 0.9750 | 1.0000 | 0.9600 | 0.9796 |
| NN-Exp3 (lr=0.01, 128-64) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## File Structure

```
Formative/
  Assignment1_ElvinCyubahiro.ipynb   # Complete notebook
  README.md                          # This file
```
