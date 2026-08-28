<div align="left">

# 🫀 Automated ECG Delineation & Multi-Label Arrhythmia Detection

**End-to-end pipeline for ECG signal pre-processing, wavelet-based fiducial point delineation, and multi-label cardiac arrhythmia classification using deep residual neural networks.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[Key Features](#-key-features) • [Pipeline Architecture](#-pipeline-architecture) • [Project Structure](#-project-structure) • [Installation](#-installation--setup) • [Usage](#-usage-guide) • [Model & Results](#-model-architecture--results)

</div>

---

## 📌 Overview

Cardiovascular diseases (CVDs) are the leading cause of mortality globally. Accurate and automated detection of cardiac arrhythmias from Electrocardiogram (ECG) recordings is vital for early diagnosis and clinical decision support.

This repository provides a complete machine learning framework:
1. **Signal Preprocessing & Baseline Correction**: Histogram mode isoline correction, median filtering with PCHIP interpolation, zero-phase Butterworth filtering, and Gaussian notch filtering.
2. **ECG Delineation via Stationary Wavelet Transform (SWT)**: Bidirectional SWT decomposition, adaptive thresholding, QRS detection, and sigmoid-based wave suppression for precise P-wave, QRS-complex, and T-wave segmentation.
3. **Multi-Label Arrhythmia Classification**: A deep Artificial Neural Network (ANN) featuring **Residual Skip Connections**, Batch Normalization, Dropout regularization, and **Cosine Annealing Learning Rate scheduling** to detect 5 distinct cardiac diagnostic classes simultaneously.

---

## ✨ Key Features

- 🧹 **Advanced Signal Preprocessing**:
  - Mode-based histogram offset correction.
  - Moving-window median baseline wander removal interpolated using **PCHIP**.
  - Zero-phase Butterworth bandpass filter (0.5–30 Hz) with edge padding to eliminate boundary artifacts.
  - Frequency-domain Gaussian notch filter targeting 50/60 Hz powerline interference and harmonics.

- 🌊 **Wavelet-Driven ECG Delineation**:
  - Bidirectional Stationary Wavelet Transform (SWT) with Haar and Biorthogonal wavelets.
  - Adaptive threshold search over multi-scale sub-bands for robust R-peak and QRS detection.
  - Sigmoid-window suppression (`Remove_PQRS` / `Remove_QRST`) to isolate low-amplitude P and T waves.

- 🧠 **Deep Residual Artificial Neural Network (ANN)**:
  - 5-layer deep neural architecture equipped with a skip/residual connection from input to intermediate layers to mitigate vanishing gradients.
  - Multi-label classification across 5 key diagnostic superclasses.
  - Sample-weighted cross-entropy loss to handle heavy class imbalance.
  - Cosine Decay Restarts learning rate schedule for optimal parameter convergence.

- 🎯 **Per-Class Threshold Tuning**:
  - Independent threshold optimization ($t \in [0.20, 0.80]$) for each arrhythmia label to maximize validation Macro F1-score.

---

## 🔬 Arrhythmia Diagnostic Categories

The classifier categorizes 12-lead ECG records into 5 non-mutually exclusive diagnostic superclasses (Multi-Label Classification):

| Code | Diagnostic Class | Clinical Description |
| :--- | :--- | :--- |
| **`NORM`** | **Normal ECG** | Normal sinus rhythm and waveform morphology |
| **`MI`** | **Myocardial Infarction** | Heart attack / ischemia-related cardiac tissue damage |
| **`STTC`** | **ST/T-Change** | Non-specific ST-segment or T-wave abnormalities |
| **`HYP`** | **Hypertrophy** | Ventricular or atrial muscle wall enlargement |
| **`CD`** | **Conduction Disturbance** | Bundle branch blocks & intraventricular conduction delays |

---
