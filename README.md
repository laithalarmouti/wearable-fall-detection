# 📌 STM32 AI Fall Detection System

This project implements a real-time fall detection system on the STM32F411RE using embedded machine learning techniques.

---

## 🚀 Overview

The system uses motion data from the MPU6050 to detect human falls in real time. A trained and quantized neural network model is deployed using TensorFlow Lite for Microcontrollers.

---

## 🧠 Features

- Real-time fall detection on microcontroller (no cloud)
- Sliding window feature extraction (statistical + frequency-domain)
- Quantized ML model (int8 / float16 optimized)
- Low-memory and low-power embedded deployment
- Designed for wearable applications

---

## ⚙️ Hardware

| Component | Details |
|-----------|---------|
| Microcontroller | STM32F411RE (ARM Cortex-M4) |
| IMU Sensor | MPU6050 (accelerometer + gyroscope) |

---

## 🧪 Methodology

- **Dataset:** SisFall dataset
- **Feature extraction:** 294 → optimized subset
- **Model:**  MLP (quantized for embedded inference)
- **Deployment:** TensorFlow Lite Micro on STM32

---

## 📊 Results

Model evaluated on the SisFall test set after int8/float16 quantization.

| Metric | Score |
|--------|-------|
| Accuracy | 84.7% |
| Precision | 86.7% |
| Recall | 63.6% |
| F1 Score | 73.4% |
| Specificity | 95.2% |
| ROC AUC | 0.899 |

> **Note:** The high specificity (95.2%) indicates the model is conservative in triggering false alarms, which is desirable in wearable safety applications. The recall gap reflects a trade-off prioritised to minimise false positives.

---

## 🎯 Goal

To evaluate the feasibility and performance of AI-based fall detection on resource-constrained embedded systems.
