# Python – Model Training and Comparison

## 1. Overview

This directory contains the Python implementation for training and evaluating the audio classification models used in the Offline Voice-Controlled Fan project.

Two deep learning architectures are considered:

- **DS-CNN (Depthwise Separable Convolutional Neural Network)**
- **ResNet8 (Residual Neural Network)**

The purpose is to train both models, evaluate their performance, and compare their suitability for deployment on the ESP32 microcontroller.

## 2. Directory Structure

```text
python/
├── train/
│   ├── train_ds_cnn.py
│   └── train_resnet8.py
│
├── comparison/
│   └── compare_models.py
│
├── results/
│   └── ...
│
├── requirements.txt
└── README.md
```

*Note: Replace the example script names above with the actual filenames in this repository.*

## 3. Models

### DS-CNN

A lightweight convolutional neural network architecture that uses depthwise separable convolutions to reduce computational complexity and model size.

### ResNet8

A compact residual neural network architecture that uses shortcut connections to improve feature learning and support efficient audio classification.

## 4. Training

The training scripts are located in the `train/` directory.

Run the corresponding script to train each model:

```bash
python train/<ds_cnn_training_script>.py
```

```bash
python train/<resnet8_training_script>.py
```

Replace the placeholders with the actual script filenames.

## 5. Model Comparison

The comparison script evaluates the trained models using the evaluation metrics implemented in the project.

Run:

```bash
python comparison/<comparison_script>.py
```

The comparison results are stored in the `results/` directory.

## 6. Evaluation

The models can be compared using metrics such as:

- Classification accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Model size
- Computational requirements

Only metrics actually calculated by the comparison script should be included in the final results.

## 7. Deployment

After evaluation, the selected model is prepared for deployment on the ESP32 using TensorFlow Lite Micro.

The deployed model is integrated into the firmware for offline voice command recognition.

## 8. Environment Setup

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## 9. Project Scope

This Python module focuses on:

1. Training the DS-CNN model.
2. Training the ResNet8 model.
3. Evaluating and comparing both architectures.
4. Supporting model selection for embedded deployment.

For the ESP32 firmware implementation, refer to the main project README.