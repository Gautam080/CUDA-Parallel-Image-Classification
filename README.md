# CUDA Parallel Image Classification

## Introduction

This repository contains code for comparing the training time of an image classification Convolutional Neural Network (CNN) using sequential and parallel programming with CUDA. Deep learning tasks, like image classification, often require significant computational resources for training complex models on large datasets. Parallel programming techniques, such as CUDA, leverage the parallel processing power of GPUs to accelerate training and reduce time.

## Rationale Behind Design Choice

### Goal
The goal of this project is to showcase the effectiveness of CUDA and parallel programming in reducing training time for image classification CNNs.

### Choice of Image Classification
Image classification serves as a computationally intensive task, making it an ideal candidate for demonstrating the benefits of parallel programming.

### Anticipated Benefits and Challenges
- Benefits: Significantly reduced training time, faster model development.
- Challenges: Setup complexity, potential code complexity.

### Performance Measurement
Training times for both sequential and parallel implementations were recorded to compare their efficiency.

## Methodology

### Experimental Setup
- Utilized Google Colab with Tesla T4 GPU for CUDA.
- Used local CPU for sequential implementation.

### Model and Dataset
- CNN architecture employed on the CIFAR-10 dataset.

### Training Procedure
Both sequential and parallel implementations followed similar training steps.

### Performance Measurement
Training times were recorded for direct comparison.

## Code Overview

### Sequential Code
```python
# Insert code snippet for sequential implementation
