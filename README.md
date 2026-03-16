# MRI Dementia Classification using Deep Learning

Deep learning model for classifying brain MRI scans into dementia severity categories using a modified **ResNet-18 convolutional neural network** implemented in **PyTorch**.

This project focuses on building a robust training pipeline while addressing key challenges in medical imaging such as **class imbalance, overfitting, and hyperparameter optimization**.

---

# Project Overview

Brain MRI scans contain structural information that can help identify neurological conditions such as dementia. However, training neural networks on medical imaging data presents challenges:

- High-dimensional inputs  
- Limited labeled datasets  
- Class imbalance  
- Overfitting due to augmented datasets  

This project builds an end-to-end pipeline that preprocesses MRI images, trains a convolutional neural network, evaluates performance, and tunes hyperparameters to improve generalization.

---

# Dataset

The dataset contains labeled MRI scans categorized into four classes:

| Class | Description |
|------|-------------|
| Non Demented | No dementia symptoms |
| Very Mild Demented | Early cognitive decline |
| Mild Demented | Moderate cognitive impairment |
| Moderate Demented | Advanced dementia |

Dataset split:

| Split | Images |
|------|--------|
| Training | 23,788 |
| Validation | 5,097 |
| Test | 5,097 |

Images are organized into folders and loaded using `torchvision.datasets.ImageFolder`.


---

# Model Architecture

The project uses a **modified ResNet-18 architecture**.

Modifications:

1. **Single-channel input**

MRI scans are grayscale, so the first convolution layer was modified:

```python
Conv2D(in_channels=1, out_channels=64)

Per-Image Z-Score Normalization

Removes brightness variation between scans.

x_norm = (x - μ) / σ

Where
μ = mean intensity of the image
σ = standard deviation of the image

Training Strategy
Loss Function

Weighted Cross Entropy Loss is used to address class imbalance.

L = - Σ wi * yi * log(ŷi)

where wi are class weights.
