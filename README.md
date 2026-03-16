# MRI Alzheimers Classification using Deep Learning

Deep learning model for classifying brain MRI scans into dementia severity categories using a modified ResNet-18 convolutional neural network.

# Dataset

The dataset contains labeled MRI scans categorized into four classes:

Class Description 

      |     Non Demented     |     Very Mild Demented   |     Mild Demented     |     Moderate Demented 

![000cdcc4-3e54-4034-a538-203c8047b564](https://github.com/user-attachments/assets/42cfbf6e-4216-4211-8734-b91c66d20cde)
![0a2db21e-81d3-461c-a23e-c133096d8f0a](https://github.com/user-attachments/assets/7a11382c-a7c7-4bf8-b084-90e70fe63f54)
![00a9c4ad-c06d-431d-a5c9-1dc324db0632](https://github.com/user-attachments/assets/81f9722b-c2a4-4687-893e-260314675fa7)
![00ca16fb-ec46-436e-b108-8ea52a52839a](https://github.com/user-attachments/assets/27d3af1e-5056-4e51-9dfb-5bb1c4409b6a)
<br>
Dataset split:
| Split | Images |
|------|--------|
| Training | 23,788 |
| Validation | 5,097 |
| Test | 5,097 |

Images are organized into folders and loaded using `torchvision.datasets.ImageFolder`.


# Model Architecture

The project uses a modified ResNet-18 architecture for its residual blocks, solving the vanishing gradient problem.

Modifications:

- MRI scans are grayscale, so the first convolution layer was modified
- Per-Image Z-Score Normalization
- Removes brightness variation between scans.

*x_norm = (x - μ) / σ*

Loss Function:
Weighted Cross Entropy Loss is used to address class imbalance.

*L = - Σ wi * yi * log(ŷi)*

Optimizer: AdamW

**Tools**
- Python
- PyTorch
- Torchvision
- CUDA 
- Optuna
- NumPy
