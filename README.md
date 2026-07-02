# MNIST Image Classifier

This project is a simple image classification model implemented using Convolutional Neural Networks (CNNs) in both PyTorch and TensorFlow.

The project is divided into two parts:
- PyTorch model implementation
- TensorFlow model implementation

---

## Dataset

Both models are trained on the Fashion-MNIST dataset, which is a standard benchmark dataset for image classification tasks. It is available directly through both the PyTorch and TensorFlow frameworks.

---

## Model: PyTorch

The PyTorch implementation uses the Fashion-MNIST dataset, which is provided as part of the torchvision package.

The model is built using a simple CNN architecture for classification of grayscale 28×28 images.

### Data Preprocessing

Since the dataset is already included in PyTorch, most preprocessing steps are handled automatically.

The main preprocessing step is normalization of the images before training.

Using ToTensor(), the images are converted into tensors with shape:
- (1 × 28 × 28) (channels × height × width)

The batch size used in training is:
- 64

---

## Model: TensorFlow

The TensorFlow implementation also uses the Fashion-MNIST dataset, available through tf.keras.datasets.

### Data Preprocessing

The dataset is loaded with minimal preprocessing required.

Before training, the pixel values are normalized to improve training stability.

Additionally, a channel dimension is added to the images, converting them from:
- (28 × 28) → (28 × 28 × 1)