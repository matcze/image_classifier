# Fashion-MNIST Image Classification with CNNs

This project implements a Convolutional Neural Network (CNN) architecture to classify apparel images using two leading deep learning frameworks: **PyTorch** and **TensorFlow**.

---

## 📊 Dataset Overview

Both models are trained on the **Fashion-MNIST** benchmark dataset, natively loaded via framework utilities (`torchvision.datasets` and `tf.keras.datasets`).

* **Dataset Size:** 60,000 training images | 10,000 test images
* **Image Dimensions:** 28 × 28 pixels (Grayscale)
* **Target Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
* **Batch Size:** 64

---

## 🔥 Model 1: PyTorch Implementation

### 🛠️ Data Preprocessing
* **Normalization:** Pixel values scaled to a `[0.0, 1.0]` range.
* **Tensor Transformation:** Converted using `ToTensor()`, mapping shapes to `(Channel × Height × Width)` → `(1 × 28 × 28)`.


### 🏗️ Network Architecture (TinyVGG Variant)
The model uses a sequential block-based CNN architecture:

| Block | Layer Type | Hyperparameters | Output Shape |
| :--- | :--- | :--- | :--- |
| **Block 1** | Conv2D | In: 1, Out: 32, Kernel: 3×3, Stride: 1, Padding: 1 | `(32, 28, 28)` |
| | ReLU | Activation function | `(32, 28, 28)` |
| | Conv2D | In: 32, Out: 64, Kernel: 3×3, Stride: 1, Padding: 1 | `(64, 28, 28)` |
| | ReLU | Activation function | `(64, 28, 28)` |
| | MaxPool2D | Kernel: 2×2, Stride: 2 | `(64, 14, 14)` |
| **Block 2** | Conv2D | In: 64, Out: 32, Kernel: 3×3, Stride: 1, Padding: 1 | `(32, 14, 14)` |
| | ReLU | Activation function | `(32, 14, 14)` |
| | Conv2D | In: 32, Out: 64, Kernel: 3×3, Stride: 1, Padding: 1 | `(64, 14, 14)` |
| | ReLU | Activation function | `(64, 14, 14)` |
| | MaxPool2D | Kernel: 2×2, Stride: 2 | `(64, 7, 7)` |
| **Classifier**| Flatten | Flattens spatial dimensions (64 × 7 × 7) | `(3136)` |
| | Linear | Input: 3136, Output: 128 | `(128)` |
| | ReLU | Activation function | `(128)` |
| | Linear | Input: 128, Output: 10 (Logits) | `(10)` |

### ⚙️ Training Configurations
* **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`)
* **Optimizer:** Adam Optimizer
* **Training Epochs:** 5

### 📈 Evaluation & Results
* **Final Training Loss:** 154.3608
* **Test Accuracy:** **92.27%**

#### Class-wise Performance (Confusion Matrix Summary)

| Class | Correct / Total | Accuracy (%) |
| :--- | :---: | :---: |
| **Trouser** | 992 / 1000 | 99.2% |
| **Sneaker** | 989 / 1000 | 98.9% |
| **Bag** | 982 / 1000 | 98.2% |
| **Sandal** | 973 / 1000 | 97.3% |
| **Ankle boot** | 952 / 1000 | 95.2% |
| **Dress** | 926 / 1000 | 92.6% |
| **Pullover** | 908 / 1000 | 90.8% |
| **T-shirt/top** | 904 / 1000 | 90.4% |
| **Coat** | 864 / 1000 | 86.4% |
| **Shirt** | 737 / 1000 | 73.7% |

---

## ⚡ Model 2: TensorFlow / Keras Implementation

### 🛠️ Data Preprocessing
* **Normalization:** Pixel values normalized from `[0, 255]` to `[0.0, 1.0]` via division by 255.0.
* **Reshaping:** Channel dimension explicitly appended to match Keras expectations: `(28 × 28)` → `(28 × 28 × 1)` `(Height × Width × Channel)`.

### 🏗️ Network Architecture (Replicated TinyVGG)


 





---

## 🔗 References & Learning Sources
* [Daniel Bourke's TensorFlow Deep Learning Course](https://dev.mrdbourke.com/tensorflow-deep-learning/03_convolutional_neural_networks_in_tensorflow/#1-import-and-become-one-with-the-data_1)
* [Learn PyTorch for Deep Learning Curriculum](https://www.learnpytorch.io/03_pytorch_computer_vision/)
* [CNN Explainer Interactive Visualizer](https://poloclub.github.io/cnn-explainer/)
