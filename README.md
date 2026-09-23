# Fashion-MNIST Image Classification with CNNs

This project implements a Convolutional Neural Network (CNN) architecture to classify apparel images using two leading deep learning frameworks: **PyTorch** and **TensorFlow**.

---

## 📊 Dataset Overview

Both models are trained on the **Fashion-MNIST** benchmark dataset, natively loaded via framework utilities (`torchvision.datasets` and `tf.keras.datasets`).

* **Dataset Size:** 60,000 training images | 10,000 test images
* **Image Dimensions:** 28 × 28 pixels (Grayscale)
* **Target Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
* **Batch Size:** 64
* **Distrubtion:** The dataset is balanced, with an equal number of images (6,000) per class in the training set and 1,000 images per class in the test set.
* **Dataset Splitting:** The original training dataset was split into 90% for training (54,000 images) and 10% for validation (6,000 images). The test dataset of 10,000 images was kept separate and was not used during training or validation.

---

## 🔥 Model 1: PyTorch Implementation

### 🛠️ Data Preprocessing
* **Data Augmentation:** Training images are augmented using `RandomRotation`, `RandomAffine`, and `RandomHorizontalFlip`
Additionally, the training dataset is loaded without augmentation for the validation split, while the test dataset is also used without augmentation.
* **Preprocessing:** Pixel values are first scaled to `[0.0, 1.0]` using `ToTensor()`, then standardized using a mean of `0.2860` and standard deviation of `0.3530`.

### 🏗️ Network Architecture (TinyVGG Variant)
The model uses a sequential block-based CNN architecture:

| Block | Layer Type | Hyperparameters | Output Shape |
| :--- | :--- | :--- | :--- |
| **Block 1** | Conv2D | In: 1, Out: 32, Kernel: 3×3, Stride: 1, Padding: 1 | `(32, 28, 28)` |
| | ReLU | Activation function | `(32, 28, 28)` |
| | Conv2D | In: 32, Out: 64, Kernel: 3×3, Stride: 1, Padding: 1 | `(64, 28, 28)` |
| | ReLU | Activation function | `(64, 28, 28)` |
| | MaxPool2D | Kernel: 2×2, Stride: 2 | `(64, 14, 14)` |
| | Dropout2d | Probability: 0,25 | `(64, 14, 14)` |
| **Block 2** | Conv2D | In: 64, Out: 64, Kernel: 3×3, Stride: 1, Padding: 1 | `(64, 14, 14)` |
| | ReLU | Activation function | `(64, 14, 14)` |
| | Conv2D | In: 64, Out: 64, Kernel: 3×3, Stride: 1, Padding: 1 | `(64, 14, 14)` |
| | ReLU | Activation function | `(64, 14, 14)` |
| | MaxPool2D | Kernel: 2×2, Stride: 2 | `(64, 7, 7)` |
| | Dropout2d | Probability: 0,25 | `(64, 14, 14)` |
| **Classifier**| Flatten | Flattens spatial dimensions (64 × 7 × 7) | `(3136)` |
| | Linear | Input: 3136, Output: 128 | `(128)` |
| | ReLU | Activation function | `(128)` |
| | Dropout | Probability: 0,5 | `(128)` |
| | Linear | Input: 128, Output: 10 (Logits) | `(10)` |

### ⚙️ Training Configurations  
* **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`)
* **Optimizer:** Adam Optimizer
* **Training Epochs:** 10

### 📈 Evaluation & Results
* **Final Training Loss:** 0.3876
* **Test Accuracy:** **90.34%**

#### Class-wise Performance (Confusion Matrix Summary)

| Class | Correct / Total | Accuracy (%) |
| :--- | :---: | :---: |
| **T-shirt/top** | 870 / 1000 | 87.0% |
| **Trouser** | 983 / 1000 | 98.3% |
| **Pullover** | 901 / 1000 | 90.1% |
| **Dress** | 926 / 1000 | 92.6% |
| **Coat** | 871 / 1000 | 87.1% |
| **Sandal** | 952 / 1000 | 95.2% |
| **Shirt** | 614 / 1000 | 61.4% |
| **Sneaker** | 981 / 1000 | 98.1% |
| **Bag** | 979 / 1000 | 97.9% |
| **Ankle boot** | 957 / 1000 | 95.7% |

---

## ⚡ Model 2: TensorFlow / Keras Implementation

### 🛠️ Data Preprocessing
* **Data Augmentation:** Training images are augmented using `rotation_range`, `width_shift_range`, `height_shift_range`,`zoom_range` and `horizontal_flip`.
* **Normalization:** Pixel values normalized from `[0, 255]` to `[0.0, 1.0]` via division by 255.0.
* **Reshaping:** Channel dimension explicitly appended to match Keras expectations: `(28 × 28)` → `(28 × 28 × 1)` `(Height × Width × Channel)`.

### 🏗️ Network Architecture (Replicated TinyVGG)

| **Block**      | **Layer Type** | **Hyperparameters**                                     | **Output Shape** |
| -------------- | -------------- | ------------------------------------------------------- | ---------------- |
| **Input**      | Input          | 28 × 28 × 1                                             | `(28, 28, 1)`    |
| **Block 1**    | Conv2D         | 32 filters, 3×3 kernel, stride 1, ReLU, padding: valid  | `(26, 26, 32)`   |
|                | Conv2D         | 32 filters, 3×3 kernel, stride 1, ReLU, padding: valid  | `(24, 24, 32)`   |
|                | MaxPooling2D   | 2×2 pool, stride 2                                      | `(12, 12, 32)`   |
|                | Dropout        | Rate: 0.25                                              | `(12, 12, 32)`   |
| **Block 2**    | Conv2D         | 128 filters, 3×3 kernel, stride 1, ReLU, padding: valid | `(10, 10, 128)`  |
|                | Conv2D         | 128 filters, 3×3 kernel, stride 1, ReLU, padding: valid | `(8, 8, 128)`    |
|                | MaxPooling2D   | 2×2 pool, stride 2                                      | `(4, 4, 128)`    |
|                | Dropout        | Rate: 0.25                                              | `(4, 4, 128)`    |
| **Classifier** | Flatten        | 128 × 4 × 4 = 2048 features                             | `(2048)`         |
|                | Dense          | 128 units, ReLU                                         | `(128)`          |
|                | Dropout        | Rate: 0.5                                               | `(128)`          |
|                | Dense          | 10 units, Softmax                                       | `(10)`           |


### ⚙️ Training Configurations
* **Loss Function:** Cross-Entropy Loss (`sparse_categorical_crossentropy`)
* **Optimizer:** Adam Optimizer
* **Training Epochs:** 15

 
### 📈 Evaluation & Results
* **Final Training Loss:** 0.3345
* **Test Accuracy:** **90.77%**

#### Class-wise Performance (Confusion Matrix Summary)

| Class | Correct / Total | Accuracy (%) |
| :--- | :---: | :---: |
| **T-shirt/top** | 894 / 1000 | 89.4% |
| **Trouser** | 984 / 1000 | 98.4% |
| **Pullover** | 869 / 1000 | 86.9% |
| **Dress** | 917 / 1000 | 91.7% |
| **Coat** | 860 / 1000 | 86.0% |
| **Sandal** | 980 / 1000 | 98.0% |
| **Shirt** | 682 / 1000 | 68.2% |
| **Sneaker** | 954 / 1000 | 95.4% |
| **Bag** | 984 / 1000 | 98.4% |
| **Ankle boot** | 953 / 1000 | 95.3% |


---

## 🔗 References & Learning Sources
* [Daniel Bourke's TensorFlow Deep Learning Course](https://dev.mrdbourke.com/tensorflow-deep-learning/03_convolutional_neural_networks_in_tensorflow/#1-import-and-become-one-with-the-data_1)
* [Learn PyTorch for Deep Learning Curriculum](https://www.learnpytorch.io/03_pytorch_computer_vision/)
* [CNN Explainer Interactive Visualizer](https://poloclub.github.io/cnn-explainer/)
 