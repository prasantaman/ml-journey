## 📘 Convolutional Neural Network (CNN) – Complete Notes

### 🔹 What is CNN?

CNN (Convolutional Neural Network) ek deep learning architecture hai jo especially **image data** ke liye design kiya gaya hai. Iska use mainly **computer vision** tasks mein hota hai — jaise image classification, object detection, facial recognition, etc.

---

### 🔸 Why Not Feedforward Neural Networks for Images?

* Feedforward Neural Networks (FNN) require **flattened input** (1D vector). E.g., 28x28 image = 784 input neurons.
* Too many **weights and parameters** → slow and overfitting.
* **Spatial information** (like pixels’ position) lost in flattening.
* CNNs solve this by preserving **spatial structure** using convolution operations.

---

### 🔸 CNN Architecture Overview

1. **Input Layer** – Image (e.g., 28x28x1 grayscale ya 32x32x3 RGB)
2. **Convolution Layer (Conv)** – Feature extraction using filters.
3. **Activation Layer (ReLU)** – Non-linearity introduce karta hai.
4. **Pooling Layer** – Downsampling karta hai (max/average).
5. **Fully Connected Layer (FC)** – Classification karta hai.
6. **Output Layer** – Final prediction (e.g., softmax for class probs).

---

### 🔸 Core Building Blocks

#### 1. 🧩 Convolution Layer

* **Filters (Kernels)** = Small matrix (e.g., 3x3, 5x5) that slides over image.
* Performs **dot product** with image patch.
* Extracts **features** like edges, corners, textures, etc.

**Formula:**
If input = $n \times n$, filter = $f \times f$, stride = $s$, padding = $p$

Output size:

$$
\text{Output} = \frac{n - f + 2p}{s} + 1
$$

#### 2. 🔥 ReLU Layer (Rectified Linear Unit)

* ReLU(x) = max(0, x)
* Adds non-linearity
* Faster convergence and better training

#### 3. 🧊 Pooling Layer

* Reduces spatial size of feature maps → less computation
* Two common types:

  * **Max Pooling**: takes maximum value
  * **Average Pooling**: takes average value

**Example**: 2x2 Max Pooling on:

```
[1 3]
[4 2] → output: 4
```

#### 4. 🧠 Fully Connected Layer (Dense Layer)

* Flatten the image → connect to FC layer → neurons represent features
* Acts like traditional neural network for final classification

#### 5. 📤 Output Layer

* Usually **Softmax** for multi-class classification
* Gives probability distribution across classes

---

### 🔸 Hyperparameters in CNN

| Parameter                         | Description                                   |
| --------------------------------- | --------------------------------------------- |
| Filter size                       | Size of the kernel (3x3, 5x5)                 |
| Number of filters                 | More filters = more features extracted        |
| Stride                            | Step size for filter movement                 |
| Padding                           | Add zeros around image to control output size |
| Pool size                         | Size of pooling region (2x2, 3x3)             |
| Epochs, batch size, learning rate | General training hyperparams                  |

---

### 🔸 CNN in Code (with `Keras`)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

### 📊 Applications of CNN

| Area                 | Use-case                             |
| -------------------- | ------------------------------------ |
| Image Classification | Cats vs Dogs, MNIST digits           |
| Object Detection     | YOLO, SSD, RCNN                      |
| Medical Imaging      | Tumor detection, X-ray analysis      |
| Face Recognition     | Face ID, attendance systems          |
| Self-driving Cars    | Lane detection, obstacle recognition |
| Satellite Imaging    | Land-use classification              |

---

### 🔍 Advanced CNN Topics (Optional)

* **Batch Normalization** – stabilizes training
* **Dropout** – avoids overfitting
* **Transfer Learning** – use pre-trained models like VGG, ResNet
* **Data Augmentation** – rotate/flip images to increase dataset size
* **1x1 Convolutions** – used in GoogLeNet (Inception)
* **Dilated Convolutions** – for larger receptive field without pooling

---

### 🧠 Mathematical Intuition

* Each convolution operation = **matrix dot product**
* Gradient descent + backpropagation used to learn filter weights
* Filters start randomly and get updated during training

---

### 📌 Important Q\&A Flashcards

**Q1. What is the main advantage of CNN over regular neural networks?**
A. CNN preserves spatial structure and requires fewer parameters.

**Q2. What does a filter do in CNN?**
A. It extracts features like edges, corners from input image.

**Q3. What is Max Pooling?**
A. A downsampling technique that keeps the maximum value in a region.

**Q4. Why use ReLU instead of sigmoid/tanh?**
A. Faster, avoids vanishing gradient, introduces non-linearity.

**Q5. What is padding in CNN?**
A. Adding zeros around the image to control output dimensions.
