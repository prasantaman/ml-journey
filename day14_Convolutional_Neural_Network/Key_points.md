## 🧠 CNN Basics & Related Concepts (Quick Notes)

---

### ✅ Flattening

* Converts a multi-dimensional tensor (e.g., 2D image) into a 1D vector.
* Example: 28×28 image → 784-length vector.
* **Downside**: Spatial (pixel position) info is lost.

---

### ✅ Grayscale Image

* Image with only one color channel.
* Each pixel has intensity (0 = black, 255 = white).
* Size: e.g., 28×28 for MNIST.

---

### ✅ Convolution

* Operation that slides a filter over the image to extract features.
* Filter (kernel) detects edges, patterns, etc.
* Output is a **feature map**.

---

### ✅ Stride

* Defines how many pixels the filter jumps after each operation.
* Stride = 1 → moves 1 pixel at a time (more overlap).
* Stride = 2 → skips 1 pixel, smaller output.

---

### ✅ Padding

* Adds pixels (usually zeros) around the image border.
* Purpose:

  * Preserve image size
  * Capture edge features
* Padding = 'same' keeps output size same as input.

---

### ✅ Pooling Layer

* Reduces size of feature maps (downsampling).
* Types:

  * Max Pooling: takes max value in region
  * Average Pooling: takes average value
* Makes model faster, reduces overfitting

---

### ✅ Fully Connected Layer (FC Layer)

* Final layer in CNN.
* Takes flattened input and predicts output class.
* Works like a regular neural network.

---

### ✅ MNIST Dataset

* Handwritten digit dataset (0–9).
* 70,000 grayscale images (28×28):

  * 60k train, 10k test
* Widely used for digit classification

---

### ✅ ResNet (Residual Network)

* Deep CNN model with **skip connections** (residuals)
* Formula: `y = F(x) + x`
* Solves vanishing gradient problem in deep networks
* Famous ResNet variants: ResNet-18, 50, 101, 152

---

### ✅ ImageNet Dataset

* Huge image dataset (14M+ images, 20K+ classes)
* Used in training large models
* ImageNet Challenge (ILSVRC) made models like ResNet, VGG famous
* Standard benchmark in computer vision
