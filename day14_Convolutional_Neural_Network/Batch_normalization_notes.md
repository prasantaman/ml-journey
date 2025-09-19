## 📘 Batch Normalization (BN) – Detailed Explanation

### 🔹 Problem Batch Normalization Solve Karta Hai

Neural networks me **training difficult ho sakti hai** agar inputs har layer me continuously change ho rahe ho. Is problem ko kehte hain:

* **Internal Covariate Shift**
  Matlab: Jab ek layer ke inputs training ke dauraan continuously change ho, toh next layers ko har iteration ke liye naye distribution ke hisaab se adapt karna padta hai.

Is wajah se training slow hoti hai aur gradients unstable ho sakte hain.

---

### 🔸 Batch Normalization Ka Idea

* Har mini-batch ke inputs ko normalize karna layer-wise.
* Inputs ka mean zero aur variance one kar dena.
* Fir trainable parameters se unhe scale aur shift karna, taaki network apne optimal representation seekh sake.

---

### 🔸 Mathematical Steps of Batch Normalization

Suppose a layer ka input ek vector $x = (x_1, x_2, ..., x_m)$ hai mini-batch ke $m$ samples se.

1. **Compute mean** of mini-batch:

$$
\mu_B = \frac{1}{m} \sum_{i=1}^m x_i
$$

2. **Compute variance** of mini-batch:

$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2
$$

3. **Normalize** each input:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

($\epsilon$ ek small number hota hai for numerical stability, e.g., $10^{-5}$)

4. **Scale and shift** (trainable parameters):

$$
y_i = \gamma \hat{x}_i + \beta
$$

* $\gamma$ (scale) and $\beta$ (shift) are learnable parameters updated during training.
* Ye network ko flexibility dete hain ki wo normalized data ko adjust kar sake agar zarurat ho.

---

### 🔸 Intuition Behind BN

* Har layer ko inputs ka **standardized version** milta hai.
* Isse gradient descent smooth hota hai.
* Faster convergence hoti hai.
* Network easily **deeper** ho sakta hai without vanishing/exploding gradients.

---

### 🔸 Where to apply Batch Normalization?

* Usually Conv layer ke baad, Activation (ReLU) ke pehle ya baad lagate hain (both ways work).
* FC layers me bhi laga sakte hain.
* Popular CNN architectures (ResNet, Inception) me BN standard hai.

---

### 🔸 Benefits of Batch Normalization

| Benefit                              | Explanation                                                                                     |
| ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| Faster Training                      | Network zyada quickly converge karta hai.                                                       |
| Higher Learning Rate                 | Large learning rates bhi use kar sakte hain.                                                    |
| Regularization Effect                | Thoda noise add hota hai due to mini-batch mean/variance estimation → overfitting kam hota hai. |
| Reduced Dependence on Initialization | Weight initialization pe kam sensitive hota hai.                                                |
| Allows Deeper Networks               | Helps in training very deep networks without gradient issues.                                   |

---

### 🔸 Batch Normalization vs Other Normalizations

| Technique              | When Used / Advantage                           |
| ---------------------- | ----------------------------------------------- |
| Batch Normalization    | Most common, during training on mini-batches    |
| Layer Normalization    | For RNNs, sequence models, normalize per layer  |
| Instance Normalization | Style transfer tasks                            |
| Group Normalization    | Small batch sizes, normalize groups of channels |

---

### 🔸 Implementation Example (Keras)

```python
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation

model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)),
    BatchNormalization(),
    Activation('relu'),
    ...
])
```

---

### 🧠 Important Points to Remember

* During **training**, mean and variance are computed per batch.
* During **inference**, use running average of mean and variance (tracked during training).
* Batch size should not be too small; otherwise, estimates of mean and variance become noisy.

---

### 📌 Quick Q\&A Flashcards

**Q1. Batch Normalization ka main purpose kya hai?**
A. Neural network ke training ko fast aur stable banana by normalizing layer inputs.

**Q2. Batch Normalization me $\gamma$ aur $\beta$ kya karte hain?**
A. Scale aur shift karte hain normalized inputs ko, trainable parameters hote hain.

**Q3. Kya Batch Normalization sirf convolution layers me use hota hai?**
A. Nahi, fully connected layers me bhi use hota hai.

**Q4. Batch Normalization kyu training me regularization ka kaam karta hai?**
A. Mini-batch statistics me thoda randomness aata hai jo overfitting reduce karta hai.
