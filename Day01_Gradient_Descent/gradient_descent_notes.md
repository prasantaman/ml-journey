## 📘 Gradient Descent – Full Explanation

---

### 🔹 What is Gradient Descent?

Gradient Descent is an **optimization algorithm** used to minimize the **cost/loss function** in machine learning and deep learning models by updating the model's parameters.

It adjusts weights to find the **minimum point (lowest error)** of the cost function.

---

### 🔹 Intuition

Imagine you are standing on a hill in fog, and want to reach the bottom. You take small steps **in the direction of steepest descent** — this is what gradient descent does mathematically!

---

### 🔹 Mathematical Formula

$$
\theta := \theta - \alpha \cdot \nabla_\theta J(\theta)
$$

Where:

* $\theta$ = model parameters (weights)
* $\alpha$ = learning rate (step size)
* $J(\theta)$ = cost function
* $\nabla_\theta J(\theta)$ = gradient (slope) of cost function w\.r.t. $\theta$

---

### 🔹 Types of Gradient Descent

| Type                                  | Description                             | Pros                         | Cons                            |
| ------------------------------------- | --------------------------------------- | ---------------------------- | ------------------------------- |
| **Batch Gradient Descent**            | Uses entire dataset to compute gradient | Stable convergence           | Slow for large datasets         |
| **Stochastic Gradient Descent (SGD)** | Uses one data point per update          | Faster per step              | Noisy updates, may not converge |
| **Mini-Batch Gradient Descent**       | Uses small batches (e.g., 32, 64, 128)  | Balance of speed + stability | Hyperparameter tuning needed    |

---

### 🔹 Learning Rate $(\alpha)$

* **Too small**: Slow convergence
* **Too large**: May overshoot or diverge

🧠 **Tip:** Use techniques like **Learning Rate Scheduling** or **Adaptive optimizers** (like Adam).

---

### 🔹 Cost Function Example (for Linear Regression)

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
$$

Gradient w\.r.t. $\theta$:

$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

---

### 🔹 Gradient Descent in Neural Networks

* Works layer by layer using **Backpropagation**.
* Gradients flow **from output to input layer**.
* Optimizers (like Adam, RMSprop) are enhanced versions of gradient descent.

---

### 🔹 Advanced Variants of Gradient Descent

| Optimizer    | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| **Momentum** | Adds velocity to gradient update, smooths path               |
| **RMSprop**  | Uses squared gradient average for adaptive learning rate     |
| **Adam**     | Combines Momentum + RMSprop (widely used)                    |
| **Adagrad**  | Adapts learning rate per parameter (not ideal for deep nets) |
| **Adadelta** | Extension of Adagrad to reduce aggressive decay              |

---

### 🔹 Challenges in Gradient Descent

* Local minima vs. global minima
* Vanishing/exploding gradients
* Choosing correct learning rate
* Saddle points (flat regions)

---

### 🔹 Visualization

* **Gradient** = slope of the cost function.
* Gradient Descent = walking down the slope.
* At the **minimum**, the slope = 0 (flat).

---

## 📌 Flashcards

**Q1: What is the goal of Gradient Descent?**
A: Minimize the cost/loss function by updating parameters.

**Q2: Which Gradient Descent type is fastest per step?**
A: Stochastic Gradient Descent (SGD)

**Q3: What happens if learning rate is too high?**
A: May overshoot or diverge.

**Q4: What does Adam optimizer combine?**
A: Momentum and RMSprop

**Q5: What is gradient in gradient descent?**
A: The derivative of the cost function w\.r.t. model parameters.
