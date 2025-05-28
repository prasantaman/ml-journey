# Logistic Regression - Complete Notes

## 🔍 Overview

Logistic Regression is a **supervised learning** algorithm used for **binary classification** tasks. Unlike Linear Regression which predicts continuous values, Logistic Regression predicts **probabilities**, which are then mapped to class labels (e.g., 0 or 1).

---

## 📈 Model Equation

Given input features $x = [x_1, x_2, \dots, x_n]$, the model computes:

$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$

To map $z$ into a probability (between 0 and 1), we use the **sigmoid function**:

$\sigma(z) = \frac{1}{1 + e^{-z}}$

Predicted probability of class 1:

$P(y=1 \mid x) = \sigma(w \cdot x + b)$

---

## 🧮 Cost Function (Loss)

The cost function used is **Binary Cross-Entropy** or **Log Loss**:

$J(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(p^{(i)}) + (1 - y^{(i)}) \log(1 - p^{(i)}) \right]$

Where:

* $p^{(i)} = \sigma(w \cdot x^{(i)} + b)$
* $y^{(i)}$ is the true label
* $m$ is the number of samples

---

## 🔁 Training the Model

**Gradient Descent** is used to minimize the cost function.

Weight and bias updates:

$w := w - \alpha \cdot \frac{\partial J(w)}{\partial w}$
$b := b - \alpha \cdot \frac{\partial J(w)}{\partial b}$

Where:

* $\alpha$ = learning rate

---

## 📊 Sigmoid Function Properties

* Input domain: $(-\infty, +\infty)$
* Output range: $(0, 1)$
* S-shaped curve (non-linear)

---

## 🚫 Why Not Linear Regression?

* Linear regression output is not bounded → not valid for probability prediction.
* Logistic Regression ensures output is always between 0 and 1 using sigmoid.

---

## ✅ Decision Rule

After predicting the probability $\hat{y}$:

* If $\hat{y} \geq 0.5 \Rightarrow$ predict class 1
* If $\hat{y} < 0.5 \Rightarrow$ predict class 0

Threshold can be adjusted for specific applications.

---

## 🧠 Applications

| Domain     | Application                         |
| ---------- | ----------------------------------- |
| Email      | Spam detection                      |
| Finance    | Credit risk modeling                |
| Healthcare | Disease prediction (e.g., diabetes) |
| E-commerce | Purchase likelihood                 |
| Law        | Recidivism prediction               |
| Tech Ads   | Click-through rate (CTR) prediction |

---

## 🔍 Pros and Cons

### ✅ Advantages:

* Simple and easy to interpret
* Fast training and prediction
* Outputs probabilities
* Works well with linearly separable data

### ❌ Disadvantages:

* Assumes linear decision boundary
* Struggles with multicollinearity
* Not suitable for complex relationships without feature engineering

---

## 🧾 Flashcards

**Q: What is the main purpose of Logistic Regression?**
A: Binary classification.

**Q: Which function is used to squash linear output into probability?**
A: Sigmoid function.

**Q: What is the cost function used in Logistic Regression?**
A: Binary Cross-Entropy (Log Loss).

**Q: Why can’t we use Linear Regression for classification?**
A: Because it doesn’t output values bounded between 0 and 1.

**Q: Which optimization algorithm is commonly used in Logistic Regression?**
A: Gradient Descent.

---

## 📦 Extras

* Logistic Regression can be regularized with **L1** (Lasso) or **L2** (Ridge) penalties.
* Multiclass version: **Multinomial Logistic Regression** (softmax based).
* Can be extended to **Elastic Net** by combining L1 and L2 regularization.

---

## 🔍 Elastic Net Regularization

Elastic Net combines **L1 (Lasso)** and **L2 (Ridge)** penalties for regularization:

$\text{Loss} = \text{CrossEntropyLoss} + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$

### Why Elastic Net?

* L1 helps in feature selection (sparsity)
* L2 helps with multicollinearity and model stability
* Elastic Net balances both effects using:

$\text{ElasticNetLoss} = \text{CrossEntropyLoss} + \lambda (\alpha \|w\|_1 + (1 - \alpha) \|w\|_2^2)$

Where:

* $\lambda$ is overall regularization strength
* $\alpha \in [0,1]$ is the mix ratio between L1 and L2

Elastic Net is helpful when you have:

* Many features
* Correlated features

## 📌 What is Variance?

**Variance** is a statistical measure that tells us how far the data values spread out from the mean (average). It shows the degree of dispersion or variability in a dataset.

- If values are tightly clustered around the mean → low variance.
- If values are widely spread out → high variance.

---

## 📐 Mathematical Definition

### 🔹 For Population:
\[
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
\]

### 🔹 For Sample:
\[
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
\]

### Where:
- \( x_i \) = Each data value
- \( \mu \) = Population mean
- \( \bar{x} \) = Sample mean
- \( N \) = Total values in population
- \( n \) = Total values in sample

---

## 🤔 Intuitive Explanation

Variance helps to answer the question:

> "How much do the values differ from the average?"

- **High Variance** → Data points are far from the mean (less consistent)
- **Low Variance** → Data points are close to the mean (more consistent)

---

## 📈 Applications of Variance

### 1. **Finance and Stock Market**
- Used to measure **risk**.
- High variance → Price is volatile → Risky investment
- Low variance → Stable performance

### 2. **Machine Learning & AI**
- Used in **Bias-Variance Tradeoff**
  - High variance → Overfitting
  - Low variance → Underfitting (if bias is also low)
- Variance is critical in model tuning and evaluation

### 3. **Quality Control (Manufacturing)**
- Checks consistency in products
- Low variance → Uniform quality
- High variance → Irregular product quality

### 4. **Climate and Weather Analysis**
- Measures stability of temperature or rainfall over time

### 5. **Sports Analytics**
- Measures player performance consistency
- High variance → Unpredictable performance

### 6. **Education & Exams**
- Helps analyze variation in student scores
- Useful for fairness and grading decisions

---

## 🔍 Related Concepts

### 🔹 Standard Deviation
- The **square root** of variance
- Easier to interpret since it's in same unit as original data

\[
\sigma = \sqrt{\sigma^2}
\]

### 🔹 Range vs Variance
| Measure | Focus           | Sensitivity |
|---------|------------------|--------------|
| Range   | Max - Min value  | Very sensitive to outliers |
| Variance| Spread around mean| Less sensitive (but still affected by outliers) |

---

## 📊 Practical Tip for Data Science

- Use **`np.var()`** for population variance and **`np.var(..., ddof=1)`** for sample variance in NumPy.
- Variance is essential in **EDA (Exploratory Data Analysis)** to check data spread before applying algorithms.
- Always **visualize** variance using histograms or boxplots for better understanding.

---

## 📌 Summary

| Term          | Meaning                              |
|---------------|--------------------------------------|
| Variance      | Average of squared deviations from mean |
| Low Variance  | More consistent data                 |
| High Variance | More spread-out / inconsistent data  |
| Uses          | ML, Finance, Quality Control, Exams, Weather, Sports |

---

## 🧠 Flashcards (Q&A Format)

**Q:** What does variance measure?  
**A:** The average squared deviation from the mean.

**Q:** What does high variance indicate?  
**A:** The data is widely spread out; less consistent.

**Q:** Is variance affected by outliers?  
**A:** Yes, heavily.

**Q:** What is the relationship between variance and standard deviation?  
**A:** Standard deviation is the square root of variance.

**Q:** Where is variance used in machine learning?  
**A:** In the bias-variance tradeoff to evaluate model generalization.

