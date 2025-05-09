# Machine Learning Foundations – Core Concepts & Optimizations

---

## 🔍 1. Homoscedasticity
**Definition**: When the **variance of error terms remains constant** across all levels of the independent variable.

- Important in **Linear Regression** assumptions.
- Opposite: **Heteroscedasticity** (variance changes).

---

## ⚙️ 2. Deterministic Model
**Meaning**: A model whose outcome is fully determined by its input values and parameters — **no randomness involved**.

Example: Simple Linear Regression always gives the same output for the same input.

---

## 🛠️ 3. Hyperparameter
**Definition**: Parameters that are **set before training** and **not learned from data**.

Examples:
- Learning rate
- Regularization strength
- Number of epochs

---

## 🔁 4. Hyperparameter Tuning
**Meaning**: The process of **finding the best hyperparameter values** to improve model performance.

Methods:
- Grid Search
- Random Search
- Bayesian Optimization

---

## 🎯 5. Optimize
**Meaning**: To **adjust parameters** in order to **minimize or maximize** a specific function (usually cost or accuracy).

---

## 🧠 6. Regularization
**Definition**: A technique to **prevent overfitting** by **adding a penalty term** to the cost function.

- **Ridge (L2)**: Adds square of weights.
- **Lasso (L1)**: Adds absolute value of weights.
- **ElasticNet**: Combines both L1 + L2.

---

## 💡 7. Hypothesis
**Meaning**: A **model's prediction function** based on input features.

Mathematically:  
\[
h(x) = w \cdot x + b
\]

---

## 📉 8. Cost Function vs. Loss Function

| Term            | Meaning                                      |
|-----------------|----------------------------------------------|
| **Loss Function** | Error for a **single data point**            |
| **Cost Function** | **Average loss** over all training examples |

---

## 📉 9. Steepest Descent
**Meaning**: Moving in the direction where the cost function **decreases fastest** (opposite of gradient).

---

## ⚡ 10. Stochastic Gradient Descent (SGD)
**Definition**: Updates model weights **after each training example**, making it faster and suitable for large datasets.

---

## 🔢 11. Gradient Descent Code (Explanation)

Python code updates weights `w` and bias `b` by calculating gradient from error:

```python
w = w - lr * dw
b = b - lr * db
````

Each epoch:

* Predict: `y_pred = w*X + b`
* Calculate error
* Update using gradient

---

## 🕳️ 12. Local Minima

**Definition**: A point where the function is lower than nearby points but **not the lowest globally**.

---

## 🟠 13. Convex vs. Non-Convex Functions

| Type           | Shape         | Gradient Behavior  |
| -------------- | ------------- | ------------------ |
| **Convex**     | Bowl-shaped   | One global minimum |
| **Non-Convex** | Bumpy or wavy | Many local minima  |

---

## ⚙️ 14. Optimizers – Key Concepts

### 🌀 Momentum

"Uses **past gradients** to smooth updates and accelerate learning."

### 📉 RMSProp

"Applies **adaptive learning rate** for each parameter using moving average of squared gradients."

### 🤖 Adam (Adaptive Moment Estimation)

"Combines **Momentum + RMSProp** → Fast + Stable → Most popular optimizer."

---

## 🚫 15. MSE – Not Robust to Outliers

**MSE (Mean Squared Error)** squares the error:
→ **Big outliers become very big** (10 → 100)
→ Affected heavily by extreme values

---

## ✅ 16. MAE – Robust to Outliers

**MAE (Mean Absolute Error)** takes absolute value:
→ Outliers affect linearly
→ **More robust** than MSE

---

## 🛡️ 17. Robust (ML Context)

**Definition**: A model or method that performs well even **in the presence of noise or outliers**.

---

## ➕ 18. Subgradient

**Definition**: A **generalization of gradient** for functions that are **not differentiable** (e.g., L1 norm).

Example:

* For $f(w) = |w|$, gradient doesn't exist at $w = 0$
* But **subgradient** = any value in $[-1, 1]$

---

## 📏 19. Not Scalable

**Meaning**: A method that does **not handle large datasets or complexity well**.

Example: Simple algorithms may be fast on small data but **too slow on large datasets**.

---

## 🎯 20. Bias–Variance Tradeoff (Overfitting Context)

| Term              | Meaning                                                 |
| ----------------- | ------------------------------------------------------- |
| **Low Bias**      | Model fits training data well                           |
| **High Variance** | Model performs poorly on test data (sensitive to noise) |

🧠 Overfitting = Low Bias + High Variance

---

## 📊 21. Evaluation Metrics

### For **Regression**:

| Metric   | Meaning                                    |
| -------- | ------------------------------------------ |
| **MSE**  | Mean Squared Error (sensitive to outliers) |
| **MAE**  | Mean Absolute Error (robust)               |
| **RMSE** | Root MSE (same unit as output)             |
| **R²**   | Explained variance score                   |

### For **Classification**:

| Metric               | Meaning                              |
| -------------------- | ------------------------------------ |
| **Accuracy**         | % of correct predictions             |
| **Precision**        | TP / (TP + FP)                       |
| **Recall**           | TP / (TP + FN)                       |
| **F1 Score**         | Balance between precision and recall |
| **ROC-AUC**          | Area under ROC curve                 |
| **Confusion Matrix** | Table of TP, FP, FN, TN              |

---

# ✅ Summary Diagram Suggestion (Optional)

* Bias vs Variance
* Convex vs Non-Convex Curve
* MSE vs MAE (Outlier impact)

---
Yeh rahe **high-quality structured notes in complete `.md` format**, jisme aaj tumne jitne bhi topics padhe — sab included hain, with clean headings, formulas, examples, and pros/cons.

---

# 📊 Linear Regression – Detailed Notes (GATE + DS Ready)

---

## 1. 📌 Ordinary Least Squares (OLS)

**Definition**:  
OLS (Ordinary Least Squares) ek statistical method hai jo best-fitting regression line find karta hai by minimizing the **sum of squared errors** between actual values and predicted values.

### 🔹 OLS Formula:

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
\]

\[
\hat{y} = \theta_0 + \theta_1 x
\]

### 🔹 OLS Closed-form Solution:

\[
\theta = (X^T X)^{-1} X^T y
\]

> Ye approach fast hai but sirf small datasets ke liye practical hai (due to matrix inversion).

---

## 2. 🔧 Hyperparameter Tuning in Linear Regression

Basic Linear Regression has few hyperparameters, but in **regularized models**, tuning becomes important.

### 🔸 Common Hyperparameters:

| Parameter         | Description                             |
|-------------------|-----------------------------------------|
| `fit_intercept`   | Whether to include bias term            |
| `normalize`       | Normalize input features                |
| `alpha`           | Regularization strength (Ridge, Lasso)  |

### 🔹 Regularized Models:

- **Ridge Regression** (L2): Penalizes large weights
- **Lasso Regression** (L1): Can eliminate useless features (feature selection)

### 🛠️ Tuning Tools:

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.1, 1, 10]}
model = Ridge()
grid = GridSearchCV(model, params, cv=5)
grid.fit(X, y)
````

---

## 3. 🚀 Gradient Descent

**Definition**:
Gradient Descent ek optimization algorithm hai jo cost function ko minimize karta hai by **iteratively updating weights** in the direction of steepest descent.

### 🔹 Update Rule:

$$
\theta = \theta - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta}
$$

Where:

* $\alpha$: Learning rate
* $\frac{\partial J}{\partial \theta}$: Gradient

### 🧠 Types of Gradient Descent:

| Type          | Description           | Pros             | Cons              |
| ------------- | --------------------- | ---------------- | ----------------- |
| Batch GD      | Full dataset per step | Stable updates   | Slow for big data |
| Stochastic GD | One sample per step   | Fast, low memory | High variance     |
| Mini-batch GD | Small batch per step  | Balance of both  | Needs tuning      |

### 📌 Learning Rate Tips:

* Too small: Slow convergence
* Too large: Divergence / oscillation

---

## 4. ❌ `load_boston` is Deprecated (Scikit-learn)

### ⚠️ Why?

`load_boston` was **removed in v1.2** due to **ethical issues** (e.g., racial feature `'B'`).

### ✅ Use Instead:

#### ✔️ `fetch_california_housing()`

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = data.data
y = data.target
```

#### ✔️ Boston Dataset via CSV (if still needed)

```python
import pandas as pd
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
X = df.drop('medv', axis=1)
y = df['medv']
```

---

## 5. 📏 RMSE – Root Mean Squared Error

### ✅ Formula:

$$
RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

### 🔹 Advantages:

* Sensitive to large errors (helps prevent big mistakes)
* Same units as target (easy to interpret)
* Smooth and differentiable (good for optimization)
* Common metric for regression models

### 🔻 Disadvantages:

* Too sensitive to outliers
* Hard to compare across different datasets (unit dependent)
* Doesn’t show error direction (positive or negative)
* Can overemphasize rare large errors

---

## 🧠 Flashcards (Quick Review)

**Q: What is the goal of OLS?**
A: To minimize the sum of squared differences between actual and predicted values.

**Q: What does the alpha parameter do in Ridge/Lasso?**
A: Controls the strength of regularization.

**Q: Why is `load_boston()` not available anymore?**
A: Ethical concerns related to racial data in the dataset.

**Q: What does RMSE tell us?**
A: The average magnitude of prediction error in the same unit as the output.

**Q: Which type of gradient descent is most balanced?**
A: Mini-batch gradient descent.


