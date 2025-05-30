# 🔄 Function Transformation in Machine Learning

Function Transformation is a preprocessing technique used to **modify features** (input variables) to make them more suitable for **machine learning algorithms**.

---

## 🤔 Why Use Function Transformations?

Real-world data often contains:

* Skewed distributions
* Outliers
* Non-linear relationships

Function transformations help to:

* **Normalize distributions**
* **Stabilize variance**
* **Linearize relationships**
* Improve model performance and training stability

---

## ✅ Common Function Transformations

| Transformation          | Formula                                                | Use Case                                                     |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| **Log Transform**       | `log(x)`                                               | Positive skew data ko normalize karta hai                    |
| **Square Root**         | `sqrt(x)`                                              | Variance kam karta hai, large values ka effect kam karta hai |
| **Reciprocal**          | `1/x`                                                  | Very large values ko compress karta hai                      |
| **Square / Polynomial** | `x²`, `x³`                                             | Non-linear relationships ko model karta hai                  |
| **Exponential**         | `e^x`                                                  | Kabhi-kabhi increasing trends ke liye use hota hai           |
| **Box-Cox**             | Automated power transform (only for positive data)     |                                                              |
| **Yeo-Johnson**         | Same as Box-Cox but works with **negative** values too |                                                              |

---

## 🧠 Sklearn Implementation

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

log_transformer = FunctionTransformer(np.log1p)  # log(x + 1)
X_transformed = log_transformer.fit_transform(X)
```

---

## 📊 Visualization Effect

* Skewed feature ko log/sqrt transform karne se wo **normal distribution** ke kareeb aata hai.
* Especially helpful for **Linear Regression**, **SVM**, etc. jinko normality pasand hai.

---

## ⚠️ Cautions

* `log`, `sqrt`, `reciprocal` work only on **positive** values.
* Always perform **EDA** (Exploratory Data Analysis) pehle.
* Transformation sab features pe na lagayein, **experiment** karein and use cross-validation to check effectiveness.

---
## ✅ Common Function Transformations

### 1. **Log Transformation**

* **Formula:** $x' = \log(x + 1)$
* **Use:** Normalize right-skewed (positively skewed) data.
* **Limitation:** Only for positive values.
* **Example:** Salary, population, income.

### 2. **Square Root Transformation**

* **Formula:** $x' = \sqrt{x}$
* **Use:** Reduce impact of large values, normalize skewed data.
* **Limitation:** Works on non-negative values.

### 3. **Reciprocal Transformation**

* **Formula:** $x' = \frac{1}{x}$
* **Use:** Strongly reduces the effect of large values.
* **Limitation:** Only for non-zero, positive values.

### 4. **Power / Polynomial Transformation**

* **Formula:** $x' = x^n$ where $n > 1$
* **Use:** For modeling non-linear relationships.
* **Examples:** $x^2, x^3$

### 5. **Exponential Transformation**

* **Formula:** $x' = e^x$
* **Use:** For rapidly increasing trends.
* **Limitation:** Can exaggerate outliers.

### 6. **Box-Cox Transformation**

* **Definition:** Power transformation that normalizes data.
* **Formula:** $x' = \frac{x^\lambda - 1}{\lambda}$ if $\lambda \neq 0$, else $\log(x)$
* **Use:** Works well to stabilize variance and normalize.
* **Limitation:** Only for positive values.
* **Applied using:** `scipy.stats.boxcox`

### 7. **Yeo-Johnson Transformation**

* **Definition:** Generalization of Box-Cox that allows negative values.
* **Use:** Similar to Box-Cox but supports zero and negative values.
* **Applied using:** `sklearn.preprocessing.PowerTransformer(method='yeo-johnson')`

---

## 🧠 Sklearn Implementation Examples

```python
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
import numpy as np

# Log Transformation
log_transformer = FunctionTransformer(np.log1p)
X_log = log_transformer.fit_transform(X)

# Square Root Transformation
sqrt_transformer = FunctionTransformer(np.sqrt)
X_sqrt = sqrt_transformer.fit_transform(X)

# Reciprocal Transformation
reciprocal_transformer = FunctionTransformer(lambda x: 1 / x)
X_reciprocal = reciprocal_transformer.fit_transform(X)

# Exponential Transformation
exp_transformer = FunctionTransformer(np.exp)
X_exp = exp_transformer.fit_transform(X)

# Box-Cox (only for positive values)
from scipy.stats import boxcox
X_boxcox, _ = boxcox(X[X > 0])  # must handle positives only

# Yeo-Johnson (sklearn way)
power_transformer = PowerTransformer(method='yeo-johnson')
X_yeojohnson = power_transformer.fit_transform(X)
```

---

## 📊 Visualization Effect

* Skewed feature ko log/sqrt transform karne se wo **normal distribution** ke kareeb aata hai.
* Especially helpful for **Linear Regression**, **SVM**, etc. jinko normality pasand hai.

---

## ⚠️ Cautions

* `log`, `sqrt`, `reciprocal` work only on **positive** values.
* Always perform **EDA** (Exploratory Data Analysis) pehle.
* Transformation sab features pe na lagayein, **experiment** karein and use cross-validation to check effectiveness.

---

## ✅ Summary

* Function transformation improves data quality and model performance.
* Common ones: `log`, `sqrt`, `reciprocal`, `Box-Cox`, `Yeo-Johnson`
* Use `FunctionTransformer` from `sklearn`.
* Always visualize feature distribution before and after.

---
