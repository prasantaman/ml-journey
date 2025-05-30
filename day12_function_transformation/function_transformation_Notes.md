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

## ✅ Summary

* Function transformation improves data quality and model performance.
* Common ones: `log`, `sqrt`, `reciprocal`, `Box-Cox`, `Yeo-Johnson`
* Use `FunctionTransformer` from `sklearn`.
* Always visualize feature distribution before and after.

---
