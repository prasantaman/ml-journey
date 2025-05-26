# 🧠 AdaBoost (Adaptive Boosting) – Machine Learning Notes

---

## 📌 Introduction

**AdaBoost** (Adaptive Boosting) is an **ensemble learning** technique that builds a **strong classifier** from a number of **weak classifiers** (usually decision stumps). It was the first practical boosting algorithm and is primarily used for **binary classification**.

---

## 🚀 Intuition

* Boosting focuses on difficult examples.
* Each model learns from the **mistakes** of the previous model.
* Final prediction is a **weighted vote** of all weak learners.

---

## ⚙️ Working Mechanism (Step-by-Step)

Given training data:
$D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$
where $y_i \in \{-1, +1\}$

### Step 1: Initialize Sample Weights

$w_i = \frac{1}{n}$

### Step 2: Repeat for $m = 1$ to $M$:

1. Train weak learner $h_m(x)$ using weighted data.
2. Compute weighted error:
   $err_m = \sum w_i \cdot I(h_m(x_i) \neq y_i)$
3. Compute learner weight:
   $\alpha_m = \frac{1}{2} \ln \left(\frac{1 - err_m}{err_m}\right)$
4. Update sample weights:
   $w_i \leftarrow w_i \cdot e^{-\alpha_m y_i h_m(x_i)}$
5. Normalize weights so they sum to 1.

### Step 3: Final Strong Classifier

$H(x) = \text{sign}\left( \sum_{m=1}^M \alpha_m h_m(x) \right)$

---

## 🧠 Mathematical Insight

* Final prediction is a **weighted vote**.
* Each $\alpha_m$ represents model's say in final result.
* AdaBoost minimizes **exponential loss**:
  $\sum \exp(-y_i f(x_i))$

---

## 📊 Weak Learner = Decision Stump

* A decision stump is a 1-level decision tree.
* Fast to train, interpretable.
* Common default weak learner for AdaBoost.

---

## ✅ Advantages

* Simple to implement.
* Less overfitting compared to other algorithms.
* Automatically focuses on hard examples.
* No need for feature scaling.

---

## ❌ Disadvantages

* Sensitive to **noisy data** and **outliers**.
* Not ideal for small datasets (may overfit).
* Sequential nature → slower training for large datasets.

---

## ⚙️ Scikit-learn: `AdaBoostClassifier`

```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(
    base_estimator=None,       # default: Decision Stump
    n_estimators=50,           # number of weak learners
    learning_rate=1.0          # controls alpha_m strength
)
```

---

## 📊 Applications of AdaBoost

| Domain             | Use Case                               |
| ------------------ | -------------------------------------- |
| 🏥 Healthcare      | Disease prediction, cancer detection   |
| 📷 Computer Vision | Face detection (Viola-Jones Algorithm) |
| 💸 Finance         | Fraud detection, credit scoring        |
| 💬 NLP             | Spam detection, text classification    |
| ⚙️ General         | Any binary/multiclass classification   |

---

## 🤖 Real-Life Example (Scikit-learn)

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit AdaBoost
model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 📒 Flashcards (Important Concepts)

**Q: What is the main goal of AdaBoost?**
A: Combine weak learners to form a strong learner.

**Q: Which loss function does Adaboost minimize?**
A: Exponential loss.

**Q: What happens to misclassified sample weights?**
A: Their weights are increased in the next round.

**Q: Which weak learner is commonly used?**
A: Decision Stump (1-level decision tree).

**Q: Is AdaBoost good for noisy data?**
A: No, it's sensitive to noise and outliers.

---

## 📆 Variants of AdaBoost

| Variant         | Description                                              |
| --------------- | -------------------------------------------------------- |
| Real AdaBoost   | Uses probability estimates instead of binary predictions |
| Gentle AdaBoost | Less sensitive to outliers                               |
| LogitBoost      | Optimizes log-loss instead of exponential loss           |
| SAMME / SAMME.R | Multiclass AdaBoost variants (used in Scikit-learn)      |

---

**End of Notes**
