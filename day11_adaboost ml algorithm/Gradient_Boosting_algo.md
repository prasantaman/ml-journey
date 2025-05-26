# 🌿 Gradient Boosting – Machine Learning Notes

---

## 📌 Introduction

**Gradient Boosting** is an ensemble machine learning technique that builds a **strong predictor** from multiple **weak learners**, typically decision trees. It optimizes model performance by minimizing a **loss function** using **gradient descent**.

> Used for both **regression** and **classification** tasks.

---

## 🚀 Intuition

* Boosting focuses on **errors** made by previous models.
* Instead of adjusting sample weights (like AdaBoost), Gradient Boosting fits the new model to the **residual errors** (gradients).
* Combines predictions by **adding them up** instead of voting.

---

## ⚙️ Working Mechanism (Step-by-Step)

Given training data:
$D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$

### Step 1: Initialize the Model

* Start with a **simple model**:
  $F_0(x) = \arg\min_\gamma \sum L(y_i, \gamma)$
* For regression: usually the **mean** of $y$

### Step 2: For $m = 1$ to $M$:

1. Compute **residuals** (negative gradients):
   $r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}}$

2. Train a **weak learner** $h_m(x)$ to predict residuals $r_{im}$

3. Compute **step size** (line search):
   $\gamma_m = \arg\min_\gamma \sum L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$

4. Update the model:
   $F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$

### Final Model:

$F_M(x) = F_0(x) + \sum_{m=1}^M \gamma_m h_m(x)$

---

## 🔊 Key Concepts

* **Loss Function**: Guides the optimization process (e.g., MSE for regression, Log Loss for classification).
* **Negative Gradient**: Direction of steepest descent.
* **Additive Model**: Boosted models are additive in nature.
* **Learning Rate ($\eta$)**: Shrinks contribution of each tree. Helps avoid overfitting.

---

## 📊 Common Loss Functions

| Task                  | Loss Function                               |
| --------------------- | ------------------------------------------- |
| Regression            | Mean Squared Error (MSE) $(y - F(x))^2$     |
| Binary Classification | Log Loss $-y \log(p) - (1 - y) \log(1 - p)$ |

---

## 🎓 Gradient Boosting vs AdaBoost

| Feature        | Gradient Boosting       | AdaBoost                      |
| -------------- | ----------------------- | ----------------------------- |
| Error handling | Fits to residuals       | Adjusts sample weights        |
| Optimization   | Gradient Descent        | Exponential Loss Minimization |
| Performance    | More accurate           | Faster to train               |
| Flexibility    | Any differentiable loss | Exponential loss only         |

---

## 📈 Advantages

* High predictive accuracy.
* Can handle different types of data and loss functions.
* Feature importance available.
* Works well with default hyperparameters.

---

## ❌ Disadvantages

* Slower to train (sequential).
* Sensitive to overfitting (mitigated by learning rate and early stopping).
* Requires careful tuning.

---

## 💡 Regularization Techniques

* **Learning Rate** ($\eta$): Reduces step size.
* **Tree Depth**: Limits complexity of each weak learner.
* **Subsampling**: Uses a random subset of data for training each tree (like bagging).
* **Number of Estimators**: More trees = better fit but risk of overfitting.

---

## 📆 Implementation (Scikit-learn)

```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8
)
model.fit(X_train, y_train)
```

---

## 📊 Applications

| Domain        | Use Case                               |
| ------------- | -------------------------------------- |
| 💼 Business   | Sales prediction, customer churn       |
| 🏥 Healthcare | Disease diagnosis, survival prediction |
| 💰 Finance    | Credit risk scoring, fraud detection   |
| 📱 Apps       | Ranking, personalization               |

---

## 📃 Popular Libraries

| Library    | Notes                                                      |
| ---------- | ---------------------------------------------------------- |
| `sklearn`  | Built-in GradientBoostingClassifier & Regressor            |
| `XGBoost`  | eXtreme Gradient Boosting (fast + regularized)             |
| `LightGBM` | Faster with large datasets (uses histogram-based learning) |
| `CatBoost` | Best for categorical features                              |

---

## 🔮 Real-Life Example (Sklearn)

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 📒 Flashcards (Important Concepts)

**Q: What is the main idea of Gradient Boosting?**
A: Add models sequentially to correct residual errors of previous models.

**Q: What type of loss functions can be used?**
A: Any differentiable loss (MSE, Log Loss, etc.).

**Q: Which direction does Gradient Boosting move?**
A: In the direction of the negative gradient (steepest descent).

**Q: What is the role of learning rate?**
A: Controls step size; smaller values reduce overfitting.

**Q: Most popular libraries for GB?**
A: XGBoost, LightGBM, CatBoost, scikit-learn.

---

**End of Notes**
