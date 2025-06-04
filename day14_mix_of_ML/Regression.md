## Regression in Machine Learning

### 🔹 What is Regression?

Regression is a **supervised learning** technique used to **predict continuous numerical values**.

**Examples:**

* Predict house price
* Predict temperature
* Predict salary based on experience

---

### 🔹 Types of Regression

#### 1. **Linear Regression**

* Assumes linear relationship between input (X) and output (Y).
* Equation:

  $$
  y = mx + c
  $$

  where `m = slope`, `c = intercept`

#### 2. **Multiple Linear Regression**

* More than one input feature:

  $$
  y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n
  $$

#### 3. **Polynomial Regression**

* Input features are raised to higher powers:

  $$
  y = a_0 + a_1x + a_2x^2 + ... + a_nx^n
  $$

#### 4. **Ridge Regression**

* Linear regression + **L2 regularization**
* Prevents overfitting by adding penalty:

  $$
  \text{Loss} = MSE + \lambda \sum w^2
  $$

#### 5. **Lasso Regression**

* Linear regression + **L1 regularization**
* Encourages sparsity (some weights become 0):

  $$
  \text{Loss} = MSE + \lambda \sum |w|
  $$

#### 6. **ElasticNet Regression**

* Combination of Ridge + Lasso:

  $$
  \text{Loss} = MSE + \lambda_1 \sum |w| + \lambda_2 \sum w^2
  $$

#### 7. **Logistic Regression** *(used for classification)*

* Used for **binary/multiclass classification**, despite the name.
* Output is between 0 and 1 using sigmoid function.

---

### 🔹 Important Terms

| Term                      | Meaning                                                               |                       |   |
| ------------------------- | --------------------------------------------------------------------- | --------------------- | - |
| MSE (Mean Squared Error)  | Average of squared errors: $\frac{1}{n} \sum (y_{true} - y_{pred})^2$ |                       |   |
| MAE (Mean Absolute Error) | Average of absolute errors: (\frac{1}{n} \sum                         | y\_{true} - y\_{pred} | ) |
| R² Score (R-squared)      | Accuracy of regression (closer to 1 is better)                        |                       |   |

---

### 🔹 Real Life Applications

* Predicting **stock prices**
* Estimating **crop yields**
* Predicting **medical costs**
* Sales forecasting

---

### 🔹 How to Train a Regression Model in Python (Sklearn)

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 📌 Q\&A Flashcards

**Q1: What is Regression in ML?**
A: A supervised learning method to predict continuous values.

**Q2: Which algorithm is used for linear relation prediction?**
A: Linear Regression.

**Q3: Which regression uses both L1 and L2 penalties?**
A: ElasticNet Regression.

**Q4: What is MSE?**
A: Mean Squared Error — average of squared prediction errors.

**Q5: Can Logistic Regression be used for continuous prediction?**
A: No, it's for classification tasks.
