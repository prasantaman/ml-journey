# Decision Tree: Complete Notes with DTC, DTR, Math & Applications

## 📌 What is a Decision Tree?

A **Decision Tree** is a supervised machine learning algorithm used for **classification and regression tasks**. It splits the data into branches based on feature values until a decision or prediction is made.

---

## 🌳 Types of Decision Trees

| Type                   | Description                               |
| ---------------------- | ----------------------------------------- |
| DecisionTreeClassifier | Used for classification (discrete output) |
| DecisionTreeRegressor  | Used for regression (continuous output)   |

---

## 📊 Components of a Decision Tree

* **Root Node**: Top node that represents the entire dataset.
* **Decision Nodes**: Intermediate nodes where data is split.
* **Leaf Nodes**: Terminal nodes where a final class/label/value is assigned.
* **Branches**: Arrows showing flow from one node to another.

---

## 📐 Mathematical Concepts

### 1. **Entropy (for Classification)**

Measures impurity or disorder:

$Entropy(S) = -\sum_{i=1}^{n} p_i \log_2(p_i)$

Where:

* $p_i$ is the probability of class $i$
* Lower entropy = purer data

### 2. **Information Gain**

Measures the decrease in entropy after a split:

$IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)$

### 3. **Gini Index (for Classification)**

An alternative to entropy:

$Gini(S) = 1 - \sum_{i=1}^{n} p_i^2$

* Lower Gini = purer node

### 4. **MSE (for Regression)**

Used to measure split quality in regression:

$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$

Where:

* $y_i$ = actual value
* $\bar{y}$ = mean of values in node

---

## ⚙️ Algorithms Used

* **ID3**: Uses entropy and information gain
* **C4.5**: Improvement over ID3 (handles missing values, pruning)
* **CART**: Classification and Regression Trees (uses Gini or MSE)

---

## 🧠 Scikit-learn Implementation

### DecisionTreeClassifier

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(X_train, y_train)
```

### DecisionTreeRegressor

```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
model.fit(X_train, y_train)
```

---

## 📌 Important Hyperparameters

| Parameter           | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| criterion           | 'gini', 'entropy' for classification; 'mse' for regression |
| max\_depth          | Maximum depth of tree                                      |
| min\_samples\_split | Minimum samples required to split                          |
| min\_samples\_leaf  | Minimum samples in leaf node                               |
| max\_features       | Number of features to consider at each split               |

---

## ✅ Advantages

* Easy to understand & visualize
* Handles both numerical & categorical data
* No need for feature scaling

## ❌ Disadvantages

* Prone to overfitting
* Small changes in data can change tree structure

---

## 🧠 Real-life Applications

| Field       | Use Case                       |
| ----------- | ------------------------------ |
| Finance     | Loan approval, credit scoring  |
| Healthcare  | Disease diagnosis              |
| Marketing   | Customer segmentation          |
| Agriculture | Crop yield prediction          |
| Education   | Predicting student performance |

---

## 🔚 Summary Table

| Concept          | Used In               | Formula/Metric                     |
| ---------------- | --------------------- | ---------------------------------- |
| Entropy          | Classification        | $-\sum p \log_2 p$                 |
| Information Gain | Classification        | Parent Entropy - Child             |
| Gini Index       | Classification (CART) | $1 - \sum p^2$                     |
| MSE              | Regression            | $\frac{1}{n} \sum (y - \bar{y})^2$ |

---

## 🧾 DecisionTreeClassifier vs DecisionTreeRegressor

### 🔷 DecisionTreeClassifier (DTC)

* **Task**: Classification (categorical output)
* **Splitting Metric**: Gini or Entropy
* **Example Output**: 'Spam' or 'Not Spam'

### 🔶 DecisionTreeRegressor (DTR)

* **Task**: Regression (continuous output)
* **Splitting Metric**: MSE, MAE
* **Example Output**: Predict house price = ₹12.5 Lakhs

---

## 📘 References

* Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
* Quinlan, J. R. (1986). Induction of decision trees
* CART: Breiman et al., 1984

Let me know if you want a visual diagram or dataset example!
