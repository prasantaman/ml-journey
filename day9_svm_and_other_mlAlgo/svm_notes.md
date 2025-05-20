
# 🧠 Support Vector Machine (SVM) - Complete Notes

## 🔷 What is SVM?
**Support Vector Machine (SVM)** is a supervised learning algorithm used for:
- Classification
- Regression
- Outlier detection

Main goal: Find the optimal **hyperplane** that maximally separates different classes.

---

## 🔷 Real-Life Analogy
Imagine drawing a straight line to separate red and blue dots. You want the line to be **as far as possible** from both colors. That’s called maximizing the margin.

---

## 🔶 Why Use SVM?
- Works well in high-dimensional spaces
- Effective when margin between classes exists
- Robust to overfitting in low-noise data

---

## 🔷 How SVM Works
1. Find the best decision boundary (hyperplane)
2. Identify **support vectors**
3. Maximize margin between support vectors and hyperplane
4. Predict based on which side of hyperplane new data lies

---

## 🔷 Types of SVM

| Type | Description |
|------|-------------|
| Linear SVM | Separates data using a linear hyperplane |
| Non-Linear SVM | Uses kernel trick for curved boundaries |
| Classification | Predicts class label |
| Regression (SVR) | Predicts continuous value |
| One-Class SVM | Detects outliers or novelty |

---

## 🔷 Mathematical Understanding

### Linear SVM:
Find hyperplane:  
\[ w \cdot x + b = 0 \]

Condition:  
\[ y_i(w \cdot x_i + b) \geq 1 \]

Objective:  
\[ \min \frac{1}{2}\|w\|^2 \]

### Soft Margin (Non-linear):
\[ \min \frac{1}{2}\|w\|^2 + C \sum \xi_i \]

Subject to:  
\[ y_i(w \cdot x_i + b) \geq 1 - \xi_i \]

---

## 🔷 Kernel Trick

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | \( x^T y \) | Linearly separable data |
| Polynomial | \( (x^T y + c)^d \) | Curved boundaries |
| RBF | \( e^{-\gamma \|x - y\|^2} \) | Default for non-linearity |
| Sigmoid | \( \tanh(kx^T y + c) \) | Neural net-like behavior |

---

## 🔷 Support Vectors
- Data points closest to the hyperplane
- Define the margin
- Model depends heavily on these points

---

## 🔷 Hyperparameters

| Parameter | Description |
|-----------|-------------|
| C | Regularization parameter |
| kernel | Type of decision boundary |
| gamma | Influence of single data point |
| degree | Used in polynomial kernel |

---

## 🔷 Advantages
✅ Works in high-dimensions  
✅ Effective with clear margin separation  
✅ Uses only support vectors  
✅ Solid theoretical foundation

---

## 🔷 Disadvantages
❌ Slow training on large data  
❌ Sensitive to noise  
❌ Requires kernel tuning  
❌ Hard to interpret

---

## 🔷 Applications

| Domain | Example |
|--------|---------|
| Healthcare | Cancer diagnosis |
| Text Mining | Spam detection |
| Image | Face recognition |
| Finance | Fraud detection |
| Agriculture | Disease classification |

---

## 🔷 SVM in Python (Scikit-learn)

### Classification
```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Regression
```python
from sklearn.svm import SVR

model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train, y_train)
```

---

## 🔷 Evaluation Metrics

| Task | Metrics |
|------|---------|
| Classification | Accuracy, F1, Confusion Matrix |
| Regression | MSE, MAE, R² |

---

## 🔷 SVM vs Others

| Model | Training | Prediction | Non-linear | Memory |
|-------|----------|------------|------------|--------|
| SVM | Slow | Fast | Yes | Low |
| KNN | Fast | Slow | No | High |
| Logistic | Medium | Fast | No | Low |

---

## 🔷 Tips for Using SVM
- Always normalize/scale data
- Use `GridSearchCV` to find best C, gamma
- Try linear kernel first, then RBF
- Not suitable for large datasets without optimization

---

## 🔷 Flashcards

**Q: What does SVM aim to do?**  
A: Maximize the margin between classes.

**Q: What is a support vector?**  
A: Points closest to the separating hyperplane.

**Q: What is the kernel trick?**  
A: Mapping data to higher dimensions to enable linear separation.

**Q: What is parameter C in SVM?**  
A: Penalty for misclassification; balances bias and variance.

**Q: Which kernel is used by default?**  
A: RBF (Radial Basis Function)

---

## ✅ Summary

| Feature | Value |
|---------|-------|
| Type | Supervised |
| Tasks | Classification, Regression |
| Main Idea | Maximize margin |
| Algorithm | Support Vector Classifier |
| Libraries | Scikit-learn, LIBSVM |
| Scaling Required | Yes |
| High-dimensional Support | Yes |
