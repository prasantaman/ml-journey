## 📘 Probability in Machine Learning & Statistics

### 🔹 What is Probability?

Probability is the **measure of the likelihood** that an event will occur.

* It ranges from 0 (impossible) to 1 (certain).
* Formula:

  $$
  P(E) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}}
  $$

---

### 🔹 Types of Probability

#### 1. **Theoretical Probability**

* Based on reasoning or formulas.
* Example: Probability of rolling a 3 on a fair die = 1/6

#### 2. **Experimental Probability**

* Based on actual experiments or observations.
* Formula:

  $$
  P(E) = \frac{\text{Number of times E occurs}}{\text{Total number of trials}}
  $$

#### 3. **Axiomatic Probability**

* Uses axioms or rules:

  * $0 \leq P(E) \leq 1$
  * $P(S) = 1$ (S = Sample Space)
  * For mutually exclusive events A and B:

    $$
    P(A \cup B) = P(A) + P(B)
    $$

---

### 🔹 Basic Terms

| Term               | Meaning                             |
| ------------------ | ----------------------------------- |
| Experiment         | An action with uncertain outcome    |
| Sample Space (S)   | All possible outcomes               |
| Event (E)          | One or more outcomes from S         |
| Mutually Exclusive | Events that cannot happen together  |
| Independent Events | Events that don’t affect each other |

---

### 🔹 Conditional Probability

* Probability of an event A given B has occurred:

  $$
  P(A|B) = \frac{P(A \cap B)}{P(B)}
  $$
* Example: Probability of passing math **given** you studied.

---

### 🔹 Bayes' Theorem

* Used for **reverse probability**:

  $$
  P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
  $$
* Used in **Naive Bayes Classifier** (ML)

---

### 🔹 Rules of Probability

1. **Addition Rule**:

   $$
   P(A \cup B) = P(A) + P(B) - P(A \cap B)
   $$
2. **Multiplication Rule**:

   $$
   P(A \cap B) = P(A) \cdot P(B|A)
   \] (for dependent events)
   \[
   P(A \cap B) = P(A) \cdot P(B)
   \] (for independent events)
   $$

---

### 🔹 Distributions Related to Probability

#### 1. **Uniform Distribution**

* All outcomes are equally likely.

#### 2. **Bernoulli Distribution**

* Two outcomes: success (1) and failure (0).
* Example: Tossing a coin.

#### 3. **Binomial Distribution**

* Repeated Bernoulli trials.
* Probability of getting x successes in n trials.
* Formula:

  $$
  P(X = x) = {n \choose x} p^x (1-p)^{n-x}
  $$

#### 4. **Normal Distribution (Gaussian)**

* Bell-shaped curve.
* Many real-world values follow this.
* Mean = Median = Mode

#### 5. **Poisson Distribution**

* For counting number of events in fixed interval.
* Example: Number of calls at a call center per hour.

---

### 🔹 Real Life Applications in ML

* Spam detection (Naive Bayes)
* Fraud detection
* Predictive analytics
* Language models (probabilistic NLP)

---

## 📌 Q\&A Flashcards

**Q1: What is the probability of an impossible event?**
A: 0

**Q2: What is the formula for conditional probability?**
A: $P(A|B) = \frac{P(A \cap B)}{P(B)}$

**Q3: What is Bayes' Theorem used for?**
A: Calculating reverse probability (P(A|B) from P(B|A))

**Q4: What is the shape of normal distribution?**
A: Bell-shaped

**Q5: Which distribution is used for binary outcomes?**
A: Bernoulli Distribution
