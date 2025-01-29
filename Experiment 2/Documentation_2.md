# Aim: Using Numpy, train a Neural Network to classify linearly separable dataset (without any hidden layer).

## Overview
This project implements **logistic regression** for binary classification. The code first applies the model to a **linearly separable dataset** and then attempts the same approach on a **non-linearly separable dataset**. Hidden Layers and Activation Functions are then added to improve and generate accurate results.


## 1. Code Breakdown

### 1.1 Importing Dependencies
```python
import numpy as np
import matplotlib.pyplot as plt
```
- **NumPy**: Used for matrix operations and numerical computations.
- **Matplotlib**: Used for visualizing the datasets and decision boundaries.

### 1.2 Generating a Linearly Separable Dataset
```python
np.random.seed(0)
num_samples = 100  # Number of points per class
X1 = np.random.randn(num_samples, 2) + np.array([2, 2])
X2 = np.random.randn(num_samples, 2) + np.array([-2, -2])
X = np.vstack((X1, X2))
y = np.hstack((np.ones(num_samples), np.zeros(num_samples)))
```
- Two Gaussian-distributed clusters of points are generated.
- Class **1** points are centered at **(2,2)**, and class **0** points are centered at **(-2,-2)**.
- **X** contains feature points, while **y** holds corresponding labels.

### 1.3 Visualizing the Dataset
```python
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.title("Linearly Separable Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
- A scatter plot is generated, using colors to differentiate classes.

### 1.4 Initializing Model Parameters
```python
W = np.random.randn(2)  # 2D weight vector
b = np.random.randn()   # Bias term
learning_rate = 0.01
epochs = 1000  # Number of training iterations
```
- The weights **W** and bias **b** are initialized randomly.
- A fixed learning rate and training epochs are defined.

### 1.5 Sigmoid Activation Function
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
- Converts input values into probabilities between 0 and 1.

### 1.6 Training the Logistic Regression Model
```python
for epoch in range(epochs):
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    dz = y_pred - y
    dW = np.dot(X.T, dz) / len(y)
    db = np.mean(dz)
    W -= learning_rate * dW
    b -= learning_rate * db
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```
- **Forward Propagation**:
  - Computes weighted sum \(z = WX + b\).
  - Applies the **sigmoid function** to obtain predictions.
- **Loss Calculation**:
  - Uses **binary cross-entropy loss**.
- **Backpropagation & Parameter Update**:
  - Computes gradients **dW** and **db**.
  - Updates parameters using gradient descent.
- **Training Progress**:
  - Prints loss every 100 epochs.

### 1.7 Plotting Decision Boundary
```python
x_values = np.linspace(-4, 4, 100)
y_values = -(W[0] * x_values + b) / W[1]
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.plot(x_values, y_values, 'k-', linewidth=2)
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
- The learned **decision boundary** is plotted along with the dataset.

### 1.8 Making Predictions
```python
def predict(X_new):
    z = np.dot(X_new, W) + b
    return (sigmoid(z) >= 0.5).astype(int)
```
- Converts outputs into **binary class labels** (0 or 1).

### 1.9 Applying Model to a Non-Linearly Separable Dataset
```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.2)
```
- Generates a **two-moons dataset**, which is **not linearly separable**.
- The previous logistic regression approach is then applied but **fails to classify correctly**.

---

## 2. Limitation: Failure on Non-Linearly Separable Data
### Why does it fail?
- Logistic regression can only learn **linear decision boundaries**.
- The **two moons dataset** requires a curved boundary, which logistic regression **cannot** capture.

### Observations on Two Moons Dataset:
- The model fails to classify data correctly.
- Loss plateaus early, indicating it cannot optimize well.
- The decision boundary remains **linear**, causing misclassifications.

---

## 3. Adding a Hidden Layer and ReLU Activation Function

### 1. Dataset Generation & Normalization
- Generates a **two-moons dataset** using `make_moons()`.
- Normalizes the input features for efficient training.

### 2. Neural Network Architecture
- **Input Layer**: 2 features (since the dataset is 2D).
- **Hidden Layer**: 10 neurons with **ReLU activation**.
- **Output Layer**: 1 neuron with **Sigmoid activation** (for binary classification).

### 3. Weight Initialization
- Uses **He initialization** for better convergence.

### 4. Forward Propagation
- Computes activations for the hidden layer using **ReLU**.
- Computes the final probability using **Sigmoid**.

### 5. Loss Calculation
- Uses **Binary Cross-Entropy Loss**.

### 6. Backpropagation & Parameter Updates
- Computes gradients using the chain rule.
- Updates weights using **Gradient Descent**.

### 7. Training & Loss Monitoring
- Runs for `10,000` epochs with a learning rate of `0.01`.
- Loss is printed every `500` epochs.


---

## Conclusion
- Logistic regression is effective for **linearly separable datasets** but fails on **non-linear problems**.
- **Enhancements like addition of Hidden Layer and Activation Functions** significantly improve performance.

