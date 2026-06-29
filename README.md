# Handwritten Digit Classification using Neural Networks

> A beginner-friendly deep learning project using TensorFlow, Keras, and the MNIST dataset to classify handwritten digits (0–9) using Artificial Neural Networks (ANN).

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Key Concepts You Must Know](#key-concepts-you-must-know)
3. [Libraries Used & Why](#libraries-used--why)
4. [The Dataset: MNIST](#the-dataset-mnist)
5. [Project Walkthrough — Step by Step](#project-walkthrough--step-by-step)
   - [Step 1: Import Libraries](#step-1-import-libraries)
   - [Step 2: Load the Dataset](#step-2-load-the-dataset)
   - [Step 3: Explore the Data](#step-3-explore-the-data)
   - [Step 4: Normalize/Scale the Data](#step-4-normalizescale-the-data)
   - [Step 5: Flatten the Images](#step-5-flatten-the-images)
   - [Step 6: Build Neural Network (No Hidden Layer)](#step-6-build-neural-network-no-hidden-layer)
   - [Step 7: Compile the Model](#step-7-compile-the-model)
   - [Step 8: Train the Model](#step-8-train-the-model)
   - [Step 9: Evaluate on Test Data](#step-9-evaluate-on-test-data)
   - [Step 10: Make Predictions](#step-10-make-predictions)
   - [Step 11: Confusion Matrix](#step-11-confusion-matrix)
   - [Step 12: Build Neural Network (With Hidden Layer)](#step-12-build-neural-network-with-hidden-layer)
6. [Model Architecture Diagrams](#model-architecture-diagrams)
7. [Results Summary](#results-summary)
8. [Common Interview Questions & Answers](#common-interview-questions--answers)
9. [Glossary of Key Terms](#glossary-of-key-terms)
10. [How to Run This Project](#how-to-run-this-project)

---

## What This Project Does

This project builds an **Artificial Neural Network (ANN)** from scratch (using Keras) that can look at an image of a handwritten digit and correctly identify which digit (0–9) it is.

Two models are built and compared:
- **Model 1**: No hidden layer (simpler, lower accuracy)
- **Model 2**: With a hidden layer (more complex, better accuracy)

This is the classic **"Hello World" of deep learning** — almost every beginner starts here because the problem is easy to understand but the concepts it teaches apply everywhere.

---

## Key Concepts You Must Know

Before diving into the code, here are the fundamental ideas behind the whole project:

### What is a Neural Network?
A neural network is a machine learning model loosely inspired by the human brain. It consists of **layers of neurons** (mathematical functions) that transform input data step by step until they produce a prediction. Each neuron takes some numbers in, multiplies them by weights, adds a bias, and passes the result through an **activation function**.

### What is Deep Learning?
Deep Learning is a subset of Machine Learning that uses neural networks with **many layers** (hence "deep"). The more layers, the more complex patterns the network can learn.

### What is Classification?
Classification is the task of putting inputs into categories. Here, the categories are the digits 0–9. Given a pixel image, the model must output **which category** the image belongs to.

### What is the difference between Training data and Test data?
- **Training data** is used to teach the model (it learns from this).
- **Test data** is data the model has never seen — used only to check if it actually learned or just memorized.

---

## Libraries Used & Why

| Library | Purpose | Why This Specific Library |
|--------|---------|--------------------------|
| `tensorflow` | The main deep learning framework | Industry standard, backed by Google, runs on GPU/CPU |
| `keras` | High-level API inside TensorFlow | Makes building neural networks simple with just a few lines of code |
| `numpy` | Numerical operations on arrays | Essential for array math — nearly every ML pipeline uses it |
| `matplotlib` | Plotting and visualization | To visually inspect images and understand what we're feeding the model |
| `seaborn` | Statistical data visualization | Makes the confusion matrix look clean and readable |

### Why TensorFlow + Keras and not PyTorch?
Both are excellent. TensorFlow/Keras is often used in production and industry; PyTorch is popular in research. For beginners, Keras is the easiest to learn because its syntax is intuitive and readable.

---

## The Dataset: MNIST

**MNIST** stands for **Modified National Institute of Standards and Technology**.

- It contains **70,000 images** of handwritten digits
- **60,000 training images** + **10,000 test images**
- Each image is **28×28 pixels** (grayscale)
- Each pixel has a value from **0 (black) to 255 (white)**
- Each image has a **label**: the correct digit (0–9)

This dataset is built directly into Keras, so no manual download is needed.

```
Sample image (28x28 pixels):
┌────────────────────────────────┐
│  0  0  0  0  0  0  0  0  0 ...│
│  0  0  0 12 200 255 100 0 ...  │
│  0  0 80 240 255 255  90 0 ... │
│  ...                           │
└────────────────────────────────┘
Label: 5
```

---

## Project Walkthrough — Step by Step

### Step 1: Import Libraries

```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
```

**What's happening:**
- `tensorflow` is the overall deep learning framework.
- `keras` is imported from TensorFlow — it provides the easy-to-use neural network building blocks.
- `matplotlib.pyplot` lets us plot images so we can visually inspect the data.
- `%matplotlib inline` is a Jupyter magic command: it tells Jupyter to display plots directly inside the notebook instead of opening a new window.
- `numpy` is used for array operations (e.g., finding the maximum predicted value).

---

### Step 2: Load the Dataset

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

**What's happening:**
- This one line downloads and loads the MNIST dataset.
- `x_train` → 60,000 images (the input, i.e., the pixel data)
- `y_train` → 60,000 labels (the output, i.e., the correct digit for each image)
- `x_test` → 10,000 images for testing
- `y_test` → 10,000 labels for testing

The `x` variables contain **what the model sees**. The `y` variables contain **what the correct answer is**.

---

### Step 3: Explore the Data

```python
len(x_train)   # → 60000
len(x_test)    # → 10000
len(y_train)   # → 60000
len(y_test)    # → 10000

x_train[0].shape   # → (28, 28)
x_train[0]         # → 2D array of pixel values
plt.matshow(x_train[0])   # Displays the first image visually
y_train[:5]        # → array([5, 0, 4, 1, 9]) — first 5 labels
```

**Why explore?**
Always inspect your data before building a model. You need to understand:
- The **shape** of your data (dimensions)
- The **range of values** (pixel values 0–255)
- What the **labels look like** (integers 0–9)

This prevents surprises later during training.

---

### Step 4: Normalize/Scale the Data

```python
x_train = x_train / 255
x_test = x_test / 255
```

**What's happening:**
Pixel values range from 0 to 255. We divide by 255 to bring them into the range **0 to 1**.

**Why is normalization important?**
- Neural networks learn by adjusting weights using gradients. Large input values make gradients very large or very small, making training unstable.
- Normalized inputs (0 to 1) make the optimizer converge **faster and more reliably**.
- This is one of the most important preprocessing steps in any ML project.

> **Rule of thumb**: Always normalize your inputs before training a neural network.

---

### Step 5: Flatten the Images

```python
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)
```

**What's happening:**
Each image is currently a **2D array of shape (28, 28)**. A standard Dense (fully connected) layer in Keras expects a **1D array** as input. So we "flatten" each 28×28 image into a single row of 784 values.

```
Before:                After:
[[1, 2, 3],            [1, 2, 3, 4, 5, 6, 7, 8, 9]
 [4, 5, 6],
 [7, 8, 9]]
Shape: (28, 28)        Shape: (784,)
```

After reshaping:
- `x_train_flattened.shape` → (60000, 784)
- `x_test_flattened.shape` → (10000, 784)

**Note:** We do NOT flatten `y_train` or `y_test` because they are already 1D arrays (just a list of numbers like 5, 0, 4...).

---

### Step 6: Build Neural Network (No Hidden Layer)

```python
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
```

**What's happening:**
- `keras.Sequential` — Defines the model as a linear stack of layers (one after another).
- `keras.layers.Dense(10, ...)` — A fully connected layer with **10 neurons** (one per digit class: 0–9).
  - Every input neuron (784) is connected to every output neuron (10) — that's what "Dense" means.
  - `input_shape=(784,)` tells the layer that each input has 784 features.
  - `activation='sigmoid'` — The activation function applied to each neuron's output.

**What is Sigmoid activation?**
Sigmoid squishes any value into the range (0, 1), making it useful for outputting probability-like values. For each of the 10 output neurons, sigmoid gives a value between 0 and 1, and we pick the highest one as our prediction.

```
sigmoid(x) = 1 / (1 + e^(-x))
```

**This model has NO hidden layer** — inputs connect directly to outputs. It can only learn simple, linear patterns.

---

### Step 7: Compile the Model

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**What's happening:**
Compiling configures how the model will be trained. Three key things are set:

**1. Optimizer: `adam`**
The optimizer is the algorithm that adjusts the model's weights to minimize the loss. Adam (Adaptive Moment Estimation) is one of the most popular optimizers because it:
- Automatically adjusts the learning rate
- Converges faster than basic gradient descent
- Works well out-of-the-box without much tuning

**2. Loss: `sparse_categorical_crossentropy`**
The loss function measures how wrong the model's predictions are.
- `categorical` → our task is multi-class classification (10 classes)
- `sparse` → the labels (`y_train`) are integers (0, 1, 2...) not one-hot encoded vectors
- Cross-entropy penalizes confident wrong predictions heavily, which is ideal for classification

**3. Metrics: `accuracy`**
During training, TensorFlow will print accuracy after each epoch so we can track how well the model is doing.

---

### Step 8: Train the Model

```python
model.fit(x_train_flattened, y_train, epochs=5)
```

**What's happening:**
This is where learning actually occurs.
- `x_train_flattened` — The input images (flattened)
- `y_train` — The correct labels
- `epochs=5` — The entire training dataset is passed through the model **5 times**

**What is an epoch?**
One epoch = one complete pass through all 60,000 training examples. Multiple epochs allow the model to refine its weights iteratively.

After each epoch, you'll see output like:
```
Epoch 1/5 — loss: 0.7321 — accuracy: 0.8140
Epoch 2/5 — loss: 0.4512 — accuracy: 0.8840
...
```
The loss should decrease and accuracy should increase each epoch.

---

### Step 9: Evaluate on Test Data

```python
model.evaluate(x_test_flattened, y_test)
```

**What's happening:**
This runs the model on data it has **never seen before** (the test set) to get a true measure of performance.

- If test accuracy ≈ training accuracy → the model generalizes well ✅
- If test accuracy << training accuracy → the model is **overfitting** (memorizing instead of learning) ❌

Typical result for the no-hidden-layer model: ~92% accuracy.

---

### Step 10: Make Predictions

```python
plt.matshow(x_test[0])   # Visualize the test image

y_predicted = model.predict(x_test_flattened)
y_predicted[0]           # Array of 10 values (probabilities for each digit)
np.argmax(y_predicted[0])  # Get the index with the highest value → the predicted digit
```

**What's happening:**
- `model.predict()` returns a 2D array: for each image, it gives 10 probability-like values (one per digit).
- `y_predicted[0]` might look like: `[0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.92, 0.01, 0.02]`
- `np.argmax()` returns the index of the **maximum value** — that index IS the predicted digit.
  - In the example above, index 7 has value 0.92 → predicted digit is **7** ✅

To make predictions for all test images:
```python
y_predicted_labels = [np.argmax(i) for i in y_predicted]
```
This is a **list comprehension** — a concise Python loop that builds a list.

---

### Step 11: Confusion Matrix

```python
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

import seaborn as sn
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
```

**What is a Confusion Matrix?**
A confusion matrix is a table that shows the performance of a classification model.

```
              Predicted
              0    1    2    3 ...
         0 [980,   0,   1,   0 ...]   ← Correctly predicted 980 zeros
Truth    1 [  0, 1130,  2,   1 ...]
         2 [  3,   5, 998,   4 ...]
         ...
```

- **Diagonal values** (top-left to bottom-right) = correct predictions ✅
- **Off-diagonal values** = mistakes ❌
- A perfect model has all values on the diagonal and zeros everywhere else.

`fmt='d'` displays integers (not scientific notation). `annot=True` shows the numbers inside each cell.

---

### Step 12: Build Neural Network (With Hidden Layer)

```python
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),  # Hidden layer
    keras.layers.Dense(10, activation='sigmoid')                      # Output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)
```

**What's different?**
A **hidden layer** with 100 neurons is added between the input and output.

**Why does this improve accuracy?**
- Without a hidden layer, the model can only learn **linear** relationships.
- With a hidden layer, the model can learn **non-linear, more complex patterns** in the data.
- The hidden layer acts as a feature extractor — it learns intermediate representations of the data.

**Why `relu` activation in the hidden layer?**
ReLU (Rectified Linear Unit) is defined as: `f(x) = max(0, x)`
- Any negative value becomes 0; positive values pass through unchanged.
- It's faster to compute than sigmoid.
- It avoids the **vanishing gradient problem** (where very deep networks stop learning because gradients become too small).
- ReLU in hidden layers + Sigmoid/Softmax in output layer is a very common pattern.

**Typical result with hidden layer:** ~97% accuracy — a significant improvement over the ~92% without one!

---

## Model Architecture Diagrams

### Model 1: No Hidden Layer
```
Input Layer          Output Layer
(784 neurons)   →   (10 neurons, sigmoid)
    ↓
Each of the 784 inputs connects to all 10 outputs
Total parameters: 784×10 + 10 (biases) = 7,850
```

### Model 2: With Hidden Layer
```
Input Layer      Hidden Layer        Output Layer
(784 neurons) → (100 neurons, relu) → (10 neurons, sigmoid)

Parameters:
  784×100 + 100 = 78,500   (input → hidden)
  100×10  + 10  = 1,010    (hidden → output)
  Total = 79,510
```
More parameters = more capacity to learn complex patterns.

---

## Results Summary

| Model | Hidden Layer | Activation | Epochs | Test Accuracy |
|-------|-------------|------------|--------|---------------|
| Model 1 | None | Sigmoid | 5 | ~92% |
| Model 2 | 100 neurons (ReLU) | ReLU + Sigmoid | 5 | ~97% |

The hidden layer gave ~5% improvement with no extra effort — just adding one line of code.

---

## Common Interview Questions & Answers

### Q1: What is a neural network?
**A:** A neural network is a computational model inspired by the human brain, made of layers of interconnected "neurons" (mathematical functions). Each neuron takes weighted inputs, sums them, adds a bias, and applies an activation function. Layers of neurons transform raw input data into useful predictions.

### Q2: What is the difference between training data and test data?
**A:** Training data is used to fit the model (update weights). Test data is held out completely and only used to evaluate final performance. This simulates how the model would perform on new, unseen data. Using test data during training would lead to overly optimistic results.

### Q3: Why do we normalize data?
**A:** Neural networks learn via gradient descent, which is sensitive to the scale of input values. Unnormalized data (e.g., pixel values 0–255) leads to large and unstable gradients. Normalizing to [0,1] makes training stable, faster, and often leads to better accuracy.

### Q4: What is an epoch?
**A:** One epoch is one complete pass through the entire training dataset. We typically train for multiple epochs, allowing the model to see each sample multiple times and progressively improve its weights.

### Q5: What is the purpose of an activation function?
**A:** Activation functions introduce **non-linearity** into the network. Without them, stacking multiple layers would collapse into a single linear transformation, making hidden layers pointless. Common activation functions: ReLU (hidden layers), Sigmoid (binary output), Softmax (multi-class output).

### Q6: What is the difference between Sigmoid and ReLU?
**A:**
- **Sigmoid:** Output ∈ (0,1). Good for output layers in binary/multi-class classification. Problem: vanishing gradients in deep networks.
- **ReLU:** Output = max(0, x). Fast, computationally simple. Preferred for hidden layers. Avoids vanishing gradient for positive inputs.

### Q7: What is the vanishing gradient problem?
**A:** During backpropagation, gradients are multiplied through each layer. Sigmoid activation squishes values to (0,1), so gradients become exponentially smaller as they travel backward through layers. Deep networks stop learning because gradients are nearly zero. ReLU avoids this because its gradient is either 0 or 1 (for positive inputs).

### Q8: What is the Adam optimizer?
**A:** Adam (Adaptive Moment Estimation) is a gradient descent optimizer that adapts the learning rate for each parameter individually using estimates of first and second moments of gradients. It's generally preferred over plain SGD because it converges faster and requires less tuning.

### Q9: What is a Dense layer?
**A:** A Dense (fully connected) layer means every neuron in the current layer is connected to every neuron in the next layer. If layer A has 784 neurons and layer B has 100, there are 784×100 = 78,400 connections (each with a learnable weight).

### Q10: What is a confusion matrix?
**A:** A confusion matrix is an N×N table (N = number of classes) where entry [i][j] shows how many times class i was predicted as class j. The diagonal represents correct predictions. Off-diagonal entries are mistakes. It reveals which classes the model confuses with each other.

### Q11: What is sparse_categorical_crossentropy?
**A:** It's a loss function for multi-class classification.
- `categorical` → labels are discrete categories (not continuous).
- `sparse` → labels are given as integers (e.g., 3) rather than one-hot vectors (e.g., [0,0,0,1,0,...]).
- Cross-entropy penalizes confident wrong predictions more harshly than uncertain wrong ones.

### Q12: Why is MNIST used as a benchmark?
**A:** MNIST is standardized, widely understood, and small enough to train quickly even on a CPU. This makes it ideal for comparing algorithms, debugging models, and teaching concepts. It's the standard "sanity check" dataset — if your model can't do well on MNIST, something is fundamentally wrong.

### Q13: What is overfitting and how do you prevent it?
**A:** Overfitting occurs when a model memorizes the training data instead of learning generalizable patterns, leading to high training accuracy but low test accuracy. Prevention techniques:
- **Dropout** — randomly disable neurons during training
- **Regularization** (L1/L2) — penalize large weights
- **Early stopping** — stop training when validation loss starts increasing
- **More data** — larger datasets reduce overfitting

### Q14: What is the difference between a shallow and deep neural network?
**A:** A **shallow** network has 0–1 hidden layers. A **deep** network has 2+ hidden layers. "Deep Learning" refers to networks with many hidden layers. More layers = ability to learn more abstract, hierarchical representations (e.g., edges → shapes → objects).

### Q15: Why do we use np.argmax() for predictions?
**A:** The model's output is a probability distribution across 10 classes (one value per digit). `np.argmax()` returns the **index** of the highest value, which corresponds to the predicted digit. For example, if index 7 has the highest probability, the predicted digit is 7.

---

## Glossary of Key Terms

| Term | Definition |
|------|-----------|
| **ANN** | Artificial Neural Network — a model made of layers of interconnected neurons |
| **Neuron** | A single computational unit that takes inputs, applies weights+bias, and passes through an activation function |
| **Weight** | Learnable parameter connecting two neurons; adjusted during training |
| **Bias** | A learnable offset added to the neuron's weighted sum; shifts the activation function |
| **Activation Function** | A non-linear function applied to a neuron's output (e.g., ReLU, Sigmoid) |
| **Layer** | A group of neurons that process data together |
| **Dense/Fully Connected** | Every neuron in one layer connects to every neuron in the next |
| **Input Layer** | The first layer that receives raw data |
| **Hidden Layer** | Layers between input and output; learn intermediate representations |
| **Output Layer** | The final layer that produces predictions |
| **Epoch** | One complete pass through the training dataset |
| **Loss Function** | Measures how wrong the model's predictions are |
| **Optimizer** | Algorithm that updates weights to minimize loss (e.g., Adam, SGD) |
| **Backpropagation** | Algorithm to compute gradients and update weights layer by layer |
| **Gradient Descent** | Optimization technique that moves weights in the direction that reduces loss |
| **Overfitting** | Model performs well on training data but poorly on unseen test data |
| **Normalization** | Scaling input values to a standard range (e.g., 0 to 1) |
| **Flattening** | Converting a 2D array into a 1D array |
| **Confusion Matrix** | A table showing correct vs incorrect predictions per class |
| **Accuracy** | Fraction of predictions that were correct |
| **ReLU** | Rectified Linear Unit: f(x) = max(0, x) |
| **Sigmoid** | Activation function that squishes outputs to (0,1) |
| **One-hot encoding** | Representing a class label as a binary vector (e.g., digit 3 → [0,0,0,1,0,0,0,0,0,0]) |
| **Hyperparameter** | Settings chosen before training (e.g., number of epochs, neurons, learning rate) |

---

## How to Run This Project

### Prerequisites

Make sure you have Python installed (3.7+ recommended).

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib seaborn
```

### Run the Notebook

```bash
jupyter notebook DEEPLEARNING.ipynb
```

Or open it in **Google Colab** (free GPU access — highly recommended for beginners):
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload the `.ipynb` file
3. Run cells one by one with Shift+Enter

### Run Order

Run cells **top to bottom** in order. Each cell depends on the ones above it.

---

## Key Takeaways

1. **Data preprocessing matters** — normalizing and reshaping data correctly is often more impactful than model architecture choices.
2. **Hidden layers unlock non-linearity** — going from ~92% to ~97% accuracy with just one hidden layer shows how powerful they are.
3. **Always evaluate on test data** — training accuracy alone is meaningless.
4. **The confusion matrix tells the full story** — accuracy is a single number; the confusion matrix shows where mistakes happen.
5. **Start simple, then add complexity** — the project correctly starts with a simple model and adds layers only to show the improvement.

---

*Built with TensorFlow 2.x + Keras | Dataset: MNIST | Task: Multi-class Classification*
