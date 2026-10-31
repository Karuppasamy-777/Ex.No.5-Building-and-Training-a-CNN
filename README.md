
# Ex. No. 5 — Building and Training a Convolutional Neural Network (CNN)

![CIFAR-10 dataset sample images](https://cdn.kastatic.org/ka-perseus-images/d8663231632907eadede6aecb68c4f00f7c21ff1.png)

## Aim

To build and train a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset.

## Objectives

- Understand the basic architecture and operation of a CNN.
- Load and preprocess the CIFAR-10 image dataset.
- Normalize image pixel values and encode class labels.
- Build a CNN using convolution, pooling, and dropout layers.
- Compile and train the model.
- Visualize training and validation accuracy and loss.
- Evaluate the trained model using test images.
- Display predicted and actual class labels.

## Introduction

A Convolutional Neural Network (CNN) is a deep learning model commonly used for image classification and computer vision. CNNs automatically learn spatial features such as edges, textures, shapes, and object patterns from images.

In this experiment, a CNN is trained on CIFAR-10, a dataset containing 60,000 color images divided into ten object categories.

## Dataset Description

The CIFAR-10 dataset contains 32 × 32 RGB images.

| Dataset | Number of Images |
|---|---:|
| Training set | 50,000 |
| Test set | 10,000 |
| Total | 60,000 |

The ten classes are:

| Label | Class |
|---:|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

## Software Requirements

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib

## Theory

### CNN Layers

| Layer | Function |
|---|---|
| Convolutional layer | Extracts features using learnable filters. |
| ReLU activation | Introduces non-linearity by replacing negative values with zero. |
| Max pooling | Reduces spatial dimensions while retaining prominent features. |
| Dropout | Reduces overfitting by randomly disabling units during training. |
| Flatten | Converts feature maps into a one-dimensional vector. |
| Dense layer | Learns high-level combinations of extracted features. |
| Softmax | Produces class probabilities for the ten output classes. |

### Workflow

```text
                 START
                   |
                   v
          Import libraries
                   |
                   v
          Load CIFAR-10 data
                   |
                   v
       Normalize image pixel values
                   |
                   v
        One-hot encode the labels
                   |
                   v
        Visualize sample images
                   |
                   v
            Build CNN model
                   |
                   v
         Compile model with Adam
                   |
                   v
         Train for 30 epochs
                   |
                   v
     Plot accuracy and loss curves
                   |
                   v
       Predict classes of test images
                   |
                   v
                  END
```

## Procedure

1. Import TensorFlow, Keras, NumPy, and Matplotlib.
2. Load the CIFAR-10 dataset.
3. Normalize image pixel values to the range 0–1.
4. Convert class labels into one-hot encoded vectors.
5. Display a 4 × 4 grid of sample training images.
6. Build the CNN using convolutional, pooling, dropout, and dense layers.
7. Compile the model using the Adam optimizer and categorical cross-entropy loss.
8. Train the model for 30 epochs with a batch size of 64.
9. Plot training and validation accuracy and loss.
10. Predict the classes of selected test images.

## Implementation

### Step 1 — Import Libraries

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
```

### Step 2 — Load CIFAR-10 Dataset

```python
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)
```

Expected shapes:

```text
Training images: (50000, 32, 32, 3)
Testing images: (10000, 32, 32, 3)
```

### Step 3 — Preprocess Data

Normalize pixel values and one-hot encode the labels.

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

### Step 4 — Visualize Sample Images

```python
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.imshow(X_train[i])
    plt.xlabel(class_names[np.argmax(y_train[i])])

plt.tight_layout()
plt.show()
```

### Step 5 — Build the CNN Model

```python
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # Convolutional block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Convolutional block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Convolutional block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Classification head
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.summary()
```

### Step 6 — Compile the Model

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

- **Optimizer:** Adam
- **Loss function:** Categorical cross-entropy
- **Metric:** Accuracy

### Step 7 — Train the Model

```python
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2
)
```

The model trains using 80% of the training data and validates using the remaining 20%. The separate CIFAR-10 test set remains reserved for final evaluation.

### Step 8 — Plot Training History

```python
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
```

**Interpreting the graphs:**

- Training accuracy should generally increase as the model learns.
- Validation accuracy indicates how well the model performs on unseen validation images.
- Training loss should generally decrease during learning.
- If training improves while validation performance deteriorates, the model may be overfitting.

### Step 9 — Evaluate on Test Data

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```

### Step 10 — Predict Test Images

```python
def plot_prediction(index):
    img = X_test[index]
    true_label = class_names[np.argmax(y_test[index])]

    probabilities = model.predict(
        np.expand_dims(img, axis=0),
        verbose=0
    )

    predicted_label = class_names[np.argmax(probabilities)]

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(
        f"True: {true_label}\nPredicted: {predicted_label}"
    )
    plt.axis("off")
    plt.show()

for i in range(5):
    plot_prediction(i)
```

## Results and Observations

- The CNN learns image features through successive convolutional layers.
- Max pooling reduces feature-map dimensions and computational cost.
- Dropout helps reduce overfitting during training.
- Training and validation curves show how the model learns and generalizes.
- Test predictions show the model's ability to classify unseen images.

Record the actual test accuracy and loss produced by your training run. Results may vary depending on the training environment and model execution.


## References

- [TensorFlow CIFAR-10 Dataset Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data)
- [Keras CIFAR-10 Dataset Documentation](https://keras.io/api/datasets/cifar10/)
- [CIFAR-10 Dataset — University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)
