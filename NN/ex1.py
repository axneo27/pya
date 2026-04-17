from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Build a simple NN for numeric data
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu", input_shape=(2,)),  # 2 input features
        layers.Dense(8, activation="relu"),
        layers.Dense(1),  # single numeric output (regression)
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

model.summary()

# Example numeric training data
# X: 100 rows with 2 features
X = np.random.rand(100, 2).astype("float32")

# y: target values
y = (X[:, 0] * 3 + X[:, 1] * 2 + 1).astype("float32")  # simple formula

# Train model
model.fit(X, y, epochs=5)

# Make a prediction
example = np.array([[0.3, 0.8]])
pred = model.predict(example)

print("Prediction:", pred)
