import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load wine dataset
wine = load_wine()
X = wine.data # type: ignore
y = wine.target # type: ignore
wine_names = wine.target_names # type: ignore

print("Wine Dataset:")
print(f"Features shape: {X.shape}")
print(f"Classes: {wine_names}")
print(f"Samples per class: {np.bincount(y)}")

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Build neural network
model = tf.keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(13,)), # type: ignore
    layers.Dense(8, activation="relu"),
    layers.Dense(3),
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train model
print("\nTraining model...")
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# Evaluate model
print("\n" + "="*50)
print("Model Evaluation:")
print("="*50)
model.evaluate(X_test, y_test)

# Predictions
y_pred_logits = model.predict(X_test)
y_pred = np.argmax(y_pred_logits, axis=1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine_names))

# Predict some wines
print("\n" + "="*50)
print("Wine Classification Examples:")
print("="*50)
for i in range(5):
    sample = X_test[i:i+1]
    pred = np.argmax(model.predict(sample), axis=1)[0]
    print(f"Wine {i+1}: Predicted class = {wine_names[pred]}")

# Conclusions
print(f"1. Accuracy: The model achieved {accuracy*100:.2f}% accuracy")