import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

wine = load_wine()
X = wine.data # type: ignore
y = wine.target # type: ignore
wine_names = wine.target_names # type: ignore

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

model.fit(X_train, y_train, epochs=20, batch_size=8)

model.evaluate(X_test, y_test)

y_pred_logits = model.predict(X_test)
y_pred = np.argmax(y_pred_logits, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine_names))

for i in range(3):
    sample = X_test[i:i+1]
    pred = np.argmax(model.predict(sample), axis=1)[0]
    print(f"\nWine {i+1}: {wine_names[pred]}")
