import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load numeric dataset (Iris)
iris = load_iris()
X = iris.data  # 4 numeric features
y = iris.target  # 3 classes

# 2. Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Build simple numeric NN
model = tf.keras.Sequential(
    [
        layers.Dense(16, activation="relu", input_shape=(4,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(3),  # 3 classes logits
    ]
)

# 5. Compile
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 6. Train
model.fit(X_train, y_train, epochs=20, batch_size=8)

# 7. Evaluate
model.evaluate(X_test, y_test)

# 8. Make a prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example iris flower
sample = scaler.transform(sample)  # Normalize
pred_logits = model.predict(sample)
pred_class = np.argmax(pred_logits, axis=1)
print("Predicted class:", pred_class)
