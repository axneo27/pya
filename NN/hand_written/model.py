import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ---------- Load the data ----------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (scale pixel values to range 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Display the first image
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.show()

# ---------- Preprocess the Data ----------
from tensorflow.keras.utils import to_categorical

# Flatten the images
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ---------- Build the Neural Network ----------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the model
model = Sequential(
    [
        Dense(
            128, activation="relu", input_shape=(28*28,)
        ),  # Hidden layer with 128 neurons
        Dense(10, activation="softmax"),  # Output layer with 10 neurons
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Summary of the model
model.summary()

# ---------- Train the Neural Network ----------
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,  # Use 20% of the training data for validation
)

model.save("num_model.h5")

# ---------- Evaluate the Model ----------
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# ---------- Visualize the Training Process ----------
# Plot training & validation accuracy values
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# ---------- Make Predictions ----------
predictions = model.predict(x_test)

# Visualize a few predictions
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(y_test[i])}")
    plt.show()
