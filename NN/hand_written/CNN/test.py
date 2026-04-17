from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# === Step 1: Load model ===
model = load_model("num_cnn_model.h5")

# === Step 2: Load custom image ===
# img_path = "numbers/5.png"

img_dir = "numbers"

# Get all image file paths
image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for image_file in sorted(image_files):  # Sorted for consistent order

    img_path = os.path.join(img_dir, image_file)    
    # Load, convert to grayscale, resize to 28x28
    img = Image.open(img_path).convert("L").resize((28, 28))

    # Invert if filename starts with 'w'
    if image_file.lower().startswith("w"):
        img = np.invert(img)

    # Convert to array and normalize
    img_array = np.array(img).astype("float32") / 255.0

    # Reshape to match input shape (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    # === Step 3: Predict ===
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # === Step 4: Show result ===
    plt.imshow(img_array[0].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted Digit: {predicted_label}")
    plt.axis("off")
    plt.show()
