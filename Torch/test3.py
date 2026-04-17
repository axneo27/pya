import torch
from PIL import Image
import numpy as np
from model3 import NumModel   # import class from separate file

model = NumModel()                               # create model instance
state_dict = torch.load("num_model.pth", map_location="cpu")

model.load_state_dict(state_dict)                # ✅ load into model
model.eval()

# Load and preprocess the image
img = Image.open("./NN/hand_written/numbers/w55.png").convert("L")  # Convert to grayscale
img = img.resize((28, 28))  # Resize to MNIST format
img = np.array(img)
img = 255 - img  # Invert (MNIST: white digit on black bg)
img = img / 255.0  # Normalize
img = img.reshape(1, 784)  # Flatten to (1, 784)

# Predict
with torch.no_grad():
    input_tensor = torch.tensor(img, dtype=torch.float32)
    prediction = model(input_tensor)
    probabilities = torch.softmax(prediction, dim=1).numpy()

for i in range(10):
    print(f"Probability of {i}: {probabilities[0][i]:.4f}")

# show image
import matplotlib.pyplot as plt
plt.imshow(img.reshape(28, 28), cmap="gray")
plt.title("Prepared image")
plt.axis("off")
plt.show()