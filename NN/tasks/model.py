import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# just dense
x_train_flat = x_train.reshape(-1, 784) / 255.0
x_test_flat  = x_test.reshape(-1, 784)  / 255.0
# advanced
x_train_cnn  = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_cnn   = x_test.reshape(-1, 28, 28, 1)  / 255.0

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test,  10)

# 1: Dense  784 inputs   128 neurons
# 2: Dense  128 inputs    64 neurons
# 3: Dense   64 inputs    10 neurons 
just_dense_model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)), # type: ignore
    Dense(64,  activation='relu'),
    Dense(10,  activation='softmax'),
])

# 1: Conv2D    32 filters, 3×3 window   output 26×26×32
# 2: MaxPool   2×2 shrink               output 13×13×32  (no weights)
# 3: Conv2D    64 filters, 3×3 window   output 11×11×64
# 4: MaxPool   2×2 shrink               output  5×5×64   (no weights)
# 5: Flatten   unrolls 5×5×64=1600      output 1600      (no weights)
# 6: Dense     1600 inputs    128 neurons
# 7: Dropout   randomly disables 30% during training      (no weights)
# 8: Dense     128 inputs    10 neurons 
advanced_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # type: ignore
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),
])

##########
is_advanced = True
if is_advanced:
    model = advanced_model 
    x_train_in = x_train_cnn 
    x_test_in  = x_test_cnn
else :
    model = just_dense_model
    x_train_in = x_train_flat
    x_test_in  = x_test_flat

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    x_train_in, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test_in, y_test),
)

model.save('fashion_model.h5')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Train accuracy')
ax1.plot(history.history['val_accuracy'], label='Val accuracy')
ax1.set_title('Accuracy over epochs (Full CNN)' if is_advanced else 'Accuracy over epochs (Just Dense Model)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'],     label='Train loss')
ax2.plot(history.history['val_loss'], label='Val loss')
ax2.set_title('Loss over epochs (Full CNN)' if is_advanced else 'Loss over epochs (Just Dense Model)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()
