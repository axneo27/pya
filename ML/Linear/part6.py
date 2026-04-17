# Install necessary packages if needed
# pip install numpy scikit-learn matplotlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
# We simulate a function: y = 2x + 3
X = np.linspace(0, 10, 100).reshape(-1, 1)  # 100 samples between 0 and 10
y = 2 * X.flatten() + 3 + np.random.normal(0, 0.5, size=X.shape[0])  # add slight noise

print("X:", X)
print("Y:", y)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict values
X_test = np.linspace(0, 15, 100).reshape(-1, 1)  # test on a broader range
y_pred = model.predict(X_test)

# Show the function
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X_test, y_pred, color='red', label='Predicted function')
plt.xlabel('X')
plt.ylabel('y')
plt.title('ML Prediction of a Function')
plt.legend()
plt.grid(True)
plt.show()

# Example: predict a specific value
x_value = 7
predicted = model.predict(np.array([[x_value]]))
print(f"Predicted value at x={x_value}: {predicted[0]:.2f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Обчислюємо метрики
mae = mean_absolute_error(y, model.predict(X))
mse = mean_squared_error(y, model.predict(X))
r2 = r2_score(y, model.predict(X))

print(f"Mean Absolute Error (MAE): {mae:.4f}") # MAE — середня абсолютна помилка. Показує середню відстань між передбаченим та реальним значеннями.
print(f"Mean Squared Error (MSE): {mse:.4f}") # Коефіцієнт детермінації. Показує, яку частину варіації цільової змінної пояснює модель.
