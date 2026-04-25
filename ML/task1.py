import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('../assets/fuel_consumption.csv')
X = data['speed_kmh'].values.reshape(-1, 1) # type: ignore
y = data['fuel_consumption_l_per_100km'].values

results = []
for degree in range(1, 6):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred) # type: ignore
    mse = mean_squared_error(y, y_pred) # type: ignore
    results.append((degree, mae, mse, model))
    print(f"Degree {degree}: MAE = {mae:.4f}, MSE = {mse:.4f}")

best_degree, best_mae, best_mse, best_model = min(results, key=lambda x: x[2])
print(f"\nBest model: Degree {best_degree} (MSE = {best_mse:.4f})")

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1) # type: ignore
y_pred_range = best_model.predict(X_range)

plt.figure(figsize=(10, 6))
plt.plot(X, y, 'o', label='Data points', color='blue') # type: ignore
plt.plot(X_range, y_pred_range, label=f'Polynomial degree {best_degree}', color='red', linestyle='-')
plt.xlabel('Speed (km/h)')
plt.ylabel('Fuel consumption (l/100km)')
plt.title('Fuel Consumption vs Speed')
plt.legend()
plt.grid(True)
plt.show()

speeds = np.array([35, 95, 140]).reshape(-1, 1)
predictions = best_model.predict(speeds)
for speed, fuel in zip(speeds.flatten(), predictions):
    print(f"Speed {speed} km/h: {fuel:.2f} l/100km")
