import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Завантаження даних
df = pd.read_csv("./assets/cars_plus.csv")

# 2. Створення нової ознаки: вік авто
df['car_age'] = 2025 - df['year']

# 3. Вибір ознак і цільової змінної
X = df[['brand', 'model', 'engine_volume', 'mileage', 'horsepower', 'car_age']]
y = df['price']

# 4. Попередня обробка: кодування категоріальних ознак
categorical_features = ['brand', 'model']
numeric_features = ['engine_volume', 'mileage', 'horsepower', 'car_age']

# 5. Побудова пайплайну
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Пропускає числові ознаки без змін
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 6. Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Навчання моделі
model.fit(X_train, y_train)

# 8. Прогноз на тестових даних
y_pred = model.predict(X_test)

# 9. Оцінка моделі
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")

# 10. Прогноз на нових даних
new_car = pd.DataFrame([{
    'brand': 'Toyota',
    'model': 'Camry',
    'engine_volume': 2.5,
    'mileage': 80,
    'horsepower': 200,
    'car_age': 2025 - 2018
}])

predicted_price = model.predict(new_car)[0]
print(f"\n Прогнозована ціна авто: {predicted_price:,.2f} грн")

# Візуалізація: справжні ціни vs прогноз
plt.scatter(y_test, y_pred)
plt.xlabel("Справжня ціна")
plt.ylabel("Прогнозована ціна")
plt.title("Справжня vs Прогнозована ціна")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()
