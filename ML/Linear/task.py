
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ЗАВДАННЯ 1: Базова модель

print("\n" + "="*80)
print("ЗАВДАННЯ 1: Прогнозування ціни авто (базові ознаки)")
print("="*80)

# Завантаження даних
df_basic = pd.read_csv("./assets/cars.csv")
X_basic = df_basic[['year', 'engine_volume', 'mileage', 'horsepower']]
y_basic = df_basic['price']

# Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_basic, y_basic, test_size=0.2, random_state=42
)

# Навчання моделі
model_basic = LinearRegression()
model_basic.fit(X_train, y_train)
y_pred_basic = model_basic.predict(X_test)

# Оцінка
mae_basic = mean_absolute_error(y_test, y_pred_basic)
r2_basic = r2_score(y_test, y_pred_basic)
mape_basic = np.mean(np.abs((y_test - y_pred_basic) / y_test)) * 100

print(f"R²: {r2_basic:.4f}")
print(f"MAE: {mae_basic:,.2f} грн")
print(f"MAPE: {mape_basic:.2f}%")

# Графік
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_basic, alpha=0.6, s=50, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         color='red', linestyle='--', linewidth=2, label='Ідеальна')
plt.xlabel('Справжня ціна (грн)', fontsize=11)
plt.ylabel('Прогнозована ціна (грн)', fontsize=11)
plt.title('Завдання 1: Справжня vs Прогнозована ціна', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ЗАВДАННЯ 2: Модель з категоріальними ознаками (brand, model)

print("\n" + "="*80)
print("ЗАВДАННЯ 2: Прогнозування ціни авто (з brand та model)")
print("="*80)

# Завантаження даних
df_extended = pd.read_csv("./assets/cars_plus.csv")
df_extended['car_age'] = 2025 - df_extended['year']

# Вибір ознак
categorical_features = ['brand', 'model']
numeric_features = ['engine_volume', 'mileage', 'horsepower', 'car_age']
X_extended = df_extended[categorical_features + numeric_features]
y_extended = df_extended['price']

# Розділення на train/test
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
    X_extended, y_extended, test_size=0.2, random_state=42
)

# Пайплайн з OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ],
    remainder='passthrough'
)

model_extended = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Навчання моделі
model_extended.fit(X_train_ext, y_train_ext)
y_pred_extended = model_extended.predict(X_test_ext)

# Оцінка
mae_extended = mean_absolute_error(y_test_ext, y_pred_extended)
r2_extended = r2_score(y_test_ext, y_pred_extended)
mape_extended = np.mean(np.abs((y_test_ext - y_pred_extended) / y_test_ext)) * 100

print(f"R²: {r2_extended:.4f}")
print(f"MAE: {mae_extended:,.2f} грн")
print(f"MAPE: {mape_extended:.2f}%")

# Графік
plt.figure(figsize=(8, 6))
plt.scatter(y_test_ext, y_pred_extended, alpha=0.6, s=50, color='purple')
plt.plot([y_test_ext.min(), y_test_ext.max()], [y_test_ext.min(), y_test_ext.max()], 
         color='red', linestyle='--', linewidth=2, label='Ідеальна')
plt.xlabel('Справжня ціна (грн)', fontsize=11)
plt.ylabel('Прогнозована ціна (грн)', fontsize=11)
plt.title('Завдання 2: Справжня vs Прогнозована ціна (з brand, model)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ЗАВДАННЯ 3: Графік залежності помилки від пробігу

print("\n" + "="*80)
print("ЗАВДАННЯ 3: Залежність помилки від пробігу")
print("="*80)

# Обчислення помилок
mileage_test = X_test_ext['mileage'].values
absolute_error = np.abs(y_test_ext.values - y_pred_extended)
relative_error = (absolute_error / y_test_ext.values) * 100

print(f"Середня абсолютна помилка: {np.mean(absolute_error):,.2f} грн")
print(f"Середня відносна помилка: {np.mean(relative_error):.2f}%")

# Графік
plt.figure(figsize=(10, 5))

# Абсолютна помилка
plt.subplot(1, 2, 1)
plt.scatter(mileage_test, absolute_error, alpha=0.6, s=50, color='darkred')
z = np.polyfit(mileage_test, absolute_error, 1)
p = np.poly1d(z)
plt.plot(sorted(mileage_test), p(sorted(mileage_test)), "r--", linewidth=2, label='Тренд')
plt.xlabel('Пробіг (тис. км)', fontsize=11)
plt.ylabel('Абсолютна помилка (грн)', fontsize=11)
plt.title('Абсолютна помилка vs Пробіг', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Відносна помилка
plt.subplot(1, 2, 2)
plt.scatter(mileage_test, relative_error, alpha=0.6, s=50, color='darkorange')
z2 = np.polyfit(mileage_test, relative_error, 1)
p2 = np.poly1d(z2)
plt.plot(sorted(mileage_test), p2(sorted(mileage_test)), "r--", linewidth=2, label='Тренд')
plt.xlabel('Пробіг (тис. км)', fontsize=11)
plt.ylabel('Відносна помилка (%)', fontsize=11)
plt.title('Відносна помилка vs Пробіг', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
