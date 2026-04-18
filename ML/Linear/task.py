
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ЗАВДАННЯ 1: Базова модель з основними ознаками

print("\n" + "="*80)
print("ЗАВДАННЯ 1: Прогнозування ціни авто (базові ознаки)")
print("="*80)

# 1. Завантаження даних
df_basic = pd.read_csv("./assets/cars.csv")
print(f"\nДані завантажені: {df_basic.shape[0]} записів, {df_basic.shape[1]} ознак")
print(f"Колони: {list(df_basic.columns)}")
print(f"\nПерші 5 рядків:\n{df_basic.head()}")

# 2. Підготовка ознак і цільової змінної
X_basic = df_basic[['year', 'engine_volume', 'mileage', 'horsepower']]
y_basic = df_basic['price']

# 3. Розділення на навчальну та тестову вибірки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_basic, y_basic, test_size=0.2, random_state=42
)

print(f"\nРозподіл даних:")
print(f"  Навчальна вибірка: {X_train.shape[0]} записів")
print(f"  Тестова вибірка: {X_test.shape[0]} записів")

# 4. Навчання моделі лінійної регресії
model_basic = LinearRegression()
model_basic.fit(X_train, y_train)

print(f"\nКоефіцієнти моделі:")
for feature, coef in zip(X_basic.columns, model_basic.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"  Вільний член: {model_basic.intercept_:.2f}")

# 5. Прогнозування на тестовій вибірці
y_pred_basic = model_basic.predict(X_test)

# 6. Оцінка моделі
mae_basic = mean_absolute_error(y_test, y_pred_basic)
r2_basic = r2_score(y_test, y_pred_basic)
rmse_basic = np.sqrt(mean_squared_error(y_test, y_pred_basic))
mape_basic = np.mean(np.abs((y_test - y_pred_basic) / y_test)) * 100

print(f"\nМетрики якості моделі:")
print(f"  R² (коефіцієнт детермінації): {r2_basic:.4f}")
print(f"  MAE (середня абсолютна помилка): {mae_basic:,.2f} грн")
print(f"  RMSE (середньоквадратична помилка): {rmse_basic:,.2f} грн")
print(f"  MAPE (середня відсоткова помилка): {mape_basic:.2f}%")

# 7. Приклад прогнозування для нового автомобіля
new_car_basic = pd.DataFrame([{
    'year': 2020,
    'engine_volume': 2.0,
    'mileage': 50,
    'horsepower': 150
}])
predicted_price = model_basic.predict(new_car_basic)[0]
print(f"\nПриклад прогнозування для авто (2020, 2.0L, 50тис.км, 150к.с.):")
print(f"  Прогнозована ціна: {predicted_price:,.2f} грн")

# 8. Візуалізація 1: Справжня vs Прогнозована ціна
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_basic, alpha=0.6, s=50, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         color='red', linestyle='--', linewidth=2, label='Ідеальна модель')
plt.xlabel('Справжня ціна (грн)', fontsize=12)
plt.ylabel('Прогнозована ціна (грн)', fontsize=12)
plt.title('Завдання 1: Справжня vs Прогнозована ціна (базові ознаки)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9. Візуалізація 2: Залишки (Residuals)
residuals = y_test - y_pred_basic
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_basic, residuals, alpha=0.6, s=50, color='green')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Прогнозована ціна (грн)', fontsize=12)
plt.ylabel('Залишки (грн)', fontsize=12)
plt.title('Завдання 1: Залишки моделі', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ЗАВДАННЯ 2: Розширена модель з категоріальними ознаками (brand, model)

print("\n" + "="*80)
print("ЗАВДАННЯ 2: Прогнозування ціни авто (з brand та model)")
print("="*80)

# 1. Завантаження розширених даних
df_extended = pd.read_csv("./assets/cars_plus.csv")
print(f"\nДані завантажені: {df_extended.shape[0]} записів, {df_extended.shape[1]} ознак")
print(f"Колони: {list(df_extended.columns)}")
print(f"\nПерші 5 рядків:\n{df_extended.head()}")

# 2. Створення нової ознаки: вік автомобіля
current_year = 2025
df_extended['car_age'] = current_year - df_extended['year']

# 3. Вибір ознак (включаючи категоріальні) і цільової змінної
categorical_features = ['brand', 'model']
numeric_features = ['engine_volume', 'mileage', 'horsepower', 'car_age']
X_extended = df_extended[categorical_features + numeric_features]
y_extended = df_extended['price']

print(f"\nКатегоріальні ознаки ({len(categorical_features)}): {categorical_features}")
print(f"Числові ознаки ({len(numeric_features)}): {numeric_features}")
print(f"Унікальні бренди: {df_extended['brand'].nunique()}")
print(f"Унікальні моделі: {df_extended['model'].nunique()}")

# 4. Розділення на навчальну та тестову вибірки (80/20)
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
    X_extended, y_extended, test_size=0.2, random_state=42
)

print(f"\nРозподіл даних:")
print(f"  Навчальна вибірка: {X_train_ext.shape[0]} записів")
print(f"  Тестова вибірка: {X_test_ext.shape[0]} записів")

# 5. Побудова пайплайну з кодуванням категоріальних ознак
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
         categorical_features),
        ('num', 'passthrough', numeric_features)
    ],
    remainder='passthrough'
)

model_extended = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 6. Навчання моделі
print("\nНавчання моделі з кодуванням категоріальних ознак...")
model_extended.fit(X_train_ext, y_train_ext)
print("Модель навчена!")

# 7. Прогнозування на тестовій вибірці
y_pred_extended = model_extended.predict(X_test_ext)

# 8. Оцінка моделі
mae_extended = mean_absolute_error(y_test_ext, y_pred_extended)
r2_extended = r2_score(y_test_ext, y_pred_extended)
rmse_extended = np.sqrt(mean_squared_error(y_test_ext, y_pred_extended))
mape_extended = np.mean(np.abs((y_test_ext - y_pred_extended) / y_test_ext)) * 100

print(f"\nМетрики якості моделі:")
print(f"  R² (коефіцієнт детермінації): {r2_extended:.4f}")
print(f"  MAE (середня абсолютна помилка): {mae_extended:,.2f} грн")
print(f"  RMSE (середньоквадратична помилка): {rmse_extended:,.2f} грн")
print(f"  MAPE (середня відсоткова помилка): {mape_extended:.2f}%")

# 9. Приклад прогнозування для нового автомобіля
new_car_extended = pd.DataFrame([{
    'brand': 'Toyota',
    'model': 'Camry',
    'engine_volume': 2.5,
    'mileage': 80,
    'horsepower': 200,
    'car_age': current_year - 2018
}])
predicted_price_ext = model_extended.predict(new_car_extended)[0]
print(f"\nПриклад прогнозування для Toyota Camry (2018, 2.5L, 80тис.км, 200к.с.):")
print(f"  Прогнозована ціна: {predicted_price_ext:,.2f} грн")

# 10. Візуалізація 1: Справжня vs Прогнозована ціна
plt.figure(figsize=(10, 6))
plt.scatter(y_test_ext, y_pred_extended, alpha=0.6, s=50, color='purple')
plt.plot([y_test_ext.min(), y_test_ext.max()], [y_test_ext.min(), y_test_ext.max()], 
         color='red', linestyle='--', linewidth=2, label='Ідеальна модель')
plt.xlabel('Справжня ціна (грн)', fontsize=12)
plt.ylabel('Прогнозована ціна (грн)', fontsize=12)
plt.title('Завдання 2: Справжня vs Прогнозована ціна (з brand та model)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 11. Візуалізація 2: Залишки (Residuals)
residuals_ext = y_test_ext - y_pred_extended
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_extended, residuals_ext, alpha=0.6, s=50, color='orange')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Прогнозована ціна (грн)', fontsize=12)
plt.ylabel('Залишки (грн)', fontsize=12)
plt.title('Завдання 2: Залишки моделі', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Порівняння моделей

print(f"\n{'Метрика':<40} {'Завдання 1':<20} {'Завдання 2':<20}")
print(f"{'-'*80}")
print(f"{'R² (детермінація)':<40} {r2_basic:<20.4f} {r2_extended:<20.4f}")
print(f"{'MAE (середня абсолютна помилка)':<40} {mae_basic:<20,.0f} {mae_extended:<20,.0f}")
print(f"{'RMSE (середньоквадратична помилка)':<40} {rmse_basic:<20,.0f} {rmse_extended:<20,.0f}")
print(f"{'MAPE (середня відсоткова помилка)':<40} {mape_basic:<20.2f}% {mape_extended:<20.2f}%")

improvement = ((mape_basic - mape_extended) / mape_basic) * 100
print(f"\nПокращення MAPE при додаванні категоріальних ознак: {improvement:.2f}%")
