import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("ЗАВДАННЯ 1: Прогнозування електроспоживання (базові ознаки)")

df1 = pd.read_csv("./assets/energy_usage.csv")
X1 = df1[['temperature', 'humidity', 'hour', 'is_weekend']]
y1 = df1['consumption']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

mape1 = np.mean(np.abs((y1_test - y1_pred) / y1_test)) * 100
r2_1 = r2_score(y1_test, y1_pred)

print(f"R2: {r2_1:.4f}")
print(f"MAPE: {mape1:.2f}%")

plt.figure(figsize=(8, 6))
plt.scatter(y1_test, y1_pred, alpha=0.6, s=50, color='blue')
plt.plot([y1_test.min(), y1_test.max()], [y1_test.min(), y1_test.max()], 
         color='red', linestyle='--', linewidth=2, label='Ідеальна')
plt.xlabel('Справжня потужність (кВт·год)', fontsize=11)
plt.ylabel('Прогнозована потужність (кВт·год)', fontsize=11)
plt.title('Завдання 1: Справжня vs Прогнозована потужність', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("ЗАВДАННЯ 2: Прогнозування електроспоживання (з категоріальними ознаками)")

df2 = pd.read_csv("./assets/energy_usage_plus.csv")

categorical_features = ['season', 'district_type']
numeric_features = ['temperature', 'humidity', 'hour', 'is_weekend']
X2 = df2[categorical_features + numeric_features]
y2 = df2['consumption']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ],
    remainder='passthrough'
)

model2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

mape2 = np.mean(np.abs((y2_test - y2_pred) / y2_test)) * 100
r2_2 = r2_score(y2_test, y2_pred)

print(f"R2: {r2_2:.4f}")
print(f"MAPE: {mape2:.2f}%")

plt.figure(figsize=(8, 6))
plt.scatter(y2_test, y2_pred, alpha=0.6, s=50, color='green')
plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], 
         color='red', linestyle='--', linewidth=2, label='Ідеальна')
plt.xlabel('Справжня потужність (кВт·год)', fontsize=11)
plt.ylabel('Прогнозована потужність (кВт·год)', fontsize=11)
plt.title('Завдання 2: Справжня vs Прогнозована потужність', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
