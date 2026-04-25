import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('../assets/internship_candidates.csv')

english_map = {1: 'Elementary', 2: 'Pre-Intermediate', 3: 'Intermediate', 4: 'Upper-Intermediate', 5: 'Advanced'}
data['EnglishLevelName'] = data['EnglishLevel'].map(english_map)

X = data[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']].values
y = data['Accepted'].values

model = LogisticRegression()
model.fit(X, y) # type: ignore

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred) # type: ignore
print(f"Accuracy: {accuracy:.4f}")

plt.scatter(data['EnglishLevel'], data['EntryTestScore'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title('Logistic Regression Predictions')
plt.xlabel('English Level')
plt.ylabel('Entry Test Score')
plt.colorbar(label='Predicted Class')
plt.show()

plt.scatter(data['Grade'], data['Age'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title('Logistic Regression Predictions')
plt.xlabel('Grade')
plt.ylabel('Age')
plt.colorbar(label='Predicted Class')
plt.show()
