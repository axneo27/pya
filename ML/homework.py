import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

num = pd.read_csv('../assets/internship_candidates_final_numeric.csv')
cefr = pd.read_csv('../assets/internship_candidates_cefr_final.csv')
data = num.copy()
data['EnglishLevelName'] = cefr['EnglishLevel']

X = data[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y = data['Accepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print('Accuracy:', accuracy_score(y_test, model.predict(X_test)))

levels = ['Elementary', 'Pre-Intermediate', 'Intermediate', 'Upper-Intermediate', 'Advanced']
entries = list(range(200, 1001, 20))
plt.figure(figsize=(10, 6))
for i, level in enumerate(levels, start=1):
    grid = pd.DataFrame({
        'Experience': [X['Experience'].median()] * len(entries),
        'Grade': [X['Grade'].median()] * len(entries),
        'EnglishLevel': [i] * len(entries),
        'Age': [X['Age'].median()] * len(entries),
        'EntryTestScore': entries,
    })
    prob = model.predict_proba(grid)[:, 1]
    plt.plot(entries, prob, label=level)

plt.xlabel('EntryTestScore')
plt.ylabel('Acceptance probability')
plt.title('Acceptance probability')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
