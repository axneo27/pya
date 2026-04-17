# create fake binary classification dataset with 1000 rows and 10 features
from sklearn.datasets import make_classification

X, y = make_classification(n_samples = 1000, n_features = 10, n_classes = 2)

# check shape of X and y
print(X.shape, y.shape)

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# import and initialize logistic regression model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

# fit logistic regression model
lr.fit(X_train, y_train)


# generate hard predictions on test set
y_pred = lr.predict(X_test)
print(y_pred)

# evaluate accuracy score of the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))