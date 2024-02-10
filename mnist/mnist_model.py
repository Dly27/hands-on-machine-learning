from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist['data'], mnist['target']

X = X[:10000]
y = y[:10000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

# GridSearchCV without specifying pos_label
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

joblib.dump(best_model, 'mnist_model.joblib')

# print(best_model.predict(X[0:]))