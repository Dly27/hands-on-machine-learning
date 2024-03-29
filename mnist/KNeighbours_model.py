from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist['data'], mnist['target']

# Reduce dataset
X = X[:10000]
y = y[:10000]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

# Define parameter grid
param_grid = {
    'pca__n_components': [50, 100, 200],
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance']
}

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Getbest model from grid search
best_model = grid_search.best_estimator_

# Save the best model to a file
joblib.dump(best_model, 'KNeighbours.joblib')

# Load saved model
best_model = joblib.load('KNeighbours.joblib')

