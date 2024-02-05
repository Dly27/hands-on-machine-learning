import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('apple_quality.csv').drop(4000, axis=0)

X = data.drop(columns=['Quality', 'A_id'], axis=1)
y = data['Quality']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), RandomForestClassifier(random_state=42))

param_dist = {
    'randomforestclassifier__n_estimators': [10, 50, 100],
    'randomforestclassifier__max_depth': [None, 10, 20],
    'randomforestclassifier__min_samples_split': [2, 5],
    'randomforestclassifier__min_samples_leaf': [1, 2],
    'randomforestclassifier__bootstrap': [True, False]
}

rand_search = RandomizedSearchCV(pipeline, scoring='accuracy', param_distributions=param_dist, cv=5)

rand_search.fit(X_train, y_train)

best_model = rand_search.best_estimator_

joblib.dump(best_model, 'best_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

final_predictions = best_model.predict(X_test)

#accuracy = accuracy_score(y_test, final_predictions)
#print(f'The accuracy is: {accuracy}')