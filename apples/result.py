import pandas as pd
import joblib

best_model = joblib.load('best_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

data = pd.read_csv('apple_quality.csv').drop(4000, axis=0)
X = data.drop(columns=['Quality', 'A_id'], axis=1)

try:
    row_index = int(input("Select a row index to predict from: "))
except ValueError:
    print("Invalid input. Please enter a valid integer.")
    exit()

if row_index < 0 or row_index >= len(X):
    print(f"Invalid row index. Please select a value between 0 and {len(X) - 1}.")
else:
    single_row_prediction = best_model.predict(X.iloc[[row_index]])
    decoded_prediction = label_encoder.inverse_transform(single_row_prediction)

    actual_quality = data.loc[row_index, 'Quality']

    print(f"Predicted Quality: {decoded_prediction[0]}")
    print(f"Actual Quality: {actual_quality}")
