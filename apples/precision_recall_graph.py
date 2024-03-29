import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load model
best_model = joblib.load('apples/best_model.joblib')

# Get data
data = pd.read_csv('apples/apple_quality.csv').drop(index=4000)
X = data.drop(columns=['Quality', 'A_id'], axis=1)
y = data['Quality']

# Encode good and bad as 1 and 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into test/training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_true, y_probs = y_test, best_model.predict_proba(X_test)[:, 1]

# Create a precision recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_probs,)

# Plot Precision-Recall on a single graph
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim([0.0, 1.0])
ax.step(recall, precision, color='b', alpha=0.2, where='post')
ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.0])
ax.set_title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
