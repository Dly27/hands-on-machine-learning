import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load model
best_model = joblib.load('best_model.joblib')

# Get data
data = pd.read_csv('apple_quality.csv').drop(index=4000)
X = data.drop(columns=['Quality', 'A_id'], axis=1)
y = data['Quality']

# Encode good and bad as 1 and 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into test/training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_score = best_model.predict_proba(X_test)[:, 1]

# Create a roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_score=y_score)
roc_auc = auc(fpr, tpr)

# Plot roc curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.show()


