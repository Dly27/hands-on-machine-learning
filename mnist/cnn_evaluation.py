import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the saved model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Convert labels to integers
y = y.astype(int)

# Reduce data size
X = X[:10000]
y = y[:10000]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to NumPy arrays and normalize
X_test = X_test.values.reshape(-1, 28, 28, 1) / 255.0

# Predict using the loaded model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Define evaluation metrics and plot confusion matrix
def error_analysis(y_true, y_pred):
    # Declare metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')

    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score (Micro): {f1_micro}")
    print(f"F1 Score (Macro): {f1_macro}")
    print(f"Precision (Micro): {precision_micro}")
    print(f"Precision (Macro): {precision_macro}")
    print(f"Recall (Micro): {recall_micro}")
    print(f"Recall (Macro): {recall_macro}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized.round(2), annot=True, cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

error_analysis(y_test, y_pred_classes)
