# ==========================================================
# Brain Tumor Detection using CNN
# Model Evaluation
# ==========================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# Project Paths
# ==========================================================

MODEL_PATH = "models/brain_tumor_cnn.keras"
TEST_DIR = "dataset/test"
RESULTS_DIR = "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================================
# Load Trained Model
# ==========================================================

print("=" * 60)
print("Loading Trained CNN Model")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully\n")

# ==========================================================
# Load Test Dataset
# ==========================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("Detected Classes")

for i, name in enumerate(class_names):
    print(f"{i} --> {name}")

# ==========================================================
# Normalize Images
# ==========================================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# ==========================================================
# Predict
# ==========================================================

print("\nPredicting...\n")

predictions = model.predict(test_dataset)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.concatenate(
    [y.numpy() for x, y in test_dataset],
    axis=0
)

# ==========================================================
# Calculate Metrics
# ==========================================================

accuracy = accuracy_score(true_labels, predicted_labels)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="weighted"
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted"
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted"
)

print("=" * 50)

print(f"Accuracy  : {accuracy*100:.2f}%")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print("=" * 50)

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(

    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )

)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(8,8))

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

)

plt.close()

print("\nConfusion Matrix Saved")

print("\nEvaluation Completed Successfully")