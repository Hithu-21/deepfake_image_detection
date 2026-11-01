
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.dataset import create_dataset


# ---------------------------------
# 1. CONFIGURATION
# ---------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "best_model_finetuned.keras"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------
# 2. LOAD TRAINED MODEL
# ---------------------------------

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ---------------------------------
# 3. LOAD TEST DATASET
# ---------------------------------

print("\nLoading test dataset...")

test_data = create_dataset(
    "test.csv",
    shuffle=False
)

print("Test dataset loaded!")


# ---------------------------------
# 4. GET ACTUAL LABELS
# ---------------------------------

actual_labels = np.concatenate([
    labels.numpy()
    for images, labels in test_data
])


# ---------------------------------
# 5. PREDICT TEST IMAGES
# ---------------------------------

print("\nPredicting test images...")

predictions = model.predict(test_data)

predicted_labels = (
    predictions.flatten() >= 0.5
).astype(int)


# ---------------------------------
# 6. CALCULATE METRICS
# ---------------------------------

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision = precision_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)


print("\n========== TEST RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ---------------------------------
# 7. CLASSIFICATION REPORT
# ---------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["FAKE", "REAL"],
        zero_division=0
    )
)


# ---------------------------------
# 8. CONFUSION MATRIX
# ---------------------------------

cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["FAKE", "REAL"]
)

display.plot(
    values_format="d"
)

plt.title("Deepfake Detection - Confusion Matrix")

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nEvaluation completed!")
