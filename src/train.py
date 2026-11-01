
import tensorflow as tf
from pathlib import Path

from src.dataset import create_dataset
from tensorflow.keras.layers import BatchNormalization


# ---------------------------------
# CONFIGURATION
# ---------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"

OLD_MODEL = MODEL_DIR / "best_model_backup.keras"
NEW_MODEL = MODEL_DIR / "best_model_finetuned.keras"

TRAIN_SIZE = 30000
VALID_SIZE = 5000
EPOCHS = 3

tf.keras.utils.set_random_seed(42)


# ---------------------------------
# LOAD DATASETS
# ---------------------------------

print("Loading datasets...")

train_data = create_dataset(
    "train.csv",
    shuffle=True,
    sample_size=TRAIN_SIZE,
    augment=True
)

valid_data = create_dataset(
    "valid.csv",
    sample_size=VALID_SIZE
)


# ---------------------------------
# LOAD EXISTING MODEL
# ---------------------------------

print("Loading existing best model...")

model = tf.keras.models.load_model(OLD_MODEL)

# Find MobileNetV2 inside the model
base_model = next(
    layer for layer in model.layers
    if isinstance(layer, tf.keras.Model)
)

# Unfreeze only the last 30 layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True


# ---------------------------------
# COMPILE MODEL
# ---------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)


# ---------------------------------
# CALLBACKS
# ---------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=NEW_MODEL,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]


# ---------------------------------
# FINE-TUNE
# ---------------------------------

print("Starting fine-tuning...")

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nFine-tuning completed!")
print("New model saved at:", NEW_MODEL)