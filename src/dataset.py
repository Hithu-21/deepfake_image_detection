
from pathlib import Path

import pandas as pd
import tensorflow as tf

# ---------------------------------
# 1. CONFIGURATION
# ---------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "140k"
)

IMAGE_DIR = (
    DATASET_DIR
    / "real_vs_fake"
    / "real-vs-fake"
)

IMAGE_SIZE = 160
BATCH_SIZE = 32


# ---------------------------------
# 2. LOAD CSV FILE
# ---------------------------------

def load_dataframe(csv_name):
    """Load dataset information from a CSV file."""

    csv_path = DATASET_DIR / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {"path", "label", "label_str"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}"
        )

    return df


# ---------------------------------
# 3. RESOLVE IMAGE PATH
# ---------------------------------

def resolve_image_path(relative_path):
    """Convert CSV image path into absolute path."""

    image_path = IMAGE_DIR / str(relative_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return str(image_path)


# ---------------------------------
# 4. PREPROCESS IMAGE
# ---------------------------------

def preprocess_image(image_path, label, augment=False):

    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    image = tf.cast(image, tf.float32) / 255.0

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(
            image, lower=0.9, upper=1.1
        )
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


# ---------------------------------
# 5. CREATE TENSORFLOW DATASET
# ---------------------------------

def create_dataset(
    csv_name,
    shuffle=False,
    sample_size=None,
    augment=False
):
    """Create a TensorFlow dataset."""

    df = load_dataframe(csv_name)

    # Select a smaller subset if requested
    if sample_size is not None:
        df = df.sample(
            n=min(sample_size, len(df)),
            random_state=42
        )

    print(
        f"Loading {len(df)} images from {csv_name}"
    )

    # Resolve image paths
    image_paths = [
        resolve_image_path(path)
        for path in df["path"]
    ]

    # Extract labels
    labels = df["label"].to_numpy(
        dtype="float32"
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    # Shuffle training data
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 10000),
            reshuffle_each_iteration=True
        )

    # Preprocess images
    dataset = dataset.map(
        lambda image_path, label: preprocess_image(
            image_path, label, augment
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch images
    dataset = dataset.batch(BATCH_SIZE)

    # Improve data loading performance
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ---------------------------------
# 6. LOAD ALL DATASETS
# ---------------------------------

def load_datasets(
    train_size=8000,
    valid_size=2000
):

    train_ds = create_dataset(
        "train.csv",
        shuffle=True,
        sample_size=train_size
    )

    valid_ds = create_dataset(
        "valid.csv",
        sample_size=valid_size
    )

    test_ds = create_dataset(
        "test.csv"
    )

    return train_ds, valid_ds, test_ds
