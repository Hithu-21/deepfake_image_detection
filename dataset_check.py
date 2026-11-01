import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load training CSV
csv_path = "dataset/140k/train.csv"
df = pd.read_csv(csv_path)

print("Dataset loaded successfully!")
print("Total training images:", len(df))

# Show distribution of real and fake images
print("\nLabel distribution:")
print(df["label_str"].value_counts())

# Take the first image
row = df.iloc[0]

print("\nSample information:")
print("ID:", row["id"])
print("Label:", row["label_str"])
print("CSV path:", row["path"])

# The actual dataset has an additional "real-vs-fake" folder
image_path = os.path.join(
    "dataset/140k/real_vs_fake/real-vs-fake",
    row["path"]
)

print("Local image path:", image_path)


# Check one real image and one fake image
sample_rows = [
    df[df["label_str"].str.lower() == "real"].iloc[0],
    df[df["label_str"].str.lower() == "fake"].iloc[0]
]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, sample in zip(axes, sample_rows):
    local_path = os.path.join(
        "dataset/140k/real_vs_fake/real-vs-fake",
        sample["path"]
    )

    print("\nLabel:", sample["label_str"])
    print("Local image path:", local_path)

    if os.path.exists(local_path):
        image = Image.open(local_path).convert("RGB")

        print("Image size:", image.size)
        print("Image found!")

        ax.imshow(image)
        ax.set_title(sample["label_str"].upper())
        ax.axis("off")
    else:
        print("Image not found:", local_path)
        ax.set_title("IMAGE NOT FOUND")
        ax.axis("off")

plt.tight_layout()
plt.show()
