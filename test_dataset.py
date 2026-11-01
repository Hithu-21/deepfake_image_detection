
from src.dataset import load_datasets

train_ds, valid_ds, test_ds = load_datasets()

print("Datasets loaded successfully!")

for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Pixel range:",
          float(images.numpy().min()),
          float(images.numpy().max()))
    print("First 5 labels:", labels[:5].numpy())
