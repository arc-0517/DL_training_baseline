"""Test script for the updated skin dataset loader."""

import sys
sys.path.append('.')

from dataloaders.datasets.my_dataset import get_skin_datasets, LABELS, LABEL_MAP
from torch.utils.data import DataLoader

# Test dataset loading
data_dir = './data'
img_size = 224
augmentation_type = 'mixed'

print("Testing skin dataset loader...")
print(f"Expected labels: {LABELS}")
print(f"Label mapping: {LABEL_MAP}")
print("-" * 50)

# Load datasets
train_dataset, val_dataset = get_skin_datasets(
    data_dir=data_dir,
    img_size=img_size,
    augmentation_type=augmentation_type
)

print(f"\nTraining dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Test loading a batch
print("\nTesting batch loading...")
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Label names: {[LABEL_MAP[label.item()] for label in labels]}")
    break

# Count samples per class
print("\nSamples per class:")
print("Training set:")
train_labels = [label for _, label in train_dataset.samples]
for i, label_name in enumerate(LABELS):
    count = train_labels.count(i)
    print(f"  {label_name}: {count}")

print("\nValidation set:")
val_labels = [label for _, label in val_dataset.samples]
for i, label_name in enumerate(LABELS):
    count = val_labels.count(i)
    print(f"  {label_name}: {count}")

print("\nDataloader test completed successfully!")
