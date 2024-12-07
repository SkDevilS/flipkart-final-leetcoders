import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# Configuration
DATA_DIR = "train/data/product_images"
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "models/product_detection.pth"

# Dataset Class
class ProductDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = img_path.replace('.jpg', '.txt')  # Label format: x_min, y_min, x_max, y_max, class
        image = Image.open(img_path).convert("RGB")
        labels = np.loadtxt(label_path, delimiter=',') if os.path.exists(label_path) else np.array([])

        if self.transform:
            image = self.transform(image)

        # Prepare targets for Faster R-CNN
        boxes = labels[:, :4] if labels.size else np.zeros((0, 4))
        labels = labels[:, -1].astype(int) if labels.size else np.zeros(0, dtype=int)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, target

# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Data Loaders
dataset = ProductDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model Definition
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(os.listdir(DATA_DIR)) + 1  # Classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# Save Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
