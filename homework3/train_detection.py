# Segments of code may be written with the aid of AI tools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.datasets.road_dataset import load_data
from homework.models import Detector, save_model
import os

# Settings
DATASET_PATH = os.environ.get("DRIVE_DATASET", "../drive_data")
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_loader = load_data(DATASET_PATH, split="train", transform_pipeline="aug", batch_size=BATCH_SIZE, shuffle=True)
val_loader = load_data(DATASET_PATH, split="val", transform_pipeline="default", batch_size=BATCH_SIZE, shuffle=False)

# Model
model = Detector().to(DEVICE)

# Losses
seg_criterion = nn.CrossEntropyLoss()
depth_criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_seg_loss = 0
    total_depth_loss = 0
    total = 0
    for batch in train_loader:
        x = batch["image"].to(DEVICE)
        seg = batch["track"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        optimizer.zero_grad()
        logits, pred_depth = model(x)
        seg_loss = seg_criterion(logits, seg)
        depth_loss = depth_criterion(pred_depth, depth)
        loss = seg_loss + depth_loss
        loss.backward()
        optimizer.step()
        total_seg_loss += seg_loss.item() * x.size(0)
        total_depth_loss += depth_loss.item() * x.size(0)
        total += x.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} | Seg Loss: {total_seg_loss/total:.4f} | Depth Loss: {total_depth_loss/total:.4f}")

    # Validation
    model.eval()
    val_seg_loss = 0
    val_depth_loss = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(DEVICE)
            seg = batch["track"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)
            logits, pred_depth = model(x)
            seg_loss = seg_criterion(logits, seg)
            depth_loss = depth_criterion(pred_depth, depth)
            val_seg_loss += seg_loss.item() * x.size(0)
            val_depth_loss += depth_loss.item() * x.size(0)
            val_total += x.size(0)
    print(f"  Val Seg Loss: {val_seg_loss/val_total:.4f} | Val Depth Loss: {val_depth_loss/val_total:.4f}")

# Save model
save_model(model)
print("Model saved.")
