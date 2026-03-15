# Segments of code may be written with the aid of AI tools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.datasets.classification_dataset import load_data
from homework.models import Classifier, save_model
import os

# Settings
DATASET_PATH = os.environ.get("CLASSIFICATION_DATASET", "../classification_data")
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_loader = load_data(DATASET_PATH, transform_pipeline="aug", batch_size=BATCH_SIZE, shuffle=True)
val_loader = load_data(DATASET_PATH, transform_pipeline="default", batch_size=BATCH_SIZE, shuffle=False)

# Model
model = Classifier().to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    train_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            val_correct += (logits.argmax(dim=1) == y).sum().item()
            val_total += x.size(0)
    val_acc = val_correct / val_total
    print(f"  Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f}")

# Save model
save_model(model)
print("Model saved.")
