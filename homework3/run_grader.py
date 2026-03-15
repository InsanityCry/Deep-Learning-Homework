import os
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from homework.datasets.classification_dataset import load_data as load_classification_data
from homework.datasets.road_dataset import load_data as load_road_data
from homework.models import Classifier, Detector, save_model

# Paths
HOMEWORK_DIR = os.path.dirname(os.path.abspath(__file__))
GRADER_CMD = [
    sys.executable, '-m', 'grader', 'homework', '-vv'
]

# Training settings
CLASSIFICATION_DATASET_PATH = os.path.join(HOMEWORK_DIR, '../classification_data')
DRIVE_DATASET_PATH = os.path.join(HOMEWORK_DIR, '../drive_data')

# --- Train Classifier ---
def train_classifier():
    print("Training Classifier...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = load_classification_data(CLASSIFICATION_DATASET_PATH, transform_pipeline="aug", batch_size=128, shuffle=True)
    val_loader = load_classification_data(CLASSIFICATION_DATASET_PATH, transform_pipeline="default", batch_size=128, shuffle=False)
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        print(f"Classifier Epoch {epoch+1}/{epochs} done.")
    save_model(model)
    print("Classifier model saved.\n")

# --- Train Detector ---
def train_detector():
    print("Training Detector...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = load_road_data(DRIVE_DATASET_PATH, split="train", batch_size=32, shuffle=True)
    val_loader = load_road_data(DRIVE_DATASET_PATH, split="val", batch_size=32, shuffle=False)
    model = Detector().to(device)
    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["image"].to(device)
            seg = batch["track"].to(device)
            depth = batch["depth"].to(device)
            optimizer.zero_grad()
            logits, pred_depth = model(x)
            seg_loss = seg_criterion(logits, seg)
            depth_loss = depth_criterion(pred_depth, depth)
            loss = seg_loss + depth_loss
            loss.backward()
            optimizer.step()
        print(f"Detector Epoch {epoch+1}/{epochs} done.")
    save_model(model)
    print("Detector model saved.\n")

def main():
    print("Training models and running grader...\n")
    train_classifier()
    train_detector()
    try:
        result = subprocess.run(GRADER_CMD, cwd=HOMEWORK_DIR, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("[stderr]", result.stderr)
        if result.returncode != 0:
            print(f"Grader exited with code {result.returncode}")
    except Exception as e:
        print(f"Error running grader: {e}")

if __name__ == "__main__":
    main()
