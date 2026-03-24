import argparse
from pathlib import Path
import torch
import torch.utils.data
from .models import MLPPlanner, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPlanner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_func = torch.nn.MSELoss()
    
    train_data = load_data("drive_data/train", batch_size=args.batch_size)
    val_data = load_data("drive_data/val", batch_size=args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_data:
            track_left, track_right, waypoints = batch['track_left'].to(device), batch['track_right'].to(device), batch['waypoints'].to(device)
            pred = model(track_left, track_right)
            loss = loss_func(pred, waypoints)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} complete.")

"""
Usage:
    python3 -m homework.train_planner --model mlp_planner --epochs 30 --batch_size 64 --lr 1e-3
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp_planner', choices=['mlp_planner', 'transformer_planner', 'cnn_planner'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data', type=str, default='../drive_data')
    args = parser.parse_args()

    device = torch.device(args.device)
    # Load data
    train_loader = load_data(args.data, split='train', batch_size=args.batch_size, shuffle=True)
    val_loader = load_data(args.data, split='val', batch_size=args.batch_size, shuffle=False)

    # Model
    model = load_model(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if args.model == 'cnn_planner':
                images = batch['image'].to(device)
                pred = model(images)
            else:
                left = batch['track_left'].to(device)
                right = batch['track_right'].to(device)
                pred = model(left, right)
            target = batch['waypoints'].to(device)
            mask = batch.get('waypoints_mask', torch.ones(target.shape[:-1], dtype=torch.bool)).to(device)
            # Compute loss only on valid waypoints
            loss = criterion(pred, target)
            mask = mask.unsqueeze(-1).expand_as(loss)
            loss = loss[mask].mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * pred.size(0)
            total += pred.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {total_loss/total:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if args.model == 'cnn_planner':
                    images = batch['image'].to(device)
                    pred = model(images)
                else:
                    left = batch['track_left'].to(device)
                    right = batch['track_right'].to(device)
                    pred = model(left, right)
                target = batch['waypoints'].to(device)
                mask = batch.get('waypoints_mask', torch.ones(target.shape[:-1], dtype=torch.bool)).to(device)
                loss = criterion(pred, target)
                mask = mask.unsqueeze(-1).expand_as(loss)
                loss = loss[mask].mean()
                val_loss += loss.item() * pred.size(0)
                val_total += pred.size(0)
        print(f"  Val Loss: {val_loss/val_total:.4f}")

    # Save model
    save_model(model)
    print("Model saved.")

if __name__ == '__main__':
    main()
