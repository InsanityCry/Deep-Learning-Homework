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

    save_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args)
