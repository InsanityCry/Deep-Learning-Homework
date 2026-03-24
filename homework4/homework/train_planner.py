import argparse
import torch
from .models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from .datasets.road_dataset import load_data

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {args.model} on {device}")

    if args.model == 'mlp_planner':
        model = MLPPlanner().to(device)
    elif args.model == 'transformer_planner':
        model = TransformerPlanner().to(device)
    elif args.model == 'cnn_planner':
        model = CNNPlanner().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_func = torch.nn.MSELoss()
    
    # CNN requires images, others require track points
    train_data = load_data("drive_data/train", batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_data:
            labels = batch['waypoints'].to(device)
            
            if args.model == 'cnn_planner':
                inputs = batch['image'].to(device)
                pred = model(inputs)
            else:
                t_left, t_right = batch['track_left'].to(device), batch['track_right'].to(device)
                pred = model(t_left, t_right)
            
            loss = loss_func(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_data):.4f}")

    save_model(model)
    print(f"{args.model} saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['mlp_planner', 'transformer_planner', 'cnn_planner'], default='mlp_planner')
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train(args)
