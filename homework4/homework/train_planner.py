# Segments of code may be written with the aid of AI tools
import argparse
import torch
import random
import torch.nn.functional as F
from .models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from .datasets.road_dataset import load_data


def planner_loss(pred, labels, labels_mask, lateral_weight: float = 4.0):
    coord_weights = pred.new_tensor([1.0, lateral_weight]).view(1, 1, 2)
    valid_mask = labels_mask[..., None].float()

    # Smooth L1 is less sensitive to occasional noisy waypoints than plain MSE.
    loss = F.smooth_l1_loss(pred, labels, reduction='none')
    loss = loss * coord_weights * valid_mask

    normalizer = valid_mask.sum() * pred.shape[-1]
    return loss.sum() / normalizer.clamp_min(1.0)

def augment_batch(batch, model_type):
    if random.random() > 0.5:
        batch['waypoints'][..., 0] *= -1
        if model_type == 'cnn_planner':
            batch['image'] = torch.flip(batch['image'], [3])
        else:
            left, right = batch['track_left'].clone(), batch['track_right'].clone()
            batch['track_left'], batch['track_right'] = right, left
            batch['track_left'][..., 0] *= -1
            batch['track_right'][..., 0] *= -1
    return batch

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training {args.model} on {device}')

    if args.model == 'mlp_planner':
        model = MLPPlanner().to(device)
    elif args.model == 'transformer_planner':
        model = TransformerPlanner().to(device)
    elif args.model == 'cnn_planner':
        model = CNNPlanner().to(device)

    if args.model == 'mlp_planner':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    train_data = load_data('drive_data/train', batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_data:
            batch = augment_batch(batch, args.model)
            labels = batch['waypoints'].to(device)
            labels_mask = batch['waypoints_mask'].to(device)

            if args.model == 'cnn_planner':
                inputs = batch['image'].to(device)
                pred = model(inputs)
            else:
                t_left, t_right = batch['track_left'].to(device), batch['track_right'].to(device)
                pred = model(t_left, t_right)

            lateral_weight = 5.0 if args.model == 'mlp_planner' else 6.0 if args.model == 'transformer_planner' else 2.0
            loss = planner_loss(pred, labels, labels_mask, lateral_weight=lateral_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_data):.4f}')

    save_model(model)
    print(f'{args.model} saved.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['mlp_planner', 'transformer_planner', 'cnn_planner'], default='mlp_planner')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 1e-6 if args.model == 'transformer_planner' else 1e-3
    train(args)
