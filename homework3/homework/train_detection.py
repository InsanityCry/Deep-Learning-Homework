import torch
import torch.nn as nn
import torch.optim as optim
from homework.models import Detector, save_model
import argparse

from homework.datasets.road_dataset import load_data 

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    
    model = Detector().to(device)
    
    # track is typically cross entropy, depth is L1/MSE
    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_loader = load_data('drive_data/train', transform_pipeline='default', batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            imgs = batch['image'].to(device)
            seg_labels = batch['track'].to(device)
            depth_labels = batch['depth'].to(device)
            
            optimizer.zero_grad()
            seg_preds, depth_preds = model(imgs)
            
            loss_seg = seg_criterion(seg_preds, seg_labels)
            loss_depth = depth_criterion(depth_preds, depth_labels)
            loss = loss_seg + loss_depth
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{args.epochs}], Total Loss: {train_loss:.4f}')
        
        # Save model after each epoch
        save_model(model)
        print('Detector Model saved!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()
    
    train(args)
