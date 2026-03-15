import torch
import torch.nn as nn
import torch.optim as optim
from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data
import argparse

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    
    # Initialize model, loss, and optimizer
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load data
    train_loader = load_data('classification_data/train', transform_pipeline='default', batch_size=args.batch_size, shuffle=True)
    val_loader = load_data('classification_data/val', transform_pipeline='default', batch_size=args.batch_size, shuffle=False)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss:.4f}')
        
        # Save model after each epoch
        save_model(model)
        print('Model saved!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()
    
    train(args)
