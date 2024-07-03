import torch
import torch.nn as nn
import numpy as np
from dataset import load_dataset
from model import ClsHead, create_model, count_parameters
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os

def train(model, loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
    
    train_loss /= len(loader)
    train_acc = correct / total
    return train_loss, train_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.logits
            loss = criterion(outputs, labels)
            total += labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            valid_loss += loss.item()
    
    valid_loss /= len(loader)
    valid_acc = correct / total
    return valid_loss, valid_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading model...")
    model = create_model().to(device)
    
    print("Total number of trainable parameters:", count_parameters(model))
    
    print("Loading dataset...")
    dataset = load_dataset('annotations.csv')
    
    print("Total number of samples in the dataset:", len(dataset))
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    num_epochs = 10
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()

if __name__ == '__main__':
    main()