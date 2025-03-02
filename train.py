from load_data import dataloader
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip


device = 'cuda'
LABELS = ["Black", "Blue", "Green", "TTR"] 

train_data = dataloader.GarbageDataset('/home/tejas/Desktop/ENEL 645/Assignment_2/garbage_data', 'CVPR_2024_dataset_Train')
val_data = dataloader.GarbageDataset('/home/tejas/Desktop/ENEL 645/Assignment_2/garbage_data', 'CVPR_2024_dataset_Val')
test_data = dataloader.GarbageDataset('/home/tejas/Desktop/ENEL 645/Assignment_2/garbage_data', 'CVPR_2024_dataset_Test')

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class GarbageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GarbageClassifier, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, image, text):
        # Convert CLIP model outputs to float32
        image_features = self.clip_model.encode_image(image).float()
        text_features = self.clip_model.encode_text(clip.tokenize(text).to(device)).float()
        
        combined_features = image_features + text_features
        return self.classifier(combined_features)


model = GarbageClassifier(num_classes=len(LABELS)).to(device)   

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, texts, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        images = images.to(device).float()  # Ensure images are float32
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return total_loss / len(train_loader), accuracy

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, texts, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device).float()  # Ensure images are float32
            labels = labels.to(device)
            
            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy

num_epochs = 50
patience = 5
best_val_loss = float('inf')
counter = 0

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break