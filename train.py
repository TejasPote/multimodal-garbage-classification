import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import MultiModalClassifier
from data import MultiModalDataset
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms
import argparse
import os

# Argument Parser
def parse_args():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description="Train a multimodal classifier for garbage classification.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset.")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation dataset.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset.")  # Add test directory
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--max_len", type=int, default=32, help="Maximum token length for text encoding.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save the best model checkpoint.")
    return parser.parse_args()

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoint_dir):
    """
    Train the multimodal model for the specified number of epochs.
    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., AdamW).
        device (torch.device): The device to run the model on (CPU or GPU).
        epochs (int): Number of epochs to train.
        checkpoint_dir (str): Directory to save the best model checkpoint.
    """
    best_acc = 0.0  # Variable to track the best accuracy
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the checkpoint directory exists

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss, correct, total = 0, 0, 0

        # Training loop
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for batch in tqdm_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()  # Track accuracy
            total += labels.size(0)

            # Update progress bar
            tqdm_bar.set_postfix(loss=total_loss / (total + 1), acc=correct / (total + 1))

        # Calculate training accuracy and validation accuracy
        train_acc = correct / total
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

        # Validate model on the validation set
        val_acc = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | Validation Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"Best model saved with validation accuracy: {best_acc:.4f}")

# Validation Function
def validate_model(model, val_loader, criterion, device):
    """
    Validate the model after each epoch.
    Args:
        model (nn.Module): The model to be validated.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): The device to run the model on (CPU or GPU).
    """
    model.eval()  # Set model to evaluation mode
    total_loss, correct, total = 0, 0, 0

    # Validation loop
    tqdm_bar = tqdm(val_loader, desc="[Validation]")
    with torch.no_grad():
        for batch in tqdm_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            tqdm_bar.set_postfix(loss=total_loss / (total + 1), acc=correct / (total + 1))

    val_acc = correct / total
    print(f"Validation Loss: {total_loss/len(val_loader):.4f} | Validation Acc: {val_acc:.4f}")
    return val_acc

# Test Function
def test_model(model, test_loader, criterion, device, checkpoint_dir):
    """
    Test the model after training by loading the best model checkpoint.
    Args:
        model (nn.Module): The model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): The device to run the model on (CPU or GPU).
        checkpoint_dir (str): Directory where the best model checkpoint is saved.
    """
    # Load the best model checkpoint
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth")))
    model.eval()  # Set model to evaluation mode

    total_loss, correct, total = 0, 0, 0
    tqdm_bar = tqdm(test_loader, desc="[Testing]")
    with torch.no_grad():
        for batch in tqdm_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            tqdm_bar.set_postfix(loss=total_loss / (total + 1), acc=correct / (total + 1))

    test_acc = correct / total
    print(f"Test Loss: {total_loss/len(test_loader):.4f} | Test Acc: {test_acc:.4f}")
    return test_acc

# Main Function
def main():
    """
    Main function to initialize components, load data, and start the training process.
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define image transformations for data preprocessing
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization
    ])

    # Load tokenizer for text processing
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Initialize training, validation, and test datasets
    train_dataset = MultiModalDataset(args.train_dir, tokenizer, args.max_len, transform=image_transforms)
    val_dataset = MultiModalDataset(args.val_dir, tokenizer, args.max_len, transform=image_transforms)
    test_dataset = MultiModalDataset(args.test_dir, tokenizer, args.max_len, transform=image_transforms)

    # DataLoaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize the model, loss function, and optimizer
    model = MultiModalClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Start the training process
    train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.checkpoint_dir)

    # After training, test the best model
    test_acc = test_model(model, test_loader, criterion, device, args.checkpoint_dir)
   
if __name__ == "__main__":
    main()