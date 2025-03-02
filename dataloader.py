import os
import re
import json
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import clip
from tqdm import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
_, preprocess = clip.load("ViT-B/32", device=device, jit=False)


# Define paths
DATA_DIR = "garbage_data"
LABELS = ["Black", "Blue", "Green", "TTR"]  # Class labels
SPLITS = ["CVPR_2024_dataset_Train", "CVPR_2024_dataset_Val", "CVPR_2024_dataset_Test"]  # Dataset splits

# Function to clean text (remove numbers)
def clean_text(filename):
    filename = filename.lower().replace("_", " ")  # Convert to lowercase, replace underscores
    filename = re.sub(r'\d+', '', filename)  # Remove numbers
    filename = filename.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")  # Remove extensions
    return filename.strip()

class GarbageDataset(Dataset):
    def __init__(self, root_dir, split):
        self.image_paths = []
        self.texts = []
        self.labels = []
        # self.transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # Resize to CLIP's expected input size
        # transforms.ToTensor(),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])
        # Iterate through dataset splits (Train, Val, Test)
        
        split_dir = os.path.join(root_dir, split)

        # Iterate through class subfolders
        for label in LABELS:
            class_dir = os.path.join(split_dir, label)
            if not os.path.exists(class_dir):
                continue  # Skip if the subfolder is missing
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                
                # Extract and clean text description
                description = clean_text(img_name)

                # Store data
                self.image_paths.append(img_path)
                self.texts.append(description)
                self.labels.append(LABELS.index(label))  # Convert label to index

        # Tokenize texts using CLIP tokenizer
        # self.text_tokens = clip.tokenize(self.texts)
        self.text_tokens = self.texts

        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_paths[idx]).convert("RGB"))  # Preprocess image
        # image = self.transform(image)
        text = self.text_tokens[idx]  # Tokenized text
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert label to tensor
        return image, text, label

