import os
import re
from torch.utils.data import Dataset
from PIL import Image
from transformers import DistilBertTokenizer
import torch 

# Custom Dataset for multimodal input (images + text)
class MultiModalDataset(Dataset):
    def __init__(self, root_dir, tokenizer, max_len, transform=None):
        """
        Initialize the dataset with image and text data.
        Args:
            root_dir (str): Path to the root directory containing the image data.
            tokenizer (Tokenizer): DistilBERT tokenizer for text processing.
            max_len (int): Maximum length for the text sequences.
            transform (callable, optional): Transform to apply to the images.
        """
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        self.data = []  # List to hold (image_path, text, label) tuples
        self.label_map = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(root_dir)))}

        # Loop through class names and files in each class directory
        for class_name in self.label_map:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        # Clean text (remove digits, replace underscores with spaces)
                        text = re.sub(r'\d+', '', os.path.splitext(file_name)[0]).replace('_', ' ')
                        self.data.append((file_path, text, self.label_map[class_name]))

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a single sample (image, text, label) at index idx."""
        file_path, text, label = self.data[idx]

        # Load and transform image
        image = Image.open(file_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Tokenize the text using DistilBERT tokenizer
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len, 
            return_token_type_ids=False, padding='max_length', truncation=True, 
            return_attention_mask=True, return_tensors='pt'
        )

        # Return processed data: text (input_ids, attention_mask), image, and label
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }
