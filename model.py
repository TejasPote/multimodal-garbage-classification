import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel
import torch

# MultiModal Classifier model definition
class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the multimodal classifier that takes both image and text as input.
        Args:
            num_classes (int): The number of output classes for classification.
        """
        super(MultiModalClassifier, self).__init__()

        # Vision Encoder (ResNet-50)
        self.resnet = models.resnet50(pretrained=True)  # Pretrained ResNet-50
        self.vision_feature_size = self.resnet.fc.in_features  # Size of ResNet's feature space before the classification head
        self.resnet.fc = nn.Identity()  # Remove classification head, to use the features

        # Text Encoder (DistilBERT)
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')  # Pretrained DistilBERT model
        self.text_fc = nn.Linear(self.bert.config.hidden_size, 512)  # Reduce to 512-d features

        # Fusion Layer: Concatenate image and text features and pass through a fully connected layer
        self.fusion_fc = nn.Linear(self.vision_feature_size + 512, 1024)  # Image + Text feature size
        self.relu = nn.ReLU()  # Non-linearity
        self.dropout = nn.Dropout(0.3)  # Dropout for regularization

        # Final Classification Head
        self.classifier = nn.Linear(1024, num_classes)  # Output class predictions

    def forward(self, input_ids, attention_mask, images):
        """
        Forward pass through the network. Extracts features from text and image inputs, fuses them, and classifies.
        Args:
            input_ids (Tensor): Tokenized input text ids.
            attention_mask (Tensor): Attention mask for text.
            images (Tensor): Image data.
        """
        # Extract features from text (CLS token)
        text_embeds = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        text_features = self.text_fc(text_embeds[:, 0])  # Extract [CLS] token and pass through FC

        # Extract image features
        image_features = self.resnet(images)  # ResNet-50 features

        # Fuse text and image features
        fused_features = torch.cat((image_features, text_features), dim=1)  # Concatenate image and text
        fused_features = self.relu(self.fusion_fc(fused_features))  # Pass through FC layer and ReLU
        fused_features = self.dropout(fused_features)  # Apply dropout

        # Output classification result
        return self.classifier(fused_features)
