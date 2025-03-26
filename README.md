# Multimodal Garbage Classification

## Project Overview
This project focuses on classifying garbage items into four categories (Black, Blue, Green, and TTR) using a multimodal deep learning approach. The model combines image and text information extracted from filenames to enhance classification accuracy. The system utilizes a vision encoder (ResNet-50) and a text encoder (DistilBERT), fusing their features before passing them through a classification layer.

## Methodology
The approach involves:
1. **Data Preprocessing**: Cleaning text from filenames and applying image transformations.
2. **Multimodal Model**: A neural network integrating a ResNet-50 for image encoding and DistilBERT for text encoding.
3. **Training**: Fine-tuning the model using cross-entropy loss and the AdamW optimizer.
4. **Evaluation**: Performance assessment using accuracy, confusion matrices, and ROC-AUC curves.
5. **Analysis**: Examining incorrect predictions and model biases.

## Repository Structure
```
multimodal-garbage-classification/
│── data.py               # Dataset handling and preprocessing
│── model.py              # Model architecture definition
│── train.py              # Training script
│── requirements.txt      # List of dependencies
│── README.md             # Project documentation
```

## Steps to Reproduce

### 1. Set Up the Environment
Ensure you have Python 3.8+ and create a virtual environment:
```bash
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Structure your dataset as follows:
```
dataset/
│── Train/
│   ├── black/
│   ├── blue/
│   ├── green/
│   ├── ttr/
│── Val/
│   ├── black/
│   ├── blue/
│   ├── green/
│   ├── ttr/
│── Test/
│   ├── black/
│   ├── blue/
│   ├── green/
│   ├── ttr/
```

### 3. Train the Model
Run the training script with the required arguments:
```bash
python train.py --train_dir dataset/Train --val_dir dataset/Val \
                --epochs 10 --batch_size 32 --learning_rate 2e-5 \
                --checkpoint_dir checkpoints/
```

