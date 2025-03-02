# Multimodal Garbage Classification

## Overview
This project focuses on classifying garbage items into different categories using a multimodal approach. The classification model helps in proper waste disposal by sorting items into predefined bins.

## Features
- Uses a pre-trained CLIP model for classification.
- Custom data preprocessing pipeline to clean and structure data.
- Supports training, validation, and testing datasets.
- Includes SLURM scripts for running on high-performance clusters.

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd multimodal-garbage-classification-master
2. Install the requirements
    ```sh
   pip install -r requirements.txt

4. To train the model
   ```sh
    python train.py
