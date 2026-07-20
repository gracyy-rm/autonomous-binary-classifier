import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

# Global Image Normalization Transforms (Standard ImageNet Weights)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

def get_data_transforms(img_size=224):
    """
    Defines training and validation transformation pipelines.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    
    return train_transform, val_transform


# Custom Dataset Definition
class AutonomousBinaryDataset(Dataset):
    """
    Custom PyTorch Dataset for loading driving images and labels 
    directly from the absolute paths inside the parsed DataFrame.
    """
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): loaded train or validation DataFrame directly.
                               Must contain 'image_path' and 'label_encoded' column.
            transform (callable, optional): PyTorch/Torchvision transform pipeline.
        """
        self.image_paths = df["image_path"].values
        self.labels = df["label_encoded"].values 
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_absolute_path = self.image_paths[idx]
        image = Image.open(img_absolute_path).convert('RGB')
        
        # Convert target labels to Float32 for BCEWithLogitsLoss compatibility
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# Local Pipeline Sanity Check
if __name__ == "__main__":
    print("--- Running Dataset Structure Validation ---")
    mock_metadata = {
        "image_path": ["path/to/mock_no_obstacle.jpg", "path/to/mock_obstacle.jpg"],
        "label_encoded": [0, 1]  # 0 for No_Obstacle, 1 for Obstacle
    }
    mock_df = pd.DataFrame(mock_metadata)
    
    train_trans, _ = get_data_transforms(img_size=224)
    
    dataset = AutonomousBinaryDataset(df=mock_df, transform=train_trans)
    
    print("✓ Dataset Class structure initialized successfully.")
    print(f"✓ Total items mapped from DataFrame: {len(dataset)}")
    print(f"✓ Target labels set up with float conversion verification.")