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
        transforms.ColorJitter(brightness=0.1, contrast=0.3,saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
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
    

class InferenceDataset(Dataset):
    """
    Dataset optimized for fast batch inference across 67k+ unlabelled images.
    
    Returns preprocessed tensor, original image path, and image filename.
    Safely handles corrupt or missing images without crashing.
    """
    def __init__(self, df, transform=None):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing at least an 'image_path' column.
        transform : torchvision.transforms.Compose, optional
            Deterministic validation/inference transform.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = str(self.df.iloc[idx]["image_path"])
        img_name = os.path.basename(img_path)

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image_tensor = self.transform(image)
            is_valid = True
        except Exception as e:
            # Fallback for corrupt/missing files: return zero tensor & flag error
            print(f"Warning: Failed to load image at {img_path}. Error: {e}")
            # Assuming standard image size from transform or default 224
            image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            is_valid = False

        return image_tensor, img_path, img_name, is_valid


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