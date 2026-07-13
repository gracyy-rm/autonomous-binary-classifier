import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

# Global Image Normalization Transforms

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
    directly from a parsed CSV metadata file/DataFrame.
    """
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): loaded train or validation DataFrame directly.
                               Must contain 'image_path' and 'label' column.
            root_dir (str): Root directory path of the images on system/Kaggle.
            transform (callable, optional): PyTorch/Torchvision transform pipeline.
        """
        self.image_paths = df["image_path"].values
        self.labels = df["label"].values
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_relative_path = self.image_paths[idx]
        img_absolute_path = os.path.join(self.root_dir, img_relative_path)
        image = Image.open(img_absolute_path).convert('RGB')
        # Convert target labels to Float32 for BCEWithLogitsLoss compatibility
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
            
        return image, label


# Local Pipeline Sanity Check
# This block runs ONLY when you execute `python dataset.py` directly in the terminal.
if __name__ == "__main__":
    print("--- Running Dataset Structure Validation ---")
    
    # 1. Create a tiny mock dictionary representing what your CSV looks like
    mock_metadata = {
        "image_path": ["day_sample.jpg", "night_sample.jpg"],
        "label": [0, 1]  # 0 for Day, 1 for Night
    }
    mock_df = pd.DataFrame(mock_metadata)
    
    # 2. Instantiate our transform pipelines
    train_trans, _ = get_data_transforms(img_size=224)
    
    # 3. Initialize the dataset using the mock DataFrame
    # We pass a dummy root directory '.' since we aren't loading real files here
    dataset = AutonomousBinaryDataset(df=mock_df, root_dir=".", transform=train_trans)
    
    print("✓ Dataset Class structure initialized successfully.")
    print(f"✓ Total items mapped from DataFrame: {len(dataset)}")
    print(f"✓ Target labels set up with float conversion verification.")
    print("\nNext structural step: Create train.py to build the DataLoader and training loop.")