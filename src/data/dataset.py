import os
import torch
from torch.utils.data import Dataset
import cvcore as cr 

class AutonomousBinaryDataset(Dataset):
    """
    Custom PyTorch Dataset for loading driving images and labels 
    directly from a filtered pandas DataFrame.
    """
    def __init__(self, X_df, y_series, root_dir, transform=None):
        """
        Args:
            X_df (pd.DataFrame): DataFrame containing the 'image_path' column.
            y_series (pd.Series): Series containing the integer-encoded labels.
            root_dir (str): Root directory path of the images on Kaggle/Disk.
            transform (callable, optional): PyTorch/Torchvision transform pipeline.
        """
        self.image_paths = X_df["image_path"].values
        self.labels = y_series.values
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_relative_path = self.image_paths[idx]
        img_absolute_path = os.path.join(self.root_dir, img_relative_path)
        
        #  Load the image array using cvcore library function
        img_array = cr.load_image(img_absolute_path)
       
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 4. Apply transforms (Resize, ToTensor, Normalization)
        if self.transform:
            img_tensor = self.transform(img_array)
        else:
            # Fallback if no transforms are passed
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img_array)
            
        return img_tensor, label