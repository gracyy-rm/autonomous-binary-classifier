import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from PIL import Image
from .dataset import get_data_transforms
from .model import create_model

from cvcore.plot_operations import image_row
from cvcore.image_operations import load_image
from tqdm import tqdm 


class BinaryClassifierInference:
    """
        Perform inference using a trained binary classification model.

        This class is responsible for

        - loading a trained model
        - preprocessing input images
        - predicting single images
        - predicting folders of images
        - visualizing predictions
    """

    def __init__(self,config):
        """
        Initialize the inferencer.

        Parameters
        ----------
        config : dict
           Parsed configuration dictionary.
        """

        # Parse configuration dictionary
        paths = config["paths"]
        model_cfg = config["model"]
        train_cfg = config["training_parameters"]

        # store configuration
        self.model_name = model_cfg["architecture"]
        self.image_size = train_cfg["image_size"]

        self.model_path = os.path.join(
            paths["model_save_dir"],
            f"best_{self.model_name}.pth"
        )
        self.class_names = {
            0: "Day",
            1: "Night"
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = self._build_transform()

        self.model = self._load_model()

    def _build_transform(self):
        """
        Build preprocessing transform for inference.

        Returns
        -------
        torchvision.transforms.Compose
            Validation transform.
        """

        _, val_transform = get_data_transforms(
            img_size=self.image_size
        )

        return val_transform
    
    def _load_model(self):
        """
        Load trained model for inference.

        Returns
        -------
        torch.nn.Module
            Loaded model in evaluation mode.
        """

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        model = create_model(
            model_name=self.model_name,
            num_classes=1,
            freeze_backbone=False
        )

        model.load_state_dict(
            torch.load(
                self.model_path,
                map_location=self.device
            )
        )

        model.to(self.device)

        model.eval()

        return model
    def _preprocess_image(self, image_path):
        """
        Load and preprocess an image.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        tuple
            Original PIL image and preprocessed tensor.
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        return image, image_tensor
    
    @torch.no_grad()
    def _predict_tensor(self, image_tensor):
        """
        Predict a preprocessed image tensor.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple
            Predicted label, probability and confidence.
        """

        logits = self.model(image_tensor)

        probability = torch.sigmoid(logits).item()

        predicted_label = int(probability >= 0.5)

        prediction = self.class_names[predicted_label]

        confidence = (
            probability if predicted_label == 1
            else 1 - probability
        ) * 100

        return prediction, probability, confidence
    
    def predict_image(self, image_path):
        """
        Predict a single image.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        dict
            Prediction results.
        """

        image, image_tensor = self._preprocess_image(image_path)

        prediction, probability, confidence = self._predict_tensor(
            image_tensor
        )

        return {
            "image_path": image_path,
            "image": image,
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
        }
    
    def predict_folder(self, folder_path):
        """
        Predict all images inside a folder.

        Parameters
        ----------
        folder_path : str
            Folder containing images.

        Returns
        -------
        pandas.DataFrame
            Prediction results.
        """

        if not os.path.exists(folder_path):
            raise FileNotFoundError(
                f"Folder not found: {folder_path}"
            )

        supported_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        )

        results = []

        for filename in tqdm(sorted(os.listdir(folder_path)),desc="Running Inference"):
            if not filename.lower().endswith(supported_extensions):
                continue

            image_path = os.path.join(folder_path, filename)

            prediction = self.predict_image(image_path)

            results.append({
                "image_name": filename,
                "prediction": prediction["prediction"],
                "probability": prediction["probability"],
                "confidence": prediction["confidence"],
            })

        return pd.DataFrame(results)
    
    def visualize_prediction(self, prediction_result):
        """
        Visualize prediction.

        Parameters
        ----------
        prediction_result : dict
            Dictionary returned by predict_image().
        """

        title = (
            f"{prediction_result['prediction']}\n"
            f"{prediction_result['confidence']:.2f}%"
        )

        image_row(
            **{
                title: prediction_result["image_path"]
            }
        )