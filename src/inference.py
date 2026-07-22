import os
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import get_data_transforms, InferenceDataset
from .model import create_model

from cvcore.plot_operations import image_row
from cvcore.image_operations import load_image

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
            0: "No Obstacle",
            1: "Obstacle"
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
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
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
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image, image_tensor
    
    @torch.no_grad()
    def _predict_tensor(self, image_tensor,decision_threshold=0.50):
        """
        Predict a preprocessed image tensor.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Preprocessed input tensor of shape [1, 3, H, W].
        decision_threshold : float, optional
            Probability threshold to classify as Obstacle (default: 0.50).

        Returns
        -------
        tuple
            Predicted string label, raw probability, confidence percentage, and raw logit.
        """

        logits = self.model(image_tensor).squeeze(-1)
        probability = torch.sigmoid(logits).item()
        raw_logit = logits.item()

        predicted_label = 1 if probability >= decision_threshold else 0
        prediction_name = self.class_names[predicted_label]

        confidence = (
            probability if predicted_label == 1
            else 1 - probability
        ) * 100

        return prediction_name, probability, confidence, raw_logit, predicted_label
    
    def predict_image(self, image_path, decision_threshold=0.50):
        """
        Predict a single image.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        decision_threshold : float, optional
            Probability threshold to flag as Obstacle.

        Returns
        -------
        dict
            Prediction results.
        """

        image, image_tensor = self._preprocess_image(image_path)

        pred_name, prob, conf, logit, pred_label = self._predict_tensor(
            image_tensor, 
            decision_threshold=decision_threshold
        )

        return {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "image": image,
            "pred_class_name": pred_name,
            "pred_label": pred_label,
            "raw_prob": round(prob, 4),
            "confidence_score": round(conf, 2),
            "raw_logit": round(logit, 4)
        }
    
    def predict_csv_batch(
            self, 
            csv_path,
            batch_size=128,
            num_workers=4,
            decision_threshold=0.50,
            uncertainty_range=(0.40, 0.60)):
        """
        Run high-throughout batch inference across a CSV containing image paths.
        """

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Folder not found: {csv_path}"
            )
        input_df = pd.read_csv(csv_path)
        if "image_path" not in input_df.columns:
            raise KeyError("Input CSV must contain an 'image_path'.")
        
        dataset = InferenceDataset(
            df=input_df,
            transform=self.transform,
            image_size=self.image_size
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True  if self.device.type=="cuda" else False
        )

        results = []
        print(f"\n--- Starting Batch Inference on {len(dataset):,} images ---")
        print(f"Device: {self.device} | Batch Size: {batch_size} | Decision Threshold: {decision_threshold}\n")

        for batch_tensors, batch_paths, batch_names, batch_valid in tqdm(dataloader, desc="Processing Batches"):
            batch_tensors = batch_tensors.to(self.device, non_blocking=True)

            logits = self.model(batch_tensors).squeeze(-1)
            probabilities = torch.sigmoid(logits)

            logits_np = logits.cpu().numpy()
            probs_np = probabilities.cpu().numpy()

            for i in range(len(batch_paths)):
                if not batch_valid[i]:
                    continue

                prob = float(probs_np[i])
                logit = float(logits_np[i])

                pred_label = 1 if prob >= decision_threshold else 0
                pred_class_name = self.class_names[pred_label]

                confidence = (prob if pred_label == 1 else (1.0 - prob)) * 100.0
                is_uncertain = uncertainty_range[0] <= prob <= uncertainty_range[1]

                results.append({
                    "image_path": batch_paths[i],
                    "image_name": batch_names[i],
                    "raw_logit": round(logit, 4),
                    "raw_prob": round(prob, 4),
                    "pred_label": pred_label,
                    "pred_class_name": pred_class_name,
                    "confidence_score": round(confidence, 2),
                    "is_uncertain": is_uncertain
                })

        output_df = pd.DataFrame(results)
        print(f"\nInference Complete! Processed {len(output_df):,} valid images.")
        return output_df
    

    def visualize_predictions_grid(
        self,
        df,
        filter_type="random",
        num_samples=16,
        grid_cols=4,
        figsize=(16, 12)
    ):
        """Visualize grid of predictions with color-coded title banners."""
        if df.empty:
            print("DataFrame is empty. Nothing to visualize.")
            return

        if filter_type == "uncertain":
            sample_df = df.sort_values(by="confidence_score", ascending=True).head(num_samples)
            title_prefix = "Most Uncertain / Low-Confidence Predictions"
        elif filter_type == "obstacles":
            sample_df = df[df["pred_label"] == 1].sort_values(by="raw_prob", ascending=False).head(num_samples)
            title_prefix = "Top Confidence Obstacle Predictions"
        elif filter_type == "no_obstacles":
            sample_df = df[df["pred_label"] == 0].sort_values(by="raw_prob", ascending=True).head(num_samples)
            title_prefix = "Top Confidence Clear Road Predictions"
        else:
            sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
            title_prefix = "Random Sample Predictions"

        num_images = len(sample_df)
        grid_rows = math.ceil(num_images / grid_cols)

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        print(f"\nDisplaying Grid: {title_prefix} ({num_images} images)...")

        for idx, (_, row) in enumerate(sample_df.iterrows()):
            ax = axes[idx]
            img_path = row["image_path"]

            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Failed to Load\n{e}", ha="center", va="center")

            if row["is_uncertain"]:
                title_color = "darkgoldenrod"
            elif row["pred_label"] == 1:
                title_color = "crimson"
            else:
                title_color = "darkgreen"

            title_text = (
                f"[{row['pred_class_name']}]\n"
                f"Conf: {row['confidence_score']:.1f}% | Prob: {row['raw_prob']:.2f}\n"
                f"{row['image_name'][:20]}"
            )

            ax.set_title(title_text, color=title_color, fontsize=10, fontweight="bold")
            ax.axis("off")

        for idx in range(num_images, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(f"Batch Inference Inspection — {title_prefix}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.show()