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


class BinaryClassifierInference:
    """
    Perform inference using a trained binary classification model.

    The model is loaded from a specific experiment checkpoint.
    """

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint = torch.load(model_path, map_location=self.device)
        self._load_checkpoint_metadata()

        self.class_names = {0: "No Obstacle", 1: "Obstacle"}
        self.transform = self._build_transform()
        self.model = self._load_model()

    # ============================================================
    # CHECKPOINT METADATA
    # ============================================================

    def _load_checkpoint_metadata(self):
        """Extract model architecture and training configuration from the checkpoint."""

        if not isinstance(self.checkpoint, dict):
            raise ValueError("Invalid checkpoint format. Expected a dictionary containing 'state_dict'.")

        if "state_dict" not in self.checkpoint:
            raise KeyError("Checkpoint does not contain 'state_dict'.")

        self.model_name = self.checkpoint.get("architecture")

        if self.model_name is None:
            raise KeyError("Checkpoint does not contain 'architecture'.")

        checkpoint_config = self.checkpoint.get("config", {})
        training_config = checkpoint_config.get("training_parameters", {})
        self.image_size = training_config.get("image_size", 224)

        print("\n--- Model Checkpoint Information ---")
        print(f"Model      : {self.model_name}")
        print(f"Image Size : {self.image_size}")
        print(f"Checkpoint : {self.model_path}")
        print("------------------------------------\n")

    # ============================================================
    # TRANSFORM AND MODEL
    # ============================================================

    def _build_transform(self):
        """Build the validation transform used for inference."""
        _, val_transform = get_data_transforms(img_size=self.image_size)
        return val_transform

    def _load_model(self):
        """Recreate the trained model and load its weights."""

        model = create_model(
            model_name=self.model_name,
            num_classes=1,
            freeze_backbone=False
        )

        model.load_state_dict(self.checkpoint["state_dict"])
        model.to(self.device)
        model.eval()

        return model

    # ============================================================
    # SINGLE IMAGE INFERENCE
    # ============================================================

    def _preprocess_image(self, image_path):
        """Load and preprocess a single image."""

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        return image, image_tensor

    @torch.no_grad()
    def _predict_tensor(self, image_tensor, decision_threshold=0.50):
        """Predict a preprocessed image tensor."""

        logits = self.model(image_tensor).squeeze()
        probability = torch.sigmoid(logits).item()
        raw_logit = logits.item()

        predicted_label = 1 if probability >= decision_threshold else 0
        prediction_name = self.class_names[predicted_label]

        confidence = (probability if predicted_label == 1 else 1 - probability) * 100

        return prediction_name, probability, confidence, raw_logit, predicted_label

    def predict_image(self, image_path, decision_threshold=0.50):
        """Predict a single image."""

        image, image_tensor = self._preprocess_image(image_path)

        pred_name, prob, conf, logit, pred_label = self._predict_tensor(
            image_tensor,
            decision_threshold
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

    # ============================================================
    # BATCH INFERENCE
    # ============================================================

    @torch.no_grad()
    def predict_csv_batch(
        self,
        csv_path,
        batch_size=32,
        num_workers=4,
        decision_threshold=0.50,
        uncertainty_range=(0.40, 0.60)
    ):
        """Run batch inference on images listed in a CSV."""

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        input_df = pd.read_csv(csv_path)

        if "image_path" not in input_df.columns:
            raise KeyError("Input CSV must contain an 'image_path' column.")

        dataset = InferenceDataset(df=input_df, transform=self.transform)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda"
        )

        results = []

        print(f"\n--- Starting Batch Inference on {len(dataset):,} images ---")
        print(
            f"Device: {self.device} | Model: {self.model_name} | "
            f"Batch Size: {batch_size} | Threshold: {decision_threshold}\n"
        )

        for batch_tensors, batch_paths, batch_names, batch_valid in tqdm(
            dataloader,
            desc="Processing Batches"
        ):
            batch_tensors = batch_tensors.to(self.device, non_blocking=True)

            logits = self.model(batch_tensors).view(-1)
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

                confidence = (prob if pred_label == 1 else 1 - prob) * 100
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

    # ============================================================
    # VISUALIZATION
    # ============================================================

    def visualize_predictions_grid(
        self,
        df,
        filter_type="random",
        num_samples=16,
        grid_cols=4,
        figsize=(16, 12)
    ):
        """Visualize a grid of inference predictions."""

        if df.empty:
            print("DataFrame is empty. Nothing to visualize.")
            return

        if filter_type == "uncertain":
            sample_df = df.sort_values("confidence_score").head(num_samples)
            title_prefix = "Most Uncertain / Low-Confidence Predictions"

        elif filter_type == "obstacles":
            sample_df = df[df["pred_label"] == 1].sort_values(
                "raw_prob",
                ascending=False
            ).head(num_samples)

            title_prefix = "Top Confidence Obstacle Predictions"

        elif filter_type == "no_obstacles":
            sample_df = df[df["pred_label"] == 0].sort_values(
                "raw_prob"
            ).head(num_samples)

            title_prefix = "Top Confidence Clear Road Predictions"

        else:
            sample_df = df.sample(n=min(num_samples, len(df)))
            title_prefix = "Random Sample Predictions"

        num_images = len(sample_df)

        if num_images == 0:
            print(f"No images available for the filter: {filter_type}")
            return

        grid_rows = math.ceil(num_images / grid_cols)
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        print(f"\nDisplaying Grid: {title_prefix} ({num_images} images)...")

        for idx, (_, row) in enumerate(sample_df.iterrows()):
            ax = axes[idx]

            try:
                img = Image.open(row["image_path"]).convert("RGB")
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
                f"Conf: {row['confidence_score']:.1f}% | "
                f"Prob: {row['raw_prob']:.2f}\n"
                f"{row['image_name'][:20]}"
            )

            ax.set_title(title_text, color=title_color, fontsize=10, fontweight="bold")
            ax.axis("off")

        for idx in range(num_images, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Batch Inference Inspection — {title_prefix}",
            fontsize=14,
            fontweight="bold",
            y=1.02
        )

        plt.tight_layout()
        plt.show()

