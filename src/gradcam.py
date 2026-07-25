import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T


class GradCAM:
    """Generates Grad-CAM heatmaps for CNN-based image classification models."""
    
    def __init__(self, model, target_layer): 
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = None
        self.backward_handle = None

        self._register_hooks()

    def _register_hooks(self): 
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output): 
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output): 
        self.gradients = grad_output[0].detach()

    def _remove_hooks(self): 
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def generate(self, input_tensor):
        self.model.eval()
        torch.set_grad_enabled(True)

        self.activations = None
        self.gradients = None

        output = self.model(input_tensor)
        score = output.squeeze()

        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = weights * self.activations
        cam = cam.sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze(0)
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        height, width = input_tensor.shape[-2:]
        cam = cv2.resize(cam, (width, height))
        return cam
    
    def overlay(self, original_image, heatmap, alpha=0.4):
        if original_image.ndim != 3:
            raise ValueError("original_image must have shape (H, W, 3).")
        if heatmap.ndim != 2:
            raise ValueError("heatmap must have shape (H, W).")

        if original_image.dtype != np.uint8:
            original_image = np.clip(original_image, 0, 1)
            original_image = (original_image * 255).astype(np.uint8)

        heatmap = np.clip(heatmap, 0, 1)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(
            original_image,
            1.0 - alpha,
            heatmap,
            alpha,
            0
        )
        return overlay

    @staticmethod
    def inspect_single_df(df, df_name, model, target_layer, device, num_samples=3):
        """
        Runs Grad-CAM on 'num_samples' randomly selected images from a single DataFrame.
        Static method so it can be called directly using GradCAM.inspect_single_df(...)
        """
        if df.empty:
            print(f"Skipping {df_name}: DataFrame is empty!")
            return

        print("=" * 55)
        print(f"   GRAD-CAM INSPECTION: {df_name.upper()} ({len(df):,} total images)")
        print("=" * 55)

        model = model.to(device)
        model.eval()

        # Preprocessing transforms
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Sample images
        n_samples = min(num_samples, len(df))
        sampled_df = df.sample(n=n_samples, random_state=42)

        for idx, row in sampled_df.iterrows():
            img_path = row["image_path"]
            
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue

            # Load & Preprocess
            pil_img = Image.open(img_path).convert("RGB")
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            orig_img_np = np.array(pil_img.resize((224, 224)))

            # Instantiate GradCAM instance & generate
            grad_cam = GradCAM(model=model, target_layer=target_layer)
            heatmap = grad_cam.generate(input_tensor)
            overlay_result = grad_cam.overlay(orig_img_np, heatmap, alpha=0.4)
            grad_cam._remove_hooks()

            # Metadata
            pred_class = row.get("pred_class_name", "N/A")
            conf_score = row.get("confidence_score", 0.0)

            # Plot 3-panel visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

            axes[0].imshow(orig_img_np)
            axes[0].set_title(f"Original Image\nPred: {pred_class}", fontsize=10)
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title(f"Grad-CAM Heatmap\nConfidence: {conf_score:.1f}%", fontsize=10)
            axes[1].axis("off")

            axes[2].imshow(overlay_result)
            axes[2].set_title("Overlay", fontsize=10)
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()