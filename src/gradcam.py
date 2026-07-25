import cv2
import torch
import numpy as np


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