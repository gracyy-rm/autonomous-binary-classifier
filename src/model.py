# models.py
import torch
import torch.nn as nn
import torchvision.models as models

class AutonomousClassifier(nn.Module):
    """
    A modular wrapper class that encapsulates a pre-trained computer vision 
    backbone and attaches a custom classification head.
    """
    def __init__(self, backbone, in_features, num_classes=1):
        super(AutonomousClassifier, self).__init__()
        self.backbone = backbone

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),  
            nn.Linear(256, num_classes)
        )
            
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def create_model(model_name='resnet18', num_classes=1, freeze_backbone=True):
    """
    Factory function to instantiate, configure, and prepare pre-trained 
    models for Transfer Learning.
    """
    # Fixed typo from 'resent18' to 'resnet18'
    if model_name == 'resnet18':
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif model_name == 'mobilenet_v3':
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
    elif model_name == 'efficientnet_b0':
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Backbone '{model_name}' is not recognized or supported.")
    
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        for param in backbone.parameters():
            param.requires_grad = True

    model = AutonomousClassifier(
        backbone=backbone,
        in_features=in_features,
        num_classes=num_classes
    )

    return model

# Dedented execution block
if __name__ == "__main__":
    print("---Running Component Architecture Validation---")
    model_binary = create_model(model_name='resnet18', num_classes=1, freeze_backbone=True)
    is_backbone_frozen = not next(model_binary.backbone.parameters()).requires_grad
    is_head_trainable = next(model_binary.classifier.parameters()).requires_grad
    print(f"\n[Binary Configuration]")
    print(f"-> Architecture initialization: SUCCESS")
    print(f"-> Is Backbone Feature Extractor Frozen?: {is_backbone_frozen}")
    print(f"-> Is Custom Classification Head Trainable?: {is_head_trainable}")
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output_binary = model_binary(dummy_input)
    print(f"-> Input image tensor dimensions: {dummy_input.shape}")
    print(f"-> Output binary logits dimensions: {output_binary.shape}")