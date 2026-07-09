from torchvision import transforms

# Standard ImageNet statistics used widely across ResNet/CNN architectures
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


train_transforms = transforms.Compose([
    transforms.ToPILImage(),       # Converts NumPy array safely to PIL image for torch transforms
    transforms.Resize((224, 224)), # Standard ResNet sizing
    transforms.ToTensor(),         # Scales pixels to [0.0, 1.0] and shifts format to [C, H, W]
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
])


val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
])