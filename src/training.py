import os
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import create_model
from dataset import AutonomousBinaryDataset, get_data_transforms

def train_one_epoch(model,dataloader,criterion,optimizer,device,epoch,writer,accumulation_steps=4):
    """Runs a single training epoch with Gradient Accumulation and TensorBoard logging."""
    model.train()
    running_loss=0.0
    correct_predictions = 0
    total_samples = 0

    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc =f"Epoch {epoch}[Train]",leave=False)

    for batch_idx, (images,labels) in enumerate(progress_bar):
        images=images.to(device)
        labels=images.to(device).unsqueeze(1)

        logits = model(images)

        loss = criterion(logits,labels)

        loss=loss / accumulation_steps
        loss.backward()

        if(batch_idx + 1) % accumulation_steps ==0 or (batch_idx + 1)==len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += (loss.items() * accumulation_steps )* images.size(0)
        predictions = (logits >= 0.0).float()
        correct_predictions += (predictions==labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = running_loss/total_samples
    epoch_acc = correct_predictions/total_samples

    writer.add_scalar("Loss/Train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/Train", epoch_acc * 100, epoch)
    return epoch_loss,epoch_acc


@torch.no_grad()
def validate(model,dataloader,criterion,device,epoch,writer):
    """Runs a single validation evaluatiion epoch across unseen images"""
    model.eval()
    running_loss =0.0
    correct_predictions = 0
    total_samples  =0

    progress_bar = tqdm(dataloader,desc=f"Epoch {epoch} [Val]",leave=False)
    for images,labels in progress_bar:
        images=images.to(device)
        labels=labels.to(device).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits,labels)

        running_loss += loss.item() + images.size(0)
        predictions =(logits >= 0.0).float()
        correct_predictions += (predictions == labels).sum().item()
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    writer.add_scalar("Loss/Validation", epoch_loss, epoch)
    writer.add_scalar("Accuracy/Validation", epoch_acc * 100, epoch)
    return epoch_loss, epoch_acc


def run_pipeline(config_path="config\config.json"):
    
    # Load and Parse JSON Configuration Document
    print(f"Reading configuration settings profile from: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    paths = config["paths"]
    model_cfg = config["model"]
    train_params = config["training_parameters"]
    hardware = config["hardware"]
   
    os.makedirs(paths["logs"], exist_ok=True)
    os.makedirs(paths["saved_models"], exist_ok=True)
    
    torch.manual_seed(train_params["seed"])

    # Instantiate TensorBoard Logging Stream
    writer = SummaryWriter(log_dir=paths["log_dir"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active runtime hardware execution target: [{device.type.upper()}]")
    
    # Data Infrastructure Setup
    train_df = pd.read_csv(paths["train_csv"])
    valid_df = pd.read_csv(paths["val_csv"])
    
    train_transforms, val_transforms = get_data_transforms(img_size=train_params["image_size"])
    
    train_dataset = AutonomousBinaryDataset(df=train_df, root_dir=paths["image_root"], transform=train_transforms)
    valid_dataset = AutonomousBinaryDataset(df=valid_df, root_dir=paths["image_root"], transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True, 
                              num_workers=hardware["num_workers"], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=train_params["batch_size"], shuffle=False, 
                              num_workers=hardware["num_workers"], pin_memory=True)
    
    
    # Model Optimization Configuration
    print(f"Configuring model engine backbone: {model_cfg['architecture']}...")
    model = create_model(model_name=model_cfg["architecture"], num_classes=1, freeze_backbone=True)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=train_params["learning_rate"])
    
    # Optimization Loop
   
    best_val_loss = float('inf')
    epochs = train_params["epochs"]
    
    print("Beginning model optimization logging sequences...\n")
    for epoch in range(1, epochs + 1):
       
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, accumulation_steps=4
        )
        val_loss, val_acc = validate(
            model, valid_loader, criterion, device, epoch, writer
        )
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} Train Acc: {train_acc*100:.2f}% "
              f"| Val Loss: {val_loss:.4f} Val Acc: {val_acc*100:.2f}%")
        
        # Save model checkpoint using directories parsed from the config file
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(paths["model_save_dir"], f"best_{model_cfg['architecture']}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ Higher validation score recorded. Weights saved to: {save_path}")
  
    writer.close()
    print("\nTraining run fully completed!")

if __name__ == "__main__":
    run_pipeline(config_path="config.json")