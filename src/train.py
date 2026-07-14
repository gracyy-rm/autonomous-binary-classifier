import os
import torch
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

    optimizer.zero_grad(set_to_none=True)
    progress_bar = tqdm(dataloader, desc =f"Epoch {epoch}[Train]",leave=False)
    progress_bar.set_postfix(loss=loss.item()*accumulation_steps)
    for batch_idx, (images,labels) in enumerate(progress_bar):
        images=images.to(device)
        labels=labels.to(device).unsqueeze(1)

        logits = model(images)

        loss = criterion(logits,labels)

        loss=loss / accumulation_steps
        loss.backward()

        if(batch_idx + 1) % accumulation_steps ==0 or (batch_idx + 1)==len(dataloader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += (loss.item() * accumulation_steps )* images.size(0)
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

        running_loss += loss.item() * images.size(0)
        predictions =(logits >= 0.0).float()
        correct_predictions += (predictions == labels).sum().item()
        total_samples += images.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    writer.add_scalar("Loss/Validation", epoch_loss, epoch)
    writer.add_scalar("Accuracy/Validation", epoch_acc * 100, epoch)
    return epoch_loss, epoch_acc


def run_pipeline(config):
    """
    Execute the complete training pipeline.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    """
    # parse configuration
    paths= config["paths"]
    model_cfg = config["model"]
    train_cfg = config["training_parameters"]
    hardware_cfg = config["hardware"]

    # reproducibility 
    torch.manual_seed(train_cfg["seed"])

    # Create directories
    os.makedirs(paths["log_dir"],exist_ok=True)
    os.makedirs(paths["model_save_dir"],exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on : {device}")

    # TensorBoard

    writer = SummaryWriter(log_dir=paths["log_dir"])

    #load datafraames
    train_df = pd.read_csv(paths["train_csv"])
    val_df = pd.read_csv(paths["val_csv"])

    #data transforms
    train_transform,val_transform = get_data_transforms(img_size=train_cfg["image_size"])

    # dataset 
    train_dataset = AutonomousBinaryDataset(
        df=train_df,
        root_dir=paths["image_root"],
        transform=train_transform
    )

    val_dataset = AutonomousBinaryDataset(
        df=val_df,
        root_dir=paths["image_root"],
        transform=val_transform
    )

    # dataloader

    train_loader  = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=hardware_cfg["num_workers"],
        pin_memory=device.type == "cuda"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=hardware_cfg["num_workers"],
        pin_memory=device.type == "cuda"
    )

    #MOdel
    model=create_model(
        model_name=model_cfg["architecture"],
        num_classes=1,
        freeze_backbone=model_cfg["freeze_backbone"]
    ).to(device)

    # loss function 
    criterion = nn.BCEWithLogitsLoss()

    #optimiser
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"]
    )

    # Training Loop
    best_val_loss = float("inf")

    print("\nStarting Training...\n")

    for epoch in range(1, train_cfg["epochs"] + 1):

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            accumulation_steps=train_cfg["accumulation_steps"]
        )

        val_loss, val_acc = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer
        )

        print(
            f"Epoch [{epoch}/{train_cfg['epochs']}] | "
            f"Train Loss : {train_loss:.4f} | "
            f"Train Acc : {train_acc*100:.2f}% | "
            f"Val Loss : {val_loss:.4f} | "
            f"Val Acc : {val_acc*100:.2f}%"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_path = os.path.join(
                paths["model_save_dir"],
                f"best_{model_cfg['architecture']}.pth"
            )

            torch.save(model.state_dict(), save_path)

            print(f"Best model saved to {save_path}")

    writer.close()

    print("\nTraining Complete.")