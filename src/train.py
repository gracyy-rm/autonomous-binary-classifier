import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from .model import create_model
from .dataset import AutonomousBinaryDataset, get_data_transforms
from sklearn.metrics import precision_score,recall_score,f1_score

def train_one_epoch(model,dataloader,criterion,optimizer,device,epoch,writer,accumulation_steps=4):
    """Runs a single training epoch with Gradient Accumulation and TensorBoard logging."""
    model.train()
    running_loss=0.0
    correct_predictions = 0
    total_samples = 0

    optimizer.zero_grad(set_to_none=True)
    progress_bar = tqdm(dataloader, desc =f"Epoch {epoch}[Train]",leave=False)
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

    all_labels = []
    all_preds=[]

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

        #collect targets and predictions
        all_labels.extend(labels.cpu().squeeze().numpy())
        all_preds.extend(predictions.cpu().squeeze().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Log metrics to TensorBoard
    writer.add_scalar("Loss/Validation", epoch_loss, epoch)
    writer.add_scalar("Accuracy/Validation", epoch_acc * 100, epoch)
    writer.add_scalar("Metrics/Precision", precision, epoch)
    writer.add_scalar("Metrics/Recall", recall, epoch)
    writer.add_scalar("Metrics/F1_Score", f1, epoch)

    return epoch_loss, epoch_acc, precision, recall, f1

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
        transform=train_transform
    )

    val_dataset = AutonomousBinaryDataset(
        df=val_df,
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
    # Loss function with positive class weighting for Obstacle detection
    # pos_weight > 1.0 penalizes missing obstacles (False Negatives) 2x more than false alarms
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer with Differential Learning Rates
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())

    base_lr = train_cfg["learning_rate"] 

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # Gentle tweaks for pre-trained weights (e.g., 1e-5)
        {'params': head_params,     'lr': base_lr}         # Standard LR for new head (e.g., 1e-3)
    ], weight_decay=1e-2)
    print(f"Backbone Params Count: {len(backbone_params)}")
    print(f"Classifier Head Params Count: {len(head_params)}")

    # LR-Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=2,
        min_lr=1e-6,
        threshold=1e-3
    )

    # Training Loop
    best_val_loss = float("inf")
    early_stopping_counter = 0
    early_stopping_patience = train_cfg["early_stopping_patience"]
    min_delta = 1e-4
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

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer
        )

        scheduler.step(val_loss)

        # tensorboard will show the lr chaning over time 
        # Log both backbone and head learning rates
        writer.add_scalar("Learning Rate/Backbone", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Learning Rate/Head", optimizer.param_groups[1]["lr"], epoch)

        print(
            f"Epoch [{epoch}/{train_cfg['epochs']}] | "
            f"Train Loss : {train_loss:.4f} | "
            f"Train Acc : {train_acc*100:.2f}% | "
            f"Val Loss : {val_loss:.4f} | "
            f"Val Acc : {val_acc*100:.2f}% |"
            f"Precision : {val_precision:.3f} | Recall : {val_recall:.3f} | F1 : {val_f1:.3f}"
        )

        if val_loss < best_val_loss-min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0
            save_path = os.path.join(
                paths["model_save_dir"],
                f"best_{model_cfg['architecture']}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

        else:
            early_stopping_counter += 1
            print(
                f"No improvement for "
                f"{early_stopping_counter} epoch(s)."
            )

            if early_stopping_counter >= early_stopping_patience:
                print("\nEarly stopping triggered.")
                break

    writer.close()

    print("\nTraining Complete.")