import os
import io
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .model import create_model
from .dataset import AutonomousBinaryDataset, get_data_transforms
from sklearn.metrics import precision_score,recall_score,f1_score
from torchinfo import summary
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def train_one_epoch(model,dataloader,criterion,optimizer,device,epoch,writer,accumulation_steps=4):
    """Runs a single training epoch with Gradient Accumulation and TensorBoard logging."""
    model.train()
    running_loss=0.0
    correct_predictions = 0
    total_samples = 0
    all_labels_train=[]
    all_preds_train=[]

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

        all_labels_train.extend(labels.cpu().squeeze().numpy())
        all_preds_train.extend(predictions.cpu().squeeze().numpy())

    epoch_loss = running_loss/total_samples
    epoch_acc = correct_predictions/total_samples
    train_precision = precision_score(all_labels_train, all_preds_train, zero_division=0)
    train_recall = recall_score(all_labels_train, all_preds_train, zero_division=0)
    train_f1 = f1_score(all_labels_train, all_preds_train, zero_division=0)

    writer.add_scalar("Loss/Train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/Train", epoch_acc * 100, epoch)
    writer.add_scalar("Metrics/Train_Precision", train_precision, epoch)
    writer.add_scalar("Metrics/Train_Recall", train_recall, epoch)
    writer.add_scalar("Metrics/Train_F1_Score", train_f1, epoch)

    return epoch_loss,epoch_acc,train_precision,train_recall,train_f1



def plot_confusion_matrix_to_tensor(y_true, y_pred, class_names=["Clean", "Obstacle"]):
    """Generates a Confusion Matrix plot and converts it into a PyTorch Image Tensor."""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()

    # Draw figure canvas
    fig.canvas.draw()

    # Convert Matplotlib figure buffer to NumPy array
    image_rgba = np.asarray(fig.canvas.buffer_rgba())
    # Convert RGBA to RGB (drop alpha channel)
    image_rgb = image_rgba[:, :, :3]

    # Transpose to PyTorch format: (Channels, Height, Width)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)

    plt.close(fig)  # Close memory buffer
    return image_tensor


@torch.no_grad()
def validate(model,dataloader,criterion,device,epoch,writer):
    """Runs a single validation evaluatiion epoch across unseen images"""
    model.eval()
    running_loss =0.0
    correct_predictions = 0
    total_samples  =0

    all_labels_val = []
    all_preds_val=[]

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
        all_labels_val.extend(labels.cpu().squeeze().numpy())
        all_preds_val.extend(predictions.cpu().squeeze().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    val_precision = precision_score(all_labels_val, all_preds_val, zero_division=0)
    val_recall = recall_score(all_labels_val, all_preds_val, zero_division=0)
    val_f1 = f1_score(all_labels_val, all_preds_val, zero_division=0)
    
    # Log metrics to TensorBoard
    writer.add_scalar("Loss/Validation", epoch_loss, epoch)
    writer.add_scalar("Accuracy/Validation", epoch_acc * 100, epoch)
    writer.add_scalar("Metrics/Val_Precision", val_precision, epoch)
    writer.add_scalar("Metrics/Val_Recall", val_recall, epoch)
    writer.add_scalar("Metrics/Val_F1_Score", val_f1, epoch)

    # --- LOG CONFUSION MATRIX IMAGE TO TENSORBOARD ---
    cm_image_tensor = plot_confusion_matrix_to_tensor(
        y_true=all_labels_val,
        y_pred=all_preds_val,
        class_names=['No Obstacle', 'Obstacle'],  # Replace with your actual class names
    )
    writer.add_image('Confusion_Matrix/Validation', cm_image_tensor, epoch)
    return epoch_loss, epoch_acc, val_precision, val_recall, val_f1

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
    
    # create a descritive run name with timestaamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_name = model_cfg["architecture"]
    lr=train_cfg["learning_rate"]
    batch_size=train_cfg["batch_size"]
    #FORMAT : "log_dir/resnet50_lr0.001_bs32_20260727_143000"
    run_name= f"{arch_name}_lr{lr}_bs{batch_size}_{timestamp}"
    run_log_dir = os.path.join(paths["log_dir"],run_name)
    # pass the specific run dir t oSummaryWriter
    writer = SummaryWriter(log_dir=run_log_dir)
    print(f"TensorBoard logging to: {run_log_dir}")

    # reproducibility 
    torch.manual_seed(train_cfg["seed"])

    # Create directories
    os.makedirs(paths["log_dir"],exist_ok=True)
    os.makedirs(paths["model_save_dir"],exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on : {device}")

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
    pos_weights_val=train_cfg["pos_weights"]
    pos_weight = torch.tensor([pos_weights_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer with Differential Learning Rates
    base_lr = train_cfg["learning_rate"] 
    weight_decay = train_cfg["weight_decay"]
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=base_lr, weight_decay=weight_decay)

    # LR-Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.1,
        patience=3,
        min_lr=1e-6,
        # threshold=0.1
    )
    # scheduler = CosineAnnealingLR(
    #     optimizer=optimizer,
    #     T_max=train_cfg["epochs"],
    #     eta_min=1e-6
    # )
    # Print Model Architecture Summary
    print("\n--- MODEL ARCHITECTURE SUMMARY ---")
    summary(model, input_size=(train_cfg["batch_size"], 3, train_cfg["image_size"], train_cfg["image_size"]))
    print("-----------------------------------\n")

    # Training Loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_path= None

    early_stopping_counter = 0
    early_stopping_patience = train_cfg["early_stopping_patience"]
    min_delta = 1e-3
    print("\nStarting Training...\n")

    for epoch in range(1, train_cfg["epochs"] + 1):

        prev_lr=optimizer.param_groups[0]['lr']
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
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

        scheduler.step(val_acc)
        # backbone_lr = optimizer.param_groups[0]["lr"]
        current_training_lr = optimizer.param_groups[0]["lr"]
        # tensorboard will show the lr chaning over time 
        # Log both backbone and head learning rates
        # writer.add_scalar("Learning Rate/Backbone", backbone_lr, epoch)
        writer.add_scalar("Learning Rate", current_training_lr, epoch)

        if current_training_lr< prev_lr:
            print(
                f'\n [LR DROP] Epoch {epoch}: Learning rate decreased from '
                f'{prev_lr:.2e} to {current_training_lr:.2e}\n '
            )

        print(
            f"Epoch [{epoch}/{train_cfg['epochs']}] | "
            f"LR : {current_training_lr:.1e} | " 
            f"Train Loss : {train_loss:.4f} | "
            f"Train Acc : {train_acc*100:.2f}% | "
            f"Train Precision : {train_precision:.3f} | Train Recall : {train_recall:.3f} | Train F1 : {train_f1:.3f} | "
            f"Val Loss : {val_loss:.4f} | "
            f"Val Acc : {val_acc*100:.2f}% |"
            f"Val Precision : {val_precision:.3f} | Val Recall : {val_recall:.3f} | Val F1 : {val_f1:.3f}"
            
        )

        if val_acc > best_val_acc + min_delta:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            early_stopping_counter = 0
            save_path = os.path.join(
                paths["model_save_dir"],
                f"best_{model_cfg['architecture']}_acc{val_acc*100:.1f}.pth"
            )

            checkpoint = {
                'architecture' : model_cfg['architecture'],
                'state_dict' : model.state_dict(),
                'val_acc':val_acc,
                'val_loss':val_loss
            }
            torch.save(checkpoint, save_path)
            best_model_path =save_path
            best_val_loss = val_loss
            print(f"Best model saved to {save_path} (Val Acc improved by {improvement:.4f})\n")

        else:
            early_stopping_counter += 1
            remaining_patience = early_stopping_patience-early_stopping_counter
            print(
                f' [PATIENCE WARNING] No val_acc improvement for'
                f' {early_stopping_counter} epoch(s). Strikes:'
                f' {early_stopping_counter}/{early_stopping_patience} '
                f'({remaining_patience} left before early stopping)\n'
            )

            if early_stopping_counter >= early_stopping_patience:
                print("\nEarly stopping triggered.")
                break

    writer.close()

    print("\nTraining Complete.")

    return {
        "best_val_acc" : best_val_acc,
        "best_val_loss" : best_val_loss,
        "best_model_path" : best_model_path,
    }