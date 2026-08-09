import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import create_model
from .dataset import AutonomousBinaryDataset, get_data_transforms

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, confusion_matrix
)

from torchinfo import summary

# TRAINING

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, accumulation_steps=4):
    """Runs a single training epoch with gradient accumulation and TensorBoard logging."""

    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels_train, all_preds_train = [], []

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels)

        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += (loss.item() * accumulation_steps) * images.size(0)

        predictions = (logits >= 0.0).float()
        correct_predictions += (predictions == labels).sum().item()
        total_samples += images.size(0)

        all_labels_train.extend(labels.cpu().squeeze().numpy())
        all_preds_train.extend(predictions.cpu().squeeze().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    train_precision = precision_score(all_labels_train, all_preds_train, zero_division=0)
    train_recall = recall_score(all_labels_train, all_preds_train, zero_division=0)
    train_f1 = f1_score(all_labels_train, all_preds_train, zero_division=0)

    writer.add_scalar("Loss/Train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/Train", epoch_acc * 100, epoch)
    writer.add_scalar("Metrics/Train_Precision", train_precision, epoch)
    writer.add_scalar("Metrics/Train_Recall", train_recall, epoch)
    writer.add_scalar("Metrics/Train_F1_Score", train_f1, epoch)

    return epoch_loss, epoch_acc, train_precision, train_recall, train_f1

# CONFUSION MATRIX

def plot_confusion_matrix_to_tensor(y_true, y_pred, class_names=None):
    """Generates a confusion matrix plot and converts it into a PyTorch image tensor."""

    if class_names is None:
        class_names = ["No Obstacle", "Obstacle"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

    plt.title("Validation Confusion Matrix")
    plt.tight_layout()

    fig.canvas.draw()
    image_rgba = np.asarray(fig.canvas.buffer_rgba())
    image_rgb = image_rgba[:, :, :3]
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)

    plt.close(fig)
    return image_tensor

# VALIDATION

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer):
    """Runs a single validation epoch across unseen images."""

    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels_val, all_preds_val = [], []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        predictions = (logits >= 0.0).float()
        correct_predictions += (predictions == labels).sum().item()
        total_samples += images.size(0)

        all_labels_val.extend(labels.cpu().squeeze().numpy())
        all_preds_val.extend(predictions.cpu().squeeze().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    val_precision = precision_score(all_labels_val, all_preds_val, zero_division=0)
    val_recall = recall_score(all_labels_val, all_preds_val, zero_division=0)
    val_f1 = f1_score(all_labels_val, all_preds_val, zero_division=0)

    writer.add_scalar("Loss/Validation", epoch_loss, epoch)
    writer.add_scalar("Accuracy/Validation", epoch_acc * 100, epoch)
    writer.add_scalar("Metrics/Val_Precision", val_precision, epoch)
    writer.add_scalar("Metrics/Val_Recall", val_recall, epoch)
    writer.add_scalar("Metrics/Val_F1_Score", val_f1, epoch)

    cm_image_tensor = plot_confusion_matrix_to_tensor(
        all_labels_val,
        all_preds_val,
        ["No Obstacle", "Obstacle"]
    )

    writer.add_image("Confusion_Matrix/Validation", cm_image_tensor, epoch)

    return epoch_loss, epoch_acc, val_precision, val_recall, val_f1


# CHECKPOINT

def create_checkpoint(model, config, epoch, train_loss, val_loss, train_acc, val_acc,
                      train_precision, val_precision, train_recall, val_recall,
                      train_f1, val_f1):
    """Creates a self-contained checkpoint for the best model."""

    return {
        "architecture": config["model"]["architecture"],
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "config": config,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_precision": train_precision,
        "val_precision": val_precision,
        "train_recall": train_recall,
        "val_recall": val_recall,
        "train_f1": train_f1,
        "val_f1": val_f1
    }


# MAIN TRAINING PIPELINE

def run_pipeline(config):
    """
    Execute the complete training pipeline.

    Each run creates:
        config.json
        history.json
        summary.json
        best_model.pth
        tensorboard/
    """

    paths = config["paths"]
    model_cfg = config["model"]
    train_cfg = config["training_parameters"]
    hardware_cfg = config["hardware"]

    # Create unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_name = model_cfg["architecture"]
    lr = train_cfg["learning_rate"]
    batch_size = train_cfg["batch_size"]

    run_name = f"{arch_name}_lr{lr}_bs{batch_size}_{timestamp}"

    # Create experiment directories
    experiments_dir = paths["experiments_dir"]
    run_dir = os.path.join(experiments_dir, run_name)
    tensorboard_dir = os.path.join(run_dir, "tensorboard")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Save configuration used for this run
    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Experiment directory: {run_dir}")
    print(f"Configuration saved to: {config_path}")

    # TensorBoard
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard logging to: {tensorboard_dir}")

    # Reproducibility
    seed = train_cfg["seed"]
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Load data
    train_df = pd.read_csv(paths["train_csv"])
    val_df = pd.read_csv(paths["val_csv"])

    print(f"Training samples  : {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")

    # Transforms
    train_transform, val_transform = get_data_transforms(
        img_size=train_cfg["image_size"]
    )

    # Datasets
    train_dataset = AutonomousBinaryDataset(
        df=train_df,
        transform=train_transform
    )

    val_dataset = AutonomousBinaryDataset(
        df=val_df,
        transform=val_transform
    )

    # Dataloaders
    train_loader = DataLoader(
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

    # Model
    model = create_model(
        model_name=model_cfg["architecture"],
        num_classes=1,
        freeze_backbone=model_cfg["freeze_backbone"]
    ).to(device)

    # Loss
    pos_weight = torch.tensor(
        [train_cfg["pos_weights"]],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.1,
        patience=3,
        min_lr=1e-6
    )

    # Model summary
    print("\n--- MODEL ARCHITECTURE SUMMARY ---")

    summary(
        model,
        input_size=(
            train_cfg["batch_size"],
            3,
            train_cfg["image_size"],
            train_cfg["image_size"]
        )
    )

    print("-----------------------------------\n")

    # Training state
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_path = None
    best_epoch = None

    early_stopping_counter = 0
    early_stopping_patience = train_cfg["early_stopping_patience"]
    min_delta = 1e-3

    # Training history
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
        "learning_rate": []
    }

    print("\nStarting Training...\n")

    # TRAINING LOOP

    for epoch in range(1, train_cfg["epochs"] + 1):

        previous_lr = optimizer.param_groups[0]["lr"]

        # Training
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

        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer
        )

        # Scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Learning Rate", current_lr, epoch)

        # Store history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_precision"].append(train_precision)
        history["val_precision"].append(val_precision)
        history["train_recall"].append(train_recall)
        history["val_recall"].append(val_recall)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["learning_rate"].append(current_lr)

        if current_lr < previous_lr:
            print(f"\n[LR DROP] Epoch {epoch}: learning rate decreased from {previous_lr:.2e} to {current_lr:.2e}\n")

        # Epoch summary
        print(
            f"Epoch [{epoch}/{train_cfg['epochs']}] | "
            f"LR: {current_lr:.1e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Train Precision: {train_precision:.3f} | "
            f"Train Recall: {train_recall:.3f} | "
            f"Train F1: {train_f1:.3f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Precision: {val_precision:.3f} | "
            f"Val Recall: {val_recall:.3f} | "
            f"Val F1: {val_f1:.3f}"
        )

        # Best model
        if val_acc > best_val_acc + min_delta:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            early_stopping_counter = 0

            best_model_path = os.path.join(run_dir, f"best_model_acc_{val_acc:.1f}.pth")


            checkpoint = create_checkpoint(
                model=model,
                config=config,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                train_precision=train_precision,
                val_precision=val_precision,
                train_recall=train_recall,
                val_recall=val_recall,
                train_f1=train_f1,
                val_f1=val_f1
            )

            torch.save(checkpoint, best_model_path)

            print(
                f"\nBest model saved to {best_model_path} "
                f"(Val Acc improved by {improvement:.4f})\n"
            )

        else:
            early_stopping_counter += 1
            remaining_patience = early_stopping_patience - early_stopping_counter

            print(
                f"[PATIENCE WARNING] No val_acc improvement for "
                f"{early_stopping_counter} epoch(s). "
                f"Strikes: {early_stopping_counter}/{early_stopping_patience} "
                f"({remaining_patience} left before early stopping)\n"
            )

            if early_stopping_counter >= early_stopping_patience:
                print("\nEarly stopping triggered.")
                break

    # SAVE EXPERIMENT ARTIFACTS

    writer.close()

    # Save training history
    history_path = os.path.join(run_dir, "history.json")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining history saved to: {history_path}")

    # Save experiment summary
    summary_data = {
        "run_name": run_name,
        "architecture": arch_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_model_path": best_model_path,
        "total_epochs_completed": len(history["epoch"]),
        "early_stopping": len(history["epoch"]) < train_cfg["epochs"],
        "device": str(device),
        "history_path": history_path,
        "tensorboard_dir": tensorboard_dir,
        "config_path": config_path
    }

    summary_path = os.path.join(run_dir, "summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"Experiment summary saved to: {summary_path}")
    print("\nTraining Complete.")

    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_model_path": best_model_path,
        "history_path": history_path,
        "summary_path": summary_path,
        "tensorboard_dir": tensorboard_dir,
        "history": history
    }
