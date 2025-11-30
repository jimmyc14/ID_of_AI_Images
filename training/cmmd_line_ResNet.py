import os
import random
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from datetime import datetime
import csv
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from jpeg_aug import RandomJPEGCompression
import argparse

# Helper Functions

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 model for AI image detection.")

    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder containing train/ and validation/ subfolders.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer.")
    parser.add_argument("--train_percent", type=float, default=1.0,
                        help="Percent of training data to use (0–1).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for DataLoader.")
    parser.add_argument("--model_name", type=str, default="model",
                        help="Name of model.")

    return parser.parse_args()


def subset_dataset(dataset, percent, seed=6050):
    if percent >= 1.0:
        return dataset
    
    random.seed(seed)
    subset_size = int(len(dataset) * percent)
    indices = random.sample(range(len(dataset)), subset_size)
    return Subset(dataset, indices)

def get_dataset_records(dataset, split_name):
    """
    Returns: list of dicts: {image_path, image_type, image_split}
    Works for both ImageFolder and Subset.
    """

    records = []

    # Determine base dataset (ImageFolder) and indices
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        indices = dataset.indices
    else:
        base = dataset
        indices = range(len(dataset))

    # Reverse mapping: 0→Real, 1→Fake
    label_map = {0: "real", 1: "fake"}

    # Collect rows
    for idx in indices:
        path, label = base.samples[idx]  # (full_path, original_label)

        # Apply label swap: real=0, fake=1
        correct_label = 1 - label
        label_name = label_map[correct_label]

        # Keep last 3 parts of path
        short_path = "/".join(Path(path).parts[-3:])

        records.append({
            "image_path": short_path,
            "image_type": label_name,
            "image_split": split_name
        })

    return records

def create_run_logger(model_name, params, timestamp, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create file paths
    log_txt = os.path.join(log_dir, f"{model_name}_run_{timestamp}.txt")
    log_json = os.path.join(log_dir, f"{model_name}_run_{timestamp}.json")

    # Initialize JSON log structure
    json_log = {
        "timestamp": timestamp,
        "model": model_name,
        "parameters": params,
        "epoch_data": []
    }

    # Write initial text header
    with open(log_txt, "w") as f:
        f.write("TRAINING RUN START \n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_name}\n")
        f.write("\nPARAMETERS \n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n==============================\n\n")

    return log_txt, log_json, json_log

# LOAD DATASETS

def loading_data(train_dir, val_dir, transform, train_ratio = 1.0, batch_size=32, num_workers=0):

    train_dataset = CustomImageFolder(root=str(train_dir), transform=transform)
    val_dataset = CustomImageFolder(root=str(val_dir), transform=transform)

    print("Original class mapping:", train_dataset.class_to_idx)  # still 'fake':0, 'real':1

    if train_ratio >= 1.0 or train_ratio <= 0.0:
        print("Using all the data")
    else:
        train_dataset = subset_dataset(train_dataset, train_ratio)
        val_dataset = subset_dataset(val_dataset, train_ratio)
        print(f"Using {train_ratio*100}% of the data")

    #DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    # logging data splits
    train_records = get_dataset_records(train_dataset, "train")
    val_records = get_dataset_records(val_dataset, "validation")

    all_records = train_records + val_records

    csv_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_name = f"dataset_split_{csv_datetime}.csv"
    csv_path = f"logs/{csv_name}"

    os.makedirs("logs", exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "image_type", "image_split"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Saved dataset split CSV to: {csv_path}")
    print(f"Total records: {len(all_records)}")

    return train_loader, val_loader, train_dataset, val_dataset, csv_name


# dataloader loads alphabetically, so we need to swap labels
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        # Swap label: make 'real' = 0, 'fake' = 1
        label = 1 - label
        return img, label

# Training loop with logging

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,
                model_name=None, csv_name_used = None):
    
    timestamp_pth = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model_name is None:
        model_name = "No model name provided"

    if csv_name_used is None:
        csv_name_used = "none provided"

    params = {
        "model": model_name,
        "dataset split used:": csv_name_used,
        "epochs_per_step": num_epochs,
        "batch_size": batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }

    log_txt, log_json, json_log = create_run_logger(model_name, params, timestamp_pth)

    # Log device + dataset sizes
    device_info = str(device)
    if device.type == "cuda":
        device_info += f" ({torch.cuda.get_device_name(0)})"

    with open(log_txt, "a") as f:
        f.write(f"Using device: {device_info}\n")
        f.write(f"Using dataset: {csv_name_used}\n")
        f.write(f"Using model: {model_name }\n")
        f.write(f"Using epochs per step: {num_epochs}\n")
        f.write(f"Using batch size: {batch_size}\n")
        f.write(f"Using learning rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Training samples: {len(train_loader.dataset)}\n")
        f.write(f"Validation samples: {len(val_loader.dataset)}\n\n")

    # Training setup
    best_val_acc = 0.0
    start_time = time.time()

    track_loss = []
    track_train_acc = []
    track_val_acc = []

    # Epoch loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n - Epoch {epoch+1}/{num_epochs}")

        # Wrap train loader with tqdm
        train_pbar = tqdm(train_loader, desc="Training", unit="batch")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_acc = 100 * correct / total
            avg_loss = running_loss / (len(train_loader) if len(train_loader) > 0 else 1)

            train_pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Train Acc": f"{train_acc:.2f}%"
            })

        # Validation Pass
        model.eval()
        val_correct = 0
        val_total = 0
        val_pbar = tqdm(val_loader, desc="Validating", unit="batch", leave=False)

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total

        # Logging epoch data
        with open(log_txt, "a") as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\n")
            f.write(f"  Loss: {avg_loss:.4f}\n")
            f.write(f"  Train Acc: {train_acc:.2f}%\n")
            f.write(f"  Val Acc: {val_acc:.2f}%\n\n")

        json_log["epoch_data"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}_{timestamp_pth}.pth")

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        

        track_loss.append(avg_loss)
        track_train_acc.append(train_acc)
        track_val_acc.append(val_acc)

    # Final log summary
    total_minutes = (time.time() - start_time) / 60

    with open(log_txt, "a") as f:
        f.write("\nTRAINING COMPLETE \n")
        f.write(f"Total time: {total_minutes:.2f} minutes\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f".pth saved as best_{model_name}_{timestamp_pth}.pth")

    json_log["total_minutes"] = total_minutes
    json_log["best_val_acc"] = best_val_acc

    with open(log_json, "w") as f:
        json.dump(json_log, f, indent=4)

    print(f"Logs saved to:\n  {log_txt}\n  {log_json}")

    return track_loss, track_train_acc, track_val_acc

# end of helper functions
#################################################################################################

# Checking Device/confirming it works with CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
x = torch.randn(10000, 10000, device=device)
print("Computation successful on", device)

# CONFIGURATION
args = parse_args()

data_root = Path(args.data_root)
train_dir = data_root / "train"
val_dir = data_root / "validation"

batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
train_percent = args.train_percent
num_workers = args.num_workers
model_name = args.model_name

# DATA TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize for ResNet
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    RandomJPEGCompression(quality_range=(30, 95)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_loader, val_loader, train_dataset, val_dataset, csv_name = loading_data(train_dir, 
                                                                    val_dir, 
                                                                    transform, 
                                                                    train_ratio = train_percent,
                                                                    batch_size=batch_size, 
                                                                    num_workers=num_workers)

# MODEL SETUP (ResNet50)
model = models.resnet50(weights=True)
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),   # hidden layer
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)           # final output
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # adding label smoothing for better generalization

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # adding L2 regularization

all_metrics = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,
                          model_name="ResNet50", csv_name_used=csv_name)

'''
EXAMPLE USAGE

I had to do this within Conda powershell, im sure there is a way to do it in powershell.

python cmmd_line_ResNet.py --data_root "C:/Users/Jimmy/OneDrive/Desktop/test/DS6050_Ai_Detection" --batch_size 32 --num_epochs 1 --learning_rate 1e-4 --train_percent 0.1 --num_workers 0 --model_name "ResNet50"

'''