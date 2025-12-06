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
from helper_functions import *

# Helper Functions

def parse_args():
    parser = argparse.ArgumentParser(description="Train Temporal model for AI image detection.")

    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder containing train/ and validation/ subfolders.")
    parser.add_argument("--model_name", type=str, default="resnet50", required=True,
                        help="Name of model.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer.")
    parser.add_argument("--train_percent", type=float, default=1.0,
                        help="Percent of training data to use (0.1–1.0).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for DataLoader.")
    parser.add_argument("--jpeg_comp", type=bool, default=True,
                        help="Use JPEG Compression.")
    parser.add_argument("--save_model_name", type=str, default=None,
                        help="Name of model for logging.")

    return parser.parse_args()

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
num_epochs_per_step = args.num_epochs
learning_rate = args.learning_rate
train_percent = args.train_percent
num_workers = args.num_workers
model_name = args.model_name
jpeg_compression = args.jpeg_comp
save_model_name = args.save_model_name

if save_model_name is None:
    save_model_name = ""

if 'fft' in model_name:
    fft = True
else:
    fft = False

train_loader, val_loader, train_dataset, val_dataset, csv_name = main_data_loading(data_root, model_name, train_percent, 
                                                                                   batch_size, num_workers=num_workers, 
                                                                                   jpeg_compression=False)

train_real_indices, train_fake_indices_by_w, val_real_indices, val_fake_indices_by_w, max_w = confirm_windows(train_dataset, val_dataset)

confirm_labels(train_dataset, val_dataset)

model, criterion, optimizer = make_model(model_name=model_name, learning_rate=learning_rate, device=device)

history = sliding_window_training(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    train_real_indices=train_real_indices,
    train_fake_indices_by_w=train_fake_indices_by_w,
    val_real_indices=val_real_indices,
    val_fake_indices_by_w=val_fake_indices_by_w,
    max_w=max_w,
    num_epochs_per_step=num_epochs_per_step,
    batch_size=batch_size,
    device=device,
    model_name=model_name,
    num_workers=num_workers,
    csv_name_used=csv_name,
    model_save_name=save_model_name #not a needed parameter, but will add something at end of model name logger and pth to make it unique if wanted
)

'''
EXAMPLE USAGE

I had to do this within Conda powershell, im sure there is a way to do it in powershell.

python cmmd_line_temporal.py --data_root "C:/Users/Jimmy/OneDrive/Desktop/test/DS6050_Ai_Detection" --model_name "resnet50" 
--batch_size 16 --num_epochs 1 --learning_rate 1e-4 --train_percent 0.1 --num_workers=0 --jpeg_comp=True --save_model_name "cmmd_line"
'''