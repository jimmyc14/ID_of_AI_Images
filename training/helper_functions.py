import os
import sys
import random
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from datetime import datetime
import csv
import json
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from torchvision.models import vit_b_16, ViT_B_16_Weights

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from jpeg_aug import RandomJPEGCompression

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from models import get_model

'''
Helper Functions
'''

def subset_dataset(dataset, percent, seed=6050):
    if percent >= 1.0:
        return dataset
    
    random.seed(seed)
    subset_size = int(len(dataset) * percent)
    indices = random.sample(range(len(dataset)), subset_size)
    return Subset(dataset, indices)

def plot_loss_and_acc(track_loss, track_train_acc, track_val_acc):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(track_train_acc, label="Train")
    plt.plot(track_val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(track_loss, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

def create_grid_of_val_images(val_dataset, grid_size=4, fft=False, device='cpu', keep=False):
    # Create a grid of images
    num_images = grid_size * grid_size

    if isinstance(val_dataset, torch.utils.data.Subset):
        subset = val_dataset
        base = val_dataset.dataset

        all_indices = subset.indices
        all_paths  = [base.samples[i][0] for i in all_indices]
        all_labels = [base.samples[i][1] for i in all_indices]

    else:
        all_paths  = [p for p, l in val_dataset.samples]
        all_labels = [l for p, l in val_dataset.samples]

    labels_tensor = torch.tensor(all_labels)

    real_indices = (labels_tensor == 0).nonzero(as_tuple=True)[0]
    fake_indices = (labels_tensor == 1).nonzero(as_tuple=True)[0]

    half_num = num_images // 2
    real_sample = real_indices[torch.randperm(len(real_indices))[:half_num]]
    fake_sample = fake_indices[torch.randperm(len(fake_indices))[:half_num]]

    subset_indices = torch.cat([real_sample, fake_sample])

    if isinstance(val_dataset, torch.utils.data.Subset):
        base = val_dataset.dataset
        original_indices = [all_indices[i.item()] for i in subset_indices]

        fixed_images = torch.stack([base[i][0] for i in original_indices]).to(device)
        fixed_labels = torch.tensor([base[i][1] for i in original_indices]).to(device)
        fixed_paths  = [all_paths[i] for i in subset_indices]

    else:
        fixed_images = torch.stack([val_dataset[i][0] for i in subset_indices]).to(device)
        fixed_labels = torch.tensor([val_dataset[i][1] for i in subset_indices]).to(device)
        fixed_paths  = [all_paths[i] for i in subset_indices]

    if fft is True:
        #UPDATE FOR FFT
        fixed_rgb = fixed_images[:, :3, :, :]   # shape (N, 3, H, W)
    else:
        fixed_rgb = fixed_images

    grid_img = make_grid(fixed_rgb.cpu(), nrow=grid_size, normalize=True)
    plt.figure(figsize=(6,6))
    plt.imshow(np.transpose(grid_img.numpy(), (1,2,0)))
    plt.axis("off")
    plt.title("Validation Images")
    plt.show()

    for path, lbl in zip(fixed_paths, fixed_labels):
        print(f"{'Real' if lbl==0 else 'Fake'}: {os.path.basename(path)}")
    
    if keep:
        return fixed_images, fixed_labels, grid_size


def main_data_loading(data_root, model_name, train_percent, batch_size, num_workers=0, jpeg_compression=False):
    #set up variables
    train_dir = data_root / "train"
    val_dir = data_root / "validation"

    model_type = model_name.split('_')[0]

    if 'fft' in model_name:
        fft = True
    else:
        fft = False

    # DATA TRANSFORMS

    print(f"Using model type: {model_type}")
    print(f"Using FFT: {fft}")

    # ViT and ResNet50 have different base transforms
    if model_type == 'vit':
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
        ]

        if jpeg_compression:
            base_transforms.append(RandomJPEGCompression(quality_range=(30, 95)))

        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    elif model_type == 'resnet50':
        base_transforms = [
            transforms.Resize((256, 256)),  # Resize for ResNet
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        if jpeg_compression:
            base_transforms.append(RandomJPEGCompression(quality_range=(30, 95)))

        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    transform = transforms.Compose(base_transforms)

    train_loader, val_loader, train_dataset, val_dataset, csv_name = loading_data(train_dir, 
                                                                    val_dir, 
                                                                    transform, 
                                                                    train_ratio = train_percent,
                                                                    batch_size=batch_size, 
                                                                    num_workers=num_workers, fft=fft)
    
    return train_loader, val_loader, train_dataset, val_dataset, csv_name

def confirm_windows(train_dataset, val_dataset):
    # Paths to your fake CSVs (this path should work in main project)
    train_fake_csv_path = r'../temporal_validation/fake_train_temporal_splits.csv'
    val_fake_csv_path = r'../temporal_validation/fake_validation_temporal_splits.csv'

    train_fake_df = pd.read_csv(train_fake_csv_path)
    val_fake_df = pd.read_csv(val_fake_csv_path)

    # Ensure filename column is just the basename (in case it has paths)
    train_fake_df["filename"] = train_fake_df["filename"].apply(os.path.basename)
    val_fake_df["filename"] = val_fake_df["filename"].apply(os.path.basename)

    # Map: filename -> window
    train_file_to_window = dict(zip(train_fake_df["filename"], train_fake_df["w_split"]))
    val_file_to_window = dict(zip(val_fake_df["filename"],   val_fake_df["w_split"]))

    # Determine max window index
    max_train_w = train_fake_df["w_split"].max()
    max_val_w = val_fake_df["w_split"].max()
    max_w = int(max(max_train_w, max_val_w))

    print(f"Max window index detected: {max_w}, should be 8")

    base_train_ds, train_indices = get_base_dataset(train_dataset)
    base_val_ds, val_indices   = get_base_dataset(val_dataset)

    FAKE_CLASS = base_train_ds.class_to_idx["fake"]  # flipped: fake=0, real=1
    REAL_CLASS = base_train_ds.class_to_idx["real"]

    # Create mapping: index_in_train_dataset → actual_path_and_class
    train_samples = base_train_ds.samples
    val_samples   = base_val_ds.samples

    # Determine max window
    max_train_w = train_fake_df["w_split"].max()
    max_val_w = val_fake_df["w_split"].max()
    max_w = int(max(max_train_w, max_val_w))

    # Storage structures
    train_fake_indices_by_w = {w: [] for w in range(max_w + 1)}
    val_fake_indices_by_w = {w: [] for w in range(max_w + 1)}
    train_real_indices = []
    val_real_indices = []

    # Build TRAIN indices properly whether subset or not
    for new_i, original_i in enumerate(train_indices):
        path, cls = train_samples[original_i]
        fname = os.path.basename(path)

        if cls == REAL_CLASS:
            train_real_indices.append(new_i)

        elif cls == FAKE_CLASS:
            if fname in train_file_to_window:
                w = int(train_file_to_window[fname])
                train_fake_indices_by_w[w].append(new_i)
            else:
                print(f"[WARN] Train fake missing from CSV: {fname}")

    # Build VAL indices properly
    for new_i, original_i in enumerate(val_indices):
        path, cls = val_samples[original_i]
        fname = os.path.basename(path)

        if cls == REAL_CLASS:
            val_real_indices.append(new_i)

        elif cls == FAKE_CLASS:
            if fname in val_file_to_window:
                w = int(val_file_to_window[fname])
                val_fake_indices_by_w[w].append(new_i)
            else:
                print(f"[WARN] Val fake missing from CSV: {fname}")

    print("Train real:", len(train_real_indices))
    for w in range(max_w + 1):
        print(f"Train fake w{w}: {len(train_fake_indices_by_w[w])}")

    print("Val real:", len(val_real_indices))
    for w in range(max_w + 1):
        print(f"Val fake w{w}: {len(val_fake_indices_by_w[w])}")
    
    return train_real_indices, train_fake_indices_by_w, val_real_indices, val_fake_indices_by_w, max_w

def confirm_labels(train_dataset, val_dataset, num_print=5):
    '''
    num_print: number of paths to print per train/val
    '''
    print("Confirm correct labels")
    print("fake -> 1")
    print("real -> 0")

    subset_train = train_dataset
    subset_val = val_dataset

    n_t = len(subset_train)
    n_v = len(subset_val)

    # Pick 5 random unique indices
    random_t_indices = random.sample(range(n_t), 5)
    random_v_indices = random.sample(range(n_v), 5)

    for idx in random_t_indices:
        if isinstance(subset_train, torch.utils.data.Subset):
            actual_idx = subset_train.indices[idx]
            img, label, _ = subset_train[idx]
            path = subset_train.dataset.imgs[actual_idx][0]
        else:
            img, label, _ = subset_train[idx]
            path = subset_train.imgs[idx][0]

        print(f"{path} --> Label: {label}")
        if 'fake' in path and label == 0:
            print("wrong label detected")
        if 'real' in path and label == 1:
            print("wrong label detected")

    for idx in random_v_indices:
        if isinstance(subset_val, torch.utils.data.Subset):
            actual_idx = subset_val.indices[idx]
            img, label, _ = subset_val[idx]
            path = subset_val.dataset.imgs[actual_idx][0]
        else:
            img, label, _ = subset_val[idx]
            path = subset_val.imgs[idx][0]

        print(f"{path} --> Label: {label}")
        if 'fake' in path and label == 0:
            print("wrong label detected")
        if 'real' in path and label == 1:
            print("wrong label detected")

def make_model(model_name, learning_rate, device='cpu'):
    model = get_model(model_name=model_name)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # adding label smoothing for better generalization

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # adding L2 regularization

    return model, criterion, optimizer

def show_jpeg_transform_effect(dataset, index, jpeg_q_range = (5, 60)):

    jpeg_aug = RandomJPEGCompression(quality_range=jpeg_q_range)

    # Load image according to dataset structure
    if isinstance(dataset, torch.utils.data.Subset):
        original_idx = dataset.indices[index]
        path = dataset.dataset.samples[original_idx][0]
        pil_img = Image.open(path).convert("RGB")
        label = dataset[ index ][1]
    else:
        path = dataset.samples[index][0]
        pil_img = Image.open(path).convert("RGB")
        label = dataset[index][1]
        
    # Apply your RandomJPEGCompression transform
    compressed_img = jpeg_aug(pil_img)

    # Plot original vs. compressed
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(pil_img)
    axs[0].set_title(f"Original\n{os.path.basename(path)}")
    axs[0].axis("off")

    axs[1].imshow(compressed_img)
    axs[1].set_title(f"JPEG Transform Applied\nClass={label}")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

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

def loading_data(train_dir, val_dir, transform, train_ratio = 1.0, batch_size=32, num_workers=0, fft=False):

    if fft is True:
        train_dataset = FFTImageFolder(root=str(train_dir), transform=transform)
        val_dataset = FFTImageFolder(root=str(val_dir), transform=transform)
    else:
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

def make_loader_for_windows(dataset, real_indices, fake_indices_by_w, 
                            windows, batch_size, shuffle, num_workers=0):
    all_indices = list(real_indices)
    for w in windows:
        all_indices.extend(fake_indices_by_w[w])
    subset = Subset(dataset, all_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, len(all_indices)

def create_run_logger(model_name, params, timestamp, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    # Log file paths
    log_txt = os.path.join(log_dir, f"{model_name}_run_{timestamp}.txt")
    log_json = os.path.join(log_dir, f"{model_name}_run_{timestamp}.json")

    # JSON structure
    json_log = {
        "timestamp": timestamp,
        "model": model_name,
        "parameters": params,
        "steps": []   # < sliding steps
    }

    # TXT file
    with open(log_txt, "w") as f:
        f.write("TRAINING RUN START\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_name}\n\n")
        f.write("PARAMETERS:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n==============================\n\n")

    return log_txt, log_json, json_log

def logger_add_step(json_log, step_index, windows_used):
    step_entry = {
        "step": step_index,
        "windows_used": windows_used,
        "epochs": []   # epochs get appended inside this
    }
    json_log["steps"].append(step_entry)

def logger_add_epoch(json_log, step_index, epoch_index, metrics):
    """
    metrics = {
        "train_loss": ...,
        "train_acc": ...,
        "val_past_acc": ...,
        "val_next_acc": ...,
        "val_all_acc": ...,
        ............,
        "epoch_minutes": ...
    }
    """
    json_log["steps"][step_index]["epochs"].append({
        "epoch": epoch_index,
        **metrics
    })

def save_json_log(json_log, json_path):
    with open(json_path, "w") as f:
        json.dump(json_log, f, indent=4)

def get_base_dataset(ds):
    """Return (base_dataset, indices_list) whether ds is Subset or full dataset."""
    if isinstance(ds, torch.utils.data.Subset):
        return ds.dataset, ds.indices
    else:
        return ds, list(range(len(ds)))
    
def convert_to_dataset_indices(dataset, idx_list):
    if isinstance(dataset, torch.utils.data.Subset):
        return { dataset.indices[i] for i in idx_list }
    return set(idx_list)

def eval_temporal_metrics(
    model,
    loader,
    device,
    desc="Evaluating",
    fake_indices_by_w=None,
    current_seen_windows=None,
    current_unseen_windows=None,
    val_dataset=None
):
    """
    Computes full evaluation metrics:
        - Accuracy
        - Precision (macro)
        - Recall (macro)
        - F1 (macro)
        - ROC-AUC (binary)
        - Seen F1 (AI-GenBench)
        - Unseen F1 (AI-GenBench)
    """

    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []

    # Seen/unseen buckets
    seen_labels = []
    seen_preds = []
    unseen_labels = []
    unseen_preds = []
    
    seen_indices = set()
    unseen_indices = set()

    if fake_indices_by_w is not None:

        if current_seen_windows is not None:
            for w in current_seen_windows:
                seen_indices |= convert_to_dataset_indices(val_dataset, fake_indices_by_w[w])

        if current_unseen_windows is not None:
            for w in current_unseen_windows:
                unseen_indices |= convert_to_dataset_indices(val_dataset, fake_indices_by_w[w])

    # EVALUATION LOOP
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
        for batch_idx, (images, labels, indices) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)

            labels_np = labels.cpu().numpy()
            preds_np = predicted.cpu().numpy()
            scores_np = probs.cpu().numpy()
            indices_np = indices.cpu().numpy()

            all_labels.extend(labels_np)
            all_preds.extend(preds_np)
            all_scores.extend(scores_np)

            
            # Track seen / unseen FAKE predictions
            for j in range(len(labels_np)):
                ds_index = int(indices_np[j])  # this IS the dataset index already

                if labels_np[j] == 1:
                    if ds_index in seen_indices:
                        seen_labels.append(labels_np[j])
                        seen_preds.append(preds_np[j])
                    elif ds_index in unseen_indices:
                        unseen_labels.append(labels_np[j])
                        unseen_preds.append(preds_np[j])

    # GLOBAL METRICS
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        rocauc = roc_auc_score(all_labels, all_scores)
    except:
        rocauc = float("nan")

    # SEEN & UNSEEN F1
    seen_f1 = f1_score(seen_labels, seen_preds, average="binary", zero_division=0) if len(seen_labels) > 0 else float("nan")
    unseen_f1 = f1_score(unseen_labels, unseen_preds, average="binary", zero_division=0) if len(unseen_labels) > 0 else float("nan")

    return {
        "accuracy": acc * 100,
        "precision_macro": precision * 100,
        "recall_macro": recall * 100,
        "f1_macro": f1macro * 100,
        "roc_auc": rocauc,
        "seen_f1": seen_f1 * 100,
        "unseen_f1": unseen_f1 * 100,
    }

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_pbar = tqdm(train_loader, desc="Training", unit="batch")
    for images, labels, _ in train_pbar:
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

        avg_loss = running_loss / (len(train_loader) if len(train_loader) > 0 else 1)
        train_acc = 100.0 * correct / total

        train_pbar.set_postfix({
            "Loss": f"{avg_loss:.4f}",
            "Train Acc": f"{train_acc:.2f}%"
        })

    avg_loss = running_loss / (len(train_loader) if len(train_loader) > 0 else 1)
    train_acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, train_acc

def sliding_window_training(
    model,
    criterion,
    optimizer,
    train_dataset,
    val_dataset,
    train_real_indices,
    train_fake_indices_by_w,
    val_real_indices,
    val_fake_indices_by_w,
    max_w,
    num_epochs_per_step,
    batch_size,
    device,
    model_name="Temporal_ResNet50+FFT",
    num_workers=0,
    csv_name_used = None,
    model_save_name = None
):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if csv_name_used is None:
        csv_name_used = "none provided"

    if model_save_name is not None:
        model_name_log = f"temporal_{model_name}_{model_save_name}"
    elif model_save_name is None:
        model_name_log = f"temporal_{model_name}"

    params = {
        "training_type": "temporal",
        "model": model_name,
        "dataset split used:": csv_name_used,
        "max_w": max_w,
        "epochs_per_step": num_epochs_per_step,
        "batch_size": batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }

    log_txt, log_json_path, json_log = create_run_logger(
        model_name=model_name_log,
        params=params,
        timestamp=timestamp
    )

    # Log device + dataset sizes
    device_info = str(device)
    if device.type == "cuda":
        device_info += f" ({torch.cuda.get_device_name(0)})"
        print(f"Using device: {device_info}")
    else:
        device_info += " (CPU)"
        print(f"Using device: {device_info}")

    with open(log_txt, "a") as f:
        f.write(f"Using device: {device_info}\n")
        f.write(f"Using dataset: {csv_name_used}\n")
        f.write(f"Using model: {model_name }\n")
        f.write(f"Using max sliding window: {max_w}\n")
        f.write(f"Using epochs per step: {num_epochs_per_step}\n")
        f.write(f"Using batch size: {batch_size}\n")
        f.write(f"Using learning rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Dataset sizes: Train: {len(train_dataset)}, Val: {len(val_dataset)}\n\n")

    history = []
    start_time = time.time()
    best_val_acc = 0
    best_val_f1 = 0

    for k in range(0, max_w):  # last step is k=max_w-1 with next window = max_w
        print("\n" + "="*70)
        print(f" - Sliding step {k}: train on w[0..{k}], validate on w{ k+1 }")
        print("="*70)

        logger_add_step(
            json_log,
            step_index=k,
            windows_used=list(range(0, k+1))
        )

        past_windows = list(range(0, k+1))
        next_window  = k + 1
        all_up_to_next = list(range(0, max_w+1))

        # Build loaders
        train_loader, n_train = make_loader_for_windows(
            train_dataset,
            train_real_indices,
            train_fake_indices_by_w,
            past_windows,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        val_past_loader, n_val_past = make_loader_for_windows(
            val_dataset,
            val_real_indices,
            val_fake_indices_by_w,
            past_windows,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        val_next_loader, n_val_next = make_loader_for_windows(
            val_dataset,
            val_real_indices,
            val_fake_indices_by_w,
            [next_window],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        val_all_loader, n_val_all = make_loader_for_windows(
            val_dataset,
            val_real_indices,
            val_fake_indices_by_w,
            all_up_to_next,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        print(f"Train: {n_train} samples (Real + Fake windows {past_windows})")
        print(f"Val past: {n_val_past} samples (windows {past_windows})")
        print(f"Val next: {n_val_next} samples (window {next_window})")
        print(f"Val all: {n_val_all} samples (windows {all_up_to_next})")

        with open(log_txt, "a") as f:
            f.write(f"Step: {k}\n")
            f.write(f" * Sliding step {k}: train on w[0..{k}], validate on w{ k+1 }\n")
            f.write(f"Train: {n_train} samples (Real + Fake windows {past_windows})\n")
            f.write(f"Val past: {n_val_past} samples (windows {past_windows})\n")
            f.write(f"Val next: {n_val_next} samples (window {next_window})\n")
            f.write(f"Val all: {n_val_all} samples (windows {all_up_to_next})\n\n")

        # Train for num_epochs_per_step on current window setup
        for epoch in range(num_epochs_per_step):
            epoch_start_time = time.time()
            print(f"\n[Step {k}] Epoch {epoch+1}/{num_epochs_per_step}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            metrics_past = eval_temporal_metrics(
            model,
            val_past_loader,
            device,
            desc="Val past",
            fake_indices_by_w=val_fake_indices_by_w,
            current_seen_windows=past_windows,    
            current_unseen_windows=[next_window],
            val_dataset=val_dataset)

            metrics_next = eval_temporal_metrics(
            model,
            val_next_loader,
            device,
            desc="Val next",
            fake_indices_by_w=val_fake_indices_by_w,
            current_seen_windows=past_windows,    
            current_unseen_windows=[next_window],
            val_dataset=val_dataset)

            metrics_all = eval_temporal_metrics(
            model,
            val_all_loader,
            device,
            desc="Val all",
            fake_indices_by_w=val_fake_indices_by_w,
            current_seen_windows=past_windows,
            current_unseen_windows=[next_window],
            val_dataset=val_dataset)

            # Accuracy
            acc_past = metrics_past["accuracy"]
            acc_next = metrics_next["accuracy"]
            acc_all = metrics_all["accuracy"]

            # Precision
            prec_past = metrics_past['precision_macro']
            prec_next = metrics_next['precision_macro']
            prec_all = metrics_all['precision_macro']

            # Recall
            recall_past = metrics_past['recall_macro']
            recall_next = metrics_next['recall_macro']
            recall_all = metrics_all['recall_macro']

            # F1
            f1_past = metrics_past['f1_macro']
            f1_next = metrics_next['f1_macro']
            f1_all = metrics_all['f1_macro']

            # SEEN F1
            seen_f1_past = metrics_past['seen_f1']
            seen_f1_next = metrics_next['seen_f1']
            seen_f1_all = metrics_all['seen_f1']

            # SEEN F1
            unseen_f1_past = metrics_past['unseen_f1']
            unseen_f1_next = metrics_next['unseen_f1']
            unseen_f1_all = metrics_all['unseen_f1']

            # ROC-AUC
            aroc_past = metrics_past['roc_auc']
            aroc_next = metrics_next['roc_auc']
            aroc_all = metrics_all['roc_auc'] 

            print(
                f"[Step {k} | Epoch {epoch+1}] \n"
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | \n"
                f"Acc_ValPast: {acc_past:.2f}% | Acc_ValNext: {acc_next:.2f}% | Acc_ValAll: {acc_all:.2f}% \n"
                f"Prec_ValPast: {prec_past:.2f}% | Prec_ValNext: {prec_next:.2f}% | Prec_ValAll: {prec_all:.2f}% \n"
                f"Recall_ValPast: {recall_past:.2f}% | Recall_ValNext: {recall_next:.2f}% | Recall_ValAll: {recall_all:.2f}% \n"
                f"F1_ValPast: {f1_past:.2f}% | F1_ValNext: {f1_next:.2f}% | F1_ValAll: {f1_all:.2f}% \n"
                f"Seen_F1_ValPast: {seen_f1_past:.2f}% | Seen_F1_ValNext: {seen_f1_next:.2f}% | Seen_F1_ValAll: {seen_f1_all:.2f}% \n"
                f"Unseen_F1_ValPast: {unseen_f1_past:.2f}% | Unseen_F1_ValNext: {unseen_f1_next:.2f}% | Unseen_F1_ValAll: {unseen_f1_all:.2f}% \n"
                f"AROC_ValPast: {aroc_past:.2f} | AROC_ValNext: {aroc_next:.2f} | AROC_ValAll: {aroc_all:.2f} \n"
            )

            history.append({
                "step": k,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                ########################
                "val_past_acc": acc_past,
                "val_next_acc": acc_next,
                "val_all_acc": acc_all,
                #######################
                "val_past_prec": prec_past,
                "val_next_prec": prec_next,
                "val_all_prec": prec_all,
                #######################
                "val_past_recall": recall_past,
                "val_next_recall": recall_next,
                "val_all_recall": recall_all,
                #######################
                "val_past_f1": f1_past,
                "val_next_f1": f1_next,
                "val_all_f1": f1_all,
                #######################
                "val_past_seenf1": seen_f1_past,
                "val_next_seenf1": seen_f1_next,
                "val_all_seenf1": seen_f1_all,
                #######################
                "val_past_unseenf1": unseen_f1_past,
                "val_next_unseenf1": unseen_f1_next,
                "val_all_unseenf1": unseen_f1_all,
                #######################
                "val_past_aroc": aroc_past,
                "val_next_aroc": aroc_next,
                "val_all_aroc": aroc_all
            })
            total_epoch_minutes = (time.time() - epoch_start_time) / 60.0

            logger_add_epoch(
                json_log,
                step_index=k,
                epoch_index=epoch+1,
                metrics={
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    #######################
                    "val_past_acc": acc_past,
                    "val_next_acc": acc_next,
                    "val_all_acc": acc_all,
                    #######################
                    "val_past_prec": prec_past,
                    "val_next_prec": prec_next,
                    "val_all_prec": prec_all,
                    #######################
                    "val_past_recall": recall_past,
                    "val_next_recall": recall_next,
                    "val_all_recall": recall_all,
                    #######################
                    "val_past_f1": f1_past,
                    "val_next_f1": f1_next,
                    "val_all_f1": f1_all,
                    #######################
                    "val_past_seenf1": seen_f1_past,
                    "val_next_seenf1": seen_f1_next,
                    "val_all_seenf1": seen_f1_all,
                    #######################
                    "val_past_unseenf1": unseen_f1_past,
                    "val_next_unseenf1": unseen_f1_next,
                    "val_all_unseenf1": unseen_f1_all,
                    #######################
                    "val_past_aroc": aroc_past,
                    "val_next_aroc": aroc_next,
                    "val_all_aroc": aroc_all,
                    "epoch_minutes": total_epoch_minutes
                }
            )

            with open(log_txt, "a") as f:
                f.write(f"Step: {k} ; Epoch = : {epoch+1}\n")
                f.write(f"\tTrain Loss: {train_loss} ; Train Acc {train_acc}\n")
                f.write(f"\tVal Past Acc: {acc_past}\n")
                f.write(f"\tVal Next Acc: {acc_next}\n")
                f.write(f"\tVal All Acc: {acc_all}\n")
                ################################################
                f.write(f"\tVal Past Prec: {prec_past}\n")
                f.write(f"\tVal Next Prec: {prec_next}\n")
                f.write(f"\tVal All Prec: {prec_all}\n")
                ################################################
                f.write(f"\tVal Past Recall: {recall_past}\n")
                f.write(f"\tVal Next Recall: {recall_next}\n")
                f.write(f"\tVal All Recall: {recall_all}\n")
                ################################################
                f.write(f"\tVal Past F1: {f1_past}\n")
                f.write(f"\tVal Next F1: {f1_next}\n")
                f.write(f"\tVal All F1: {f1_all}\n")
                ################################################
                f.write(f"\tVal Past Seen F1: {seen_f1_past}\n")
                f.write(f"\tVal Next Seen F1: {seen_f1_next}\n")
                f.write(f"\tVal All Seen F1: {seen_f1_all}\n")
                ################################################
                f.write(f"\tVal Past Unseen F1: {unseen_f1_past}\n")
                f.write(f"\tVal Next Unseen F1: {unseen_f1_next}\n")
                f.write(f"\tVal All Unseen F1: {unseen_f1_all}\n")
                ################################################
                f.write(f"\tVal Past AROC: {aroc_past}\n")
                f.write(f"\tVal Next AROC: {aroc_next}\n")
                f.write(f"\tVal All AROC: {aroc_all}\n")

                if epoch+1 == num_epochs_per_step:
                    f.write(f"\tTotal Epoch Minutes {total_epoch_minutes:.2f}\n\n")
                    f.write(f"{'='*30}\n")
                else:
                    f.write(f"\tTotal Epoch Minutes {total_epoch_minutes:.2f}\n")

            # if acc_all > best_val_acc:
            #     best_val_acc = acc_all
            #     torch.save(model.state_dict(), f"best_{model_name_log}_{timestamp}.pth")

            if f1_all > best_val_f1:
                best_val_f1 = f1_all
                torch.save(model.state_dict(), f"best_{model_name_log}_{timestamp}.pth")

            save_json_log(json_log, log_json_path)

            print(f"Total Epoch time: {total_epoch_minutes:.2f} minutes")

    total_minutes = (time.time() - start_time) / 60.0
    print(f"\nSliding-window training complete in {total_minutes:.2f} minutes.")

    with open(log_txt, "a") as f:
        f.write(f"Best Validation accuracy for all windows: {best_val_acc}")
        f.write(f"Best Validation F1 for all windows: {best_val_f1}")
        f.write(f"Training Complete in: {total_minutes:.2f} minutes.")

    return history

''' 
Helper Classes
'''

# dataloader loads alphabetically, so we need to swap labels + adding FFT
class FFTImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        
        # Swap label: real=0, fake=1
        label = 1 - label
        
        fft = torch.fft.fft2(img, dim=(-2, -1))
        fft = torch.fft.fftshift(fft) 
        
        fft_mag = torch.abs(fft)
        
        fft_mag = torch.log1p(fft_mag)

        fft_mag = (fft_mag - fft_mag.mean()) / (fft_mag.std() + 1e-6)

        # Concatenate into a 6-channel tensor
        img_fft = torch.cat([img, fft_mag], dim=0)

        return img_fft, label, index
    
# dataloader loads alphabetically, so we need to swap labels
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        # Swap label: make 'real' = 0, 'fake' = 1
        label = 1 - label
        return img, label, index