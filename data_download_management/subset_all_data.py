import os
import csv
from pathlib import Path
import random
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

def trim_and_copy_dataset(folder_path, csv_path, fraction_to_keep=0.6, seed=6050):
    """
    Randomly sample a fraction of images from a folder and copy them (and matching CSV rows)
    to a new mirrored folder called 'trimmed_main_dataset'.

    Assumes CSV has a column named 'filename' containing relative paths from the main dataset root.

    Parameters:
        folder_path (str | Path): Path to the folder containing the dataset (e.g., main_dataset/train/real).
        csv_path (str | Path): Path to the CSV file containing metadata.
        fraction_to_keep (float): Fraction of images to keep (0.0–1.0).
        seed (int): Random seed for reproducibility (default=6050).

    Returns:
        Path to updated CSV in the new 'trimmed_main_dataset' directory.
    """

    random.seed(seed)
    np.random.seed(seed)

    # Convert to Path objects
    folder_path = Path(folder_path)
    csv_path = Path(csv_path)

    # Define original dataset root (you can make this an argument too)
    main_data_folder = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/main_dataset")
    trimmed_root = main_data_folder.parent / "trimmed_main_data"

    # Create mirrored folder structure
    target_folder = trimmed_root / folder_path.relative_to(main_data_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    # Load and filter CSV
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("CSV must contain a 'filename' column referencing image paths.")

    # Keep only rows that correspond to actual files
    df = df[df["filename"].apply(lambda p: Path(main_data_folder / p).exists())]
    total_images = len(df)

    if total_images == 0:
        raise ValueError(f"No valid images found in {folder_path}")

    # Determine number to keep
    n_to_keep = int(total_images * fraction_to_keep)
    n_to_remove = total_images - n_to_keep

    # Randomly sample images to keep
    keep_indices = random.sample(range(total_images), n_to_keep)
    df_keep = df.iloc[keep_indices]

    print(f"Keeping {n_to_keep} / {total_images} images ({fraction_to_keep*100:.1f}%)")
    print(f"Copying to {target_folder}")

    # Copy images to the new structure
    for rel_path in tqdm(df_keep["filename"], desc="Copying images"):
        src = main_data_folder / rel_path
        dst = trimmed_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Save new CSV in the trimmed folder (mirrored structure)
    new_csv_path = trimmed_root / csv_path.relative_to(main_data_folder)
    new_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_keep.to_csv(new_csv_path, index=False)

    print(f"\nTrimmed CSV saved to: {new_csv_path}")
    print(f"Total kept images: {len(df_keep)}")

    return new_csv_path


if __name__ == "__main__":
    seed = 6050
    random.seed(seed)
    np.random.seed(seed)

    # Train Fake
    train_fake_csv = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/main_dataset/train_fake_metadata.csv")
    trim_and_copy_dataset(
        folder_path=r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/main_dataset/train/fake",
        csv_path=train_fake_csv,
        fraction_to_keep=0.4,
        seed=seed
    )

    # Validation Fake
    val_fake_csv = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/main_dataset/validation_fake_metadata.csv")
    trim_and_copy_dataset(
        folder_path=r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/main_dataset/validation/fake",
        csv_path=val_fake_csv,
        fraction_to_keep=0.4,
        seed=seed
    )

    # Train Fake
    train_real_csv = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/main_dataset/train_real_metadata.csv")
    trim_and_copy_dataset(
        folder_path=r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/main_dataset/train/real",
        csv_path=train_real_csv,
        fraction_to_keep=0.4,
        seed=seed
    )

    # Validation REal
    val_real_csv = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/main_dataset/validation_real_metadata.csv")
    trim_and_copy_dataset(
        folder_path=r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/main_dataset/validation/real",
        csv_path=val_real_csv,
        fraction_to_keep=0.4,
        seed=seed
    )

