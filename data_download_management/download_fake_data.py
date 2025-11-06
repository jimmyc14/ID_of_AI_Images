import os
import csv
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np

output_root = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST")  # where you want to save data
hf_dataset_name = "lrzpellegrini/AI-GenBench-fake_part"
splits = ["train", "validation"]  # dataset splits to download

seed = 6050
random.seed(seed)
np.random.seed(seed)

def save_split(split_name):
    print(f"\nProcessing split: {split_name}")
    dataset = load_dataset(hf_dataset_name, split=split_name)

    # Create directories
    split_folder = output_root / split_name / "fake"
    split_folder.mkdir(parents=True, exist_ok=True)

    # Metadata CSV (placed in main dataset folder)
    metadata_file = output_root / f"{split_name}_fake_metadata.csv"

    #include all metadata columns
    csv_fields = list(dataset.features.keys()) + ["filename"]

    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for idx, sample in enumerate(tqdm(dataset)):
            image = sample["image"]
            image_name = f"{idx:06d}.png"
            image_path = split_folder / image_name
            image.save(image_path)

            # Build metadata row with all available columns
            row = {k: sample.get(k, None) for k in dataset.features.keys()}
            row["filename"] = str(image_path.relative_to(output_root))
            writer.writerow(row)

    print(f"Saved {len(dataset)} images for split '{split_name}'")
    print(f"Metadata CSV saved to {metadata_file}")


for split in splits:
    save_split(split)
