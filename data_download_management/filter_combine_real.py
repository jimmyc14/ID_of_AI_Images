import os
import csv
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import shutil

'''
so, the 2 real datasets are laion and coco, and both have training and validation splits
we want to combine them together into a single train split, and a single validation split
however, there are (at least in the training data) many more images than in the fake dataset
coco is in jpg, and laion is in png, so file naming shouldnt be an issue, but should be aware of that later down the line

so, we need to sample from the 2 datasets, to match the fake dataset size. 

Fake training: 144,000 images
Fake validation: 36,000 images

however, both the laion and coco datasets do not have enougth validation images, so we may need to reduce the fake validation set.

we also need to take into consideration the csvs for both datasets, we need to combine them and make sure the correct images are pulled. 

Workflow todo:
1. sample from both datasets to make fake dataset size
2. combine the 2 datasets into single datasets
3. make sure to pull the correct image row from their csvs and combine them into one main csv
'''

seed = 6050
random.seed(seed)
np.random.seed(seed)
splits = ["train", "validation"]

main_data_folder = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST")

train_fake = pd.read_csv(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST/train_fake_metadata.csv")
val_fake = pd.read_csv(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST/validation_fake_metadata.csv")

laion_train = pd.read_csv(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST/train_laion_metadata.csv")
laion_val = pd.read_csv(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST/validation_laion_metadata.csv")

coco_train = pd.read_csv(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST/train_coco_metadata.csv")
coco_val = pd.read_csv(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST/validation_coco_metadata.csv")

# paired_train = train_fake[train_fake['paired_real_images'] != '[]']['paired_real_images']
# paried_val = val_fake[val_fake['paired_real_images'] != '[]']['paired_real_images']

laion2_train = laion_train.drop(columns=['key','status','error_message','original_width','original_height','exif','sha256'])
laion2_val = laion_val.drop(columns=['key','status','error_message','original_width','original_height','exif','sha256'])

coco2_train = coco_train.drop(columns=['license','date_captured','flickr_url'])
coco2_val = coco_val.drop(columns=['license','date_captured','flickr_url'])
coco2_train.columns = ['id','filename','url','height','width','description']
coco2_val.columns = ['id','filename','url','height','width','description']

concat_val = pd.concat([laion2_val, coco2_val], ignore_index=True)
# concat_train = pd.concat([laion2_train, coco2_train], ignore_index=True)

n = 144000 //2 # number of images we want from each dataset

sampled_laion_train = laion2_train.sample(n=n, random_state=seed)
sampled_coco_train = coco2_train.sample(n=n, random_state=seed)

concat_train = pd.concat([sampled_laion_train, sampled_coco_train], ignore_index=True)

for split in splits:
    df = concat_train if split == "train" else concat_val

    real_folder = main_data_folder / split / "real"
    real_folder.mkdir(parents=True, exist_ok=True)

    new_filenames = []
    for old_path in tqdm(df["filename"], desc="Copying images"):
        old_path = Path(os.path.join(main_data_folder,old_path))
        if not old_path.exists():
            print(f"Missing file: {old_path}")
            new_filenames.append("")  # keep placeholder
            continue

        # Keep just the file name (flattened)
        new_path_name = os.path.join(fr"{split}\real", old_path.name)
        new_path = real_folder / old_path.name

        # If duplicate filenames exist, rename to avoid overwrite
        if new_path.exists():
            stem, suffix = new_path.stem, new_path.suffix
            counter = 1
            while new_path.exists():
                new_path_name = f"{split}_real" / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.copy2(old_path, new_path)
        new_filenames.append(str(new_path_name))

    df["filename"] = new_filenames
    updated_csv_path = main_data_folder / f"{split}_real_metadata.csv"
    df.to_csv(updated_csv_path, index=False)

    print(f"Copied images to: {real_folder}")
    print(f"New real CSV saved at: {updated_csv_path}")
# concat_train.to_csv(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST/train_real_metadata.csv", index=False)
# concat_val.to_csv(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST/validation_real_metadata.csv", index=False)

trim_fake_val = len(val_fake) - len(concat_val)

# this function is needed to trim the fake dataset since our real dataset is smaller (for validation split)
def trim_dataset(folder_path, csv_path, n_to_remove, seed=6050):
    """
    Randomly remove n images from a folder and update the CSV accordingly.

    assumed the path to the image is in column called filename

    Parameters:
        folder_path (str | Path): Path to the folder containing images.
        csv_path (str | Path): Path to the CSV file that has image metadata.
        n_to_remove (int): Number of images to remove.
        seed (int): Random seed for reproducibility (default=6050).

    Returns:
        Path to updated CSV.
    """

    #this should probably be a variable to input, but the csv only store the path from the data storage root, so we need this
    main_data_folder_to_add = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST")

    # Load CSV
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("CSV must contain a 'filename' column referencing image paths.")

    # Only consider rows that actually exist as files in the folder
    df = df[df["filename"].apply(lambda p: Path(os.path.join(main_data_folder_to_add,p)).exists())]

    total_images = len(df)
    if n_to_remove >= total_images:
        raise ValueError(f"Cannot remove {n_to_remove} images — only {total_images} available.")

    random.seed(seed)
    to_remove_indices = random.sample(range(total_images), n_to_remove)
    to_remove = df.iloc[to_remove_indices]

    print(f"Removing {len(to_remove)} of {total_images} images.")

    # Remove the actual image files
    for img_path in to_remove["filename"]:
        img_path = Path(os.path.join(main_data_folder_to_add,img_path))
        try:
            os.remove(img_path)
        except FileNotFoundError:
            print(f"Missing file: {img_path}")

    # Keep remaining rows
    df_trimmed = df.drop(to_remove_indices)

    # Save updated CSV
    updated_csv_path = csv_path.parent / f"{csv_path.stem}_trimmed.csv"
    df_trimmed.to_csv(updated_csv_path, index=False)

    print(f"Trimmed dataset saved to: {updated_csv_path}")
    print(f"Remaining images: {len(df_trimmed)}")

    return updated_csv_path

trim_dataset(main_data_folder / "validation", main_data_folder / "validation_fake_metadata.csv", trim_fake_val)
pass