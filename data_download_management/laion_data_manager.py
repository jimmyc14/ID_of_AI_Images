# this assumes the data is already downloaded as 
# simple_laion400m_elsad3_subset_download ---
#                                            |
#                                            - > subset_train
#                                            - > subset_validation

import os
import csv
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
from natsort import natsorted
import glob
import json
import shutil

output_root = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST")  # where you want to save data
splits = ["train", "validation"]  # dataset splits to download

main_laion_folder = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_TEST_DOWNLOAD/simple_laion400m_elsad3_subset_download")

# def list_directories_only(path):
#     directories = []
#     for entry in os.listdir(path):
#         full_path = os.path.join(path, entry)
#         if os.path.isdir(full_path):
#             directories.append(entry)
#     return directories

def list_directories_only(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d != "_tmp"]

seed = 6050
random.seed(seed)
np.random.seed(seed)

for split in splits:
    split_folder = os.path.join(main_laion_folder, 'subset_' + split)
    output_split_folder = output_root / split / "laion"
    output_split_folder.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / f"{split}_laion_metadata.csv" 

    print(f"\nProcessing split: {split}")
    all_dirs = list_directories_only(split_folder)
    print(f"Found {len(all_dirs)} source folders: {all_dirs}")

    # all_dirs = list_directories_only(split_folder)
    # if '_tmp' in all_dirs:
    #     all_dirs.remove('_tmp')
    # print(all_dirs)
    # for sub_dir in all_dirs:
    #     sub_dir_path = os.path.join(split_folder, sub_dir)

    #     images = glob.glob(os.path.join(sub_dir_path, '*.png')) # get all png images
    #     jsons = glob.glob(os.path.join(sub_dir_path, '*.json')) # get all jsons
    #     images = natsorted(images) # ensure the same order
    #     jsons = natsorted(jsons) # ensure the same order

    #     assert len(images) == len(jsons)

    #     for image, json_file in zip(images, jsons):
    #         print(image)
    #         print(json_file)
    #         with open(json_file, 'r') as f:
    #             json_data = json.load(f)
    #             print(json_data)
    #             exit()

        # Open CSV for writing
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = None
        img_index = 0

        for sub_dir in tqdm(all_dirs, desc=f"Processing {split} dirs"):
            sub_dir_path = os.path.join(split_folder,sub_dir)

            images = natsorted(glob.glob(os.path.join(sub_dir_path,"*.png")))
            jsons = natsorted(glob.glob(os.path.join(sub_dir_path,"*.json")))

            assert len(images) == len(jsons), f"Image/JSON mismatch in {sub_dir_path}"

            for image_path, json_path in zip(images, jsons):
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                # Make CSV header once we know all the fields
                if writer is None:
                    fieldnames = list(json_data.keys()) + ["filename"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                # Save image with unique name (flat structure)
                image_name = f"{img_index:07d}.png"
                dest_path = output_split_folder / image_name
                shutil.copy2(image_path, dest_path)

                # Add metadata
                json_data["filename"] = str(dest_path.relative_to(output_root))
                writer.writerow(json_data)

                img_index += 1
            print(f"Finished split '{split}' - {sub_dir}")

    print(f"Finished split '{split}' — {img_index} images saved")
    print(f"Metadata CSV: {csv_path}")
