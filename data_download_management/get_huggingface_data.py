from datasets import load_from_disk
from pathlib import Path
import shutil
import os
from PIL import Image as PILImage, ImageFile
from tqdm import tqdm
import time

# Settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
PILImage.MAX_IMAGE_PIXELS = None  # disable PIL decompression bomb check, we are checking it ourselves

###  UPDATE THIS LINE ####
where_you_saved_data = r"C:/Users/Jimmy/OneDrive/Desktop/test/DS6050_Ai_Detection" 
##########################

hf_local_path = Path(where_you_saved_data)
splits = ["train", "validation"]
types = ["real", "fake"]

SCALE_FACTOR = 0.1  # Reduce images to 10% of original size if needed
MAX_PIXELS = 178_956_970  # Threshold before resizing

overall_start = time.time()

for split in splits:
    temp_split_folder = hf_local_path / f"temp_{split}"
    temp_split_folder.mkdir(exist_ok=True)
    print(f"\nStarting reconstruction for {split} split...")

    for dtype in types:
        folder_start = time.time()
        arrow_path = hf_local_path / split / dtype
        if not arrow_path.exists():
            print(f"Skipping: {arrow_path} does not exist.")
            continue

        print(f"Loading Arrow dataset from {arrow_path} ...")
        ds = load_from_disk(str(arrow_path))

        out_folder = temp_split_folder / dtype
        out_folder.mkdir(parents=True, exist_ok=True)

        skipped_count = 0
        resized_count = 0

        # tqdm progress bar
        pbar = tqdm(enumerate(ds), total=len(ds), desc=f"{split}/{dtype}", unit="img")
        for idx, row in pbar:
            try:
                img_data = row["image"]

                # Extract original filename
                fname = row['filename']
                if fname:
                    fname = Path(fname).name
                    ext = Path(fname).suffix.lower()

                    if ext not in [".jpg", ".jpeg", ".png"]:
                        ext = ".jpg"
                        fname = f"{Path(fname).stem}{ext}"
                else:
                    fname = f"{idx:06d}.jpg"
                    ext = ".jpg"

                dst_path = out_folder / fname

                # Handle PIL images
                if hasattr(img_data, "save"):
                    img = img_data

                    # Resize oversized images
                    if img.width * img.height > MAX_PIXELS:
                        resized_count += 1
                        new_size = (int(img.width * SCALE_FACTOR), int(img.height * SCALE_FACTOR))
                        img = img.resize(new_size, PILImage.Resampling.LANCZOS)

                    # Convert modes for compatibility
                    if img.mode in ("P", "RGBA"):
                        img = img.convert("RGB") if ext != ".png" else img.convert("RGBA")

                    # Save with original extension
                    img_format = "PNG" if ext == ".png" else "JPEG"
                    save_kwargs = {"format": img_format}
                    if img_format == "JPEG":
                        save_kwargs.update({"quality": 95, "optimize": True})

                    img.save(dst_path, **save_kwargs)

                # Handle bytes
                elif isinstance(img_data, bytes):
                    with open(dst_path, "wb") as f:
                        f.write(img_data)

            except Exception as e:
                skipped_count += 1
                print(f"Skipped {fname} ({idx}): {e}")

            pbar.set_postfix(resized=resized_count, skipped=skipped_count)

        folder_end = time.time()
        folder_duration = folder_end - folder_start
        print(f"Finished {split}/{dtype}: saved {len(ds)-skipped_count}, resized {resized_count}, skipped {skipped_count}.")
        print(f"Time taken: {folder_duration / 60:.2f} minutes\n")

        # Delete Arrow folder to save space
        shutil.rmtree(arrow_path, ignore_errors=True)

    # Cleanup split folder and rename temp folder
    split_folder = hf_local_path / split
    if split_folder.exists():
        shutil.rmtree(split_folder, ignore_errors=True)
    os.rename(temp_split_folder, hf_local_path / split)

overall_end = time.time()
overall_duration = overall_end - overall_start
print(f"Dataset reconstruction complete! Total time: {overall_duration / 60:.2f} minutes")