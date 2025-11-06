import os
import json
import csv
from pathlib import Path

output_root = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/A_NEW_TEST")
splits = ["train", "validation"]
coco_splits = ["train2017", "val2017"]

for sp, split in enumerate(splits):
    json_file = output_root / f"captions_{coco_splits[sp]}.json"
    split_folder = output_root / split / "coco"

    # Load JSON file
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    images = json_data["images"]
    annotations = json_data["annotations"]

    # Build mapping of annotations per image_id
    captions_by_image = {}
    for ann in annotations:
        img_id = ann["image_id"]
        caption = ann.get("caption", "").strip()
        if caption:
            captions_by_image.setdefault(img_id, []).append(caption)

    # Prepare CSV path
    csv_path = output_root / f"{split}_coco_metadata.csv"

    # Define CSV header, by hand :/
    header = [
        "id",
        "license",
        "file_name",
        "coco_url",
        "height",
        "width",
        "date_captured",
        "flickr_url",
        "captions"
    ]

    # Write metadata to CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        for img in images:
            img_id = img["id"]
            captions = captions_by_image.get(img_id, [])
            combined_captions = " || ".join(captions) if captions else ""

            # Add pathing relative to dataset root
            file_path = f"{split}/coco/{img['file_name']}"

            row = {
                "id": img_id,
                "license": img.get("license", ""),
                "file_name": file_path,
                "coco_url": img.get("coco_url", ""),
                "height": img.get("height", ""),
                "width": img.get("width", ""),
                "date_captured": img.get("date_captured", ""),
                "flickr_url": img.get("flickr_url", ""),
                "captions": combined_captions
            }

            writer.writerow(row)

    print(f"Saved CSV for {split}: {csv_path}")
