from datasets import load_from_disk
from pathlib import Path
import shutil

# Where to reconstruct
reconstructed_root = Path("reconstructed_dataset")
reconstructed_root.mkdir(exist_ok=True)

splits = ["train", "validation"]
types = ["real", "fake"]

# Local clone of repo
hf_local_path = Path("DS6050_Ai_Detection")  # path where arrows are stored (path you just git cloned)

for split in splits:
    for dtype in types:
        arrow_path = hf_local_path / f"{split}" / dtype
        print(f"Loading Arrow dataset from {arrow_path} ...")
        ds = load_from_disk(arrow_path)

        # Create output folder
        out_folder = reconstructed_root / split / dtype
        out_folder.mkdir(parents=True, exist_ok=True)

        print(f"Saving images to {out_folder}...")
        for row in ds:
            img = row["image"]
            fname = Path(img.path).name
            shutil.copy(img.path, out_folder / fname)

        print(f"Saved {len(ds)} images for {split}/{dtype}.\n")

print("Dataset reconstruction complete!")