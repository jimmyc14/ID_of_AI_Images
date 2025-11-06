import pandas as pd
from pathlib import Path
from datasets import Dataset, Features, Value, Image
import os

arrow_root = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/arrow")  # where Arrow datasets will be saved
data_root = Path(r"C:/Users/Jimmy/OneDrive/Desktop/AI-GenBench/trimmed_main_data")   # or "validation"

splits = ["train", "validation"]
types = ["real", "fake"]

for split in splits:
    for dtype in types:
        print(f"Processing {split} {dtype} images...")
        
        # CSV path
        csv_path = Path(os.path.join(data_root,f"{split}_{dtype}_metadata.csv"))
        df = pd.read_csv(csv_path)

        # Ensure the image paths are absolute or relative correctly
        df["image"] = df["filename"].apply(lambda f: str(os.path.join(data_root, Path(split) / dtype / Path(f).name)))

        # Define features: "image" is the actual image, others can be left as Value("string") or numeric
        features = {}
        for col in df.columns:
            if col == "image":
                features[col] = Image()
            else:
                features[col] = Value("string")
        
        ds = Dataset.from_pandas(df, features=Features(features), preserve_index=False)

        # Save to disk
        save_path = arrow_root / split/ dtype
        ds.save_to_disk(save_path)
        print(f"Saved {split} {dtype} dataset to {save_path}\n")


