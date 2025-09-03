import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_file = os.path.join(BASE_DIR,"data", "raw", "global_food_wastage_dataset.csv")
processed_file = os.path.join(BASE_DIR, "data", "processed", "food_waste_clean.csv")

raw_sample = os.path.join(BASE_DIR, "data", "raw", "sample_global_wastage_dataset.csv")
processed_sample = os.path.join(BASE_DIR, "data", "processed", "sample_food_waste_clean.csv")

if os.path.exists(raw_file):
    df_raw = pd.read_csv(raw_file)
    df_raw.head(10).to_csv(raw_sample, index=False)
    print(f"Raw sample created: {raw_sample}")
else:
    print(f"File not found: {raw_file}")

if os.path.exists(processed_file):
    df_processed = pd.read_csv(processed_file)
    df_processed.head(10).to_csv(processed_sample, index=False)
    print(f"Processed sample created: {processed_sample}")
else:
    print(f"File not found: {processed_file}")

print("Done!")
