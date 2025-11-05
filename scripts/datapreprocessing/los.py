import pandas as pd
import numpy as np
import os

# Get the script directory and resolve paths relative to workspace root
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))

ORIGINAL_PATH = os.path.join(workspace_root, "dataset", "processed", "heart_failure_patients.csv")
ENCODED_PATH = os.path.join(workspace_root, "encoded_new.csv")  # encoded version
OUTPUT_PATH = os.path.join(workspace_root, "cleaned_new_encoded_with_los.csv")


print("Loading datasets...")
df_orig = pd.read_csv(ORIGINAL_PATH, parse_dates=["admittime", "dischtime"], low_memory=False)
df_encoded = pd.read_csv(ENCODED_PATH)

print(f"Original: {df_orig.shape}, Encoded: {df_encoded.shape}")


print("Computing length of stay...")
df_orig["length_of_stay"] = (df_orig["dischtime"] - df_orig["admittime"]).dt.total_seconds() / (24 * 3600)
df_orig["length_of_stay"] = df_orig["length_of_stay"].clip(lower=0)  # avoid negatives
df_orig["length_of_stay"] = df_orig["length_of_stay"].fillna(0).round(2)


los_df = df_orig[["hadm_id", "length_of_stay"]].drop_duplicates()


if "hadm_id" in df_encoded.columns:
    merged_df = df_encoded.merge(los_df, on="hadm_id", how="left")
else:
    print("Warning: hadm_id not found in encoded dataset; merge may not align perfectly.")
    merged_df = df_encoded.copy()
    merged_df["length_of_stay"] = np.nan

print(f"After merging: {merged_df.shape}")


median_los = merged_df["length_of_stay"].median()
merged_df["length_of_stay"] = merged_df["length_of_stay"].fillna(median_los)
print(f"Filled missing LOS with median = {median_los:.2f} days")


output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir:  # Only create directory if there's a directory path
    os.makedirs(output_dir, exist_ok=True)
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved dataset with length_of_stay to: {OUTPUT_PATH}")
