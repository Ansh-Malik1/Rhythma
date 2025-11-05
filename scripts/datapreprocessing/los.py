import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))

ORIGINAL_PATH = os.path.join(workspace_root, "dataset", "processed", "heart_failure_patients.csv")
ENCODED_PATH = os.path.join(workspace_root, "encoded_new.csv")  # encoded version
OUTPUT_PATH = os.path.join(workspace_root, "scaled_new.csv")


print("Loading datasets...")
df_orig = pd.read_csv(ORIGINAL_PATH, parse_dates=["admittime", "dischtime"], low_memory=False)
df_encoded = pd.read_csv(ENCODED_PATH)

print(f"Original: {df_orig.shape}, Encoded: {df_encoded.shape}")

if len(df_orig) != len(df_encoded):
    print(f"Warning: Row mismatch ({len(df_orig)} vs {len(df_encoded)}). Aligning by index.")
    df_orig = df_orig.head(len(df_encoded))

print("Computing length of stay...")
df_orig["length_of_stay"] = (df_orig["dischtime"] - df_orig["admittime"]).dt.total_seconds() / (24 * 3600)
df_orig["length_of_stay"] = df_orig["length_of_stay"].clip(lower=0)  # no negative stays
df_orig["length_of_stay"] = df_orig["length_of_stay"].fillna(df_orig["length_of_stay"].median()).round(2)


df_encoded["length_of_stay"] = df_orig["length_of_stay"].values


median_los = df_encoded["length_of_stay"].median()
df_encoded["length_of_stay"].fillna(median_los, inplace=True)


output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
df_encoded.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved dataset with length_of_stay to: {OUTPUT_PATH}")
print(f"Final shape: {df_encoded.shape}")


print("\nLength of stay summary:")
print(df_encoded["length_of_stay"].describe())
