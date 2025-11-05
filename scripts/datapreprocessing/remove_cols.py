import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))
INPUT_PATH = os.path.join(workspace_root, "cleaned.csv")
OUTPUT_PATH = os.path.join(workspace_root, "cleaned_2.0.csv")

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)
print(f"Initial shape: {df.shape}")

drop_cols = [
    # Identifiers
    "row_id_x", "row_id_y", "subject_id", "hadm_id", "seq_num","icd9_code"
    # Timestamps
    "admittime", "dischtime", "deathtime", "dob", "dod", "edregtime", "edouttime",
    # Low-utility demographics
    "language", "religion", "marital_status", "ethnicity"
]


existing_drops = [col for col in drop_cols if col in df.columns]
df.drop(columns=existing_drops, inplace=True, errors="ignore")

print(f"Dropped {len(existing_drops)} columns.")
print(f"Remaining columns: {len(df.columns)}")

target_col = "readmission_30days"
if target_col in df.columns:
    cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[cols]

output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir:  # Only create directory if there's a directory path
    os.makedirs(output_dir, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned dataset saved successfully at: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")


print("\nColumns after cleaning:")
print(df.columns.tolist())
