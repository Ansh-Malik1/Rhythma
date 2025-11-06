import pandas as pd
import os

# ======================================
# CONFIGURATION
# ======================================
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)

REAL_PATH = os.path.join(workspace_root, "scaled_new.csv")
SYNTH_PATH = os.path.join(workspace_root, "synthetic_scaled.csv")
OUTPUT_PATH = os.path.join(workspace_root, "scaled_combined.csv")

TARGET_COL = "readmission_30days"

# ======================================
# STEP 1: LOAD BOTH DATASETS
# ======================================
print("Loading datasets...")
df_real = pd.read_csv(REAL_PATH)
df_synth = pd.read_csv(SYNTH_PATH)

print(f"Real dataset shape: {df_real.shape}")
print(f"Synthetic dataset shape: {df_synth.shape}")

# Drop duplicate flags if exist
if "synthetic_flag" in df_real.columns:
    df_real = df_real.drop(columns=["synthetic_flag"])
if "synthetic_flag" in df_synth.columns:
    df_synth = df_synth.drop(columns=["synthetic_flag"])

# ======================================
# STEP 2: ALIGN COLUMNS
# ======================================
# Ensure both have same columns
missing_in_synth = set(df_real.columns) - set(df_synth.columns)
missing_in_real = set(df_synth.columns) - set(df_real.columns)

if missing_in_synth:
    print(f" Adding missing columns to synthetic dataset: {missing_in_synth}")
    for col in missing_in_synth:
        df_synth[col] = 0

if missing_in_real:
    print(f" Adding missing columns to real dataset: {missing_in_real}")
    for col in missing_in_real:
        df_real[col] = 0

# Reorder columns to match
df_synth = df_synth[df_real.columns]

# ======================================
# STEP 3: CHECK TARGET BALANCE
# ======================================
if TARGET_COL in df_synth.columns:
    print("\nReal data target distribution:")
    print(df_real[TARGET_COL].value_counts(normalize=True))
    print("\nSynthetic data target distribution:")
    print(df_synth[TARGET_COL].value_counts(normalize=True))
else:
    print(" Synthetic data does not contain target column. Copying from real data pattern...")
    df_synth[TARGET_COL] = df_real[TARGET_COL].sample(len(df_synth), replace=True).values

# ======================================
# STEP 4: MERGE BOTH DATASETS
# ======================================
combined_df = pd.concat([df_real, df_synth], axis=0).reset_index(drop=True)
combined_df["synthetic_flag"] = [0] * len(df_real) + [1] * len(df_synth)

print(f"\nCombined dataset shape: {combined_df.shape}")
print(f"Real + Synthetic ratio: {len(df_real)} real / {len(df_synth)} synthetic")

# Shuffle combined dataset
combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# ======================================
# STEP 5: SAVE
# ======================================
combined_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n Combined dataset saved successfully at: {OUTPUT_PATH}")
