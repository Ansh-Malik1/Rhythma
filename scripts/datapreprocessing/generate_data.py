import pandas as pd
import numpy as np
import os

# ======================================
# CONFIGURATION
# ======================================
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
INPUT_PATH = os.path.join(workspace_root, "scaled_new.csv")
OUTPUT_PATH = os.path.join(workspace_root, "synthetic_scaled.csv")

np.random.seed(42)

# ======================================
# STEP 1: Load dataset
# ======================================
df = pd.read_csv(INPUT_PATH)
print(f"Original shape: {df.shape}")

# ======================================
# STEP 2: Add small Gaussian noise to numeric columns
# ======================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {numeric_cols}")

synthetic_data = df.copy()

for col in numeric_cols:
    noise = np.random.normal(0, df[col].std() * 0.05, size=len(df))
    synthetic_data[col] = df[col] + noise

# Clip negative values if any
for col in numeric_cols:
    synthetic_data[col] = synthetic_data[col].clip(lower=0)

# ======================================
# STEP 3: Add identifier
# ======================================
df["synthetic_flag"] = 0
synthetic_data["synthetic_flag"] = 1

# Combine both
combined = pd.concat([df, synthetic_data], axis=0).reset_index(drop=True)
print(f"Combined dataset shape: {combined.shape}")

# ======================================
# STEP 4: Save
# ======================================
combined.to_csv(OUTPUT_PATH, index=False)
print(f"Synthetic data generated and saved to: {OUTPUT_PATH}")
