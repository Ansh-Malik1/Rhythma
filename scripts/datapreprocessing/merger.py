import pandas as pd
import numpy as np


ADMISSIONS_PATH = "dataset/processed/admissions_202208161605_with_readmission_flag.csv"  # already has readmission flag
DIAGNOSES_PATH = "dataset/raw/diagnoses_icd_202208161605.csv"
PATIENTS_PATH = "dataset/raw/patients_202208161605.csv"
LABEVENTS_PATH = "dataset/raw/labevents_202208161605.csv"

OUTPUT_PATH = "dataset/processed/merged_master.csv"


print("Loading base tables...")

admissions = pd.read_csv(ADMISSIONS_PATH)
diagnoses = pd.read_csv(DIAGNOSES_PATH)
patients = pd.read_csv(PATIENTS_PATH)

print(f"Admissions: {admissions.shape}, Diagnoses: {diagnoses.shape}, Patients: {patients.shape}")


print("Merging admissions and diagnoses...")
merged_df = admissions.merge(diagnoses, on=["subject_id", "hadm_id"], how="left")
print(f"After diagnoses merge: {merged_df.shape}")


print("Adding patient demographics...")

patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")
merged_df = merged_df.merge(
    patients[["subject_id", "gender", "dob", "dod"]],
    on="subject_id", how="left"
)

merged_df["admittime"] = pd.to_datetime(merged_df["admittime"], errors="coerce")

merged_df["age"] = np.nan


mask = merged_df["admittime"].notna() & merged_df["dob"].notna()
valid_indices = merged_df[mask].index

for idx in valid_indices:
    try:
        admittime = merged_df.loc[idx, "admittime"]
        dob = merged_df.loc[idx, "dob"]
        if pd.notna(admittime) and pd.notna(dob):
            age_days = (admittime - dob).days
            age = age_days / 365.25
            if age >= 0 and age <= 150:  # Reasonable age range
                merged_df.loc[idx, "age"] = age
    except (OverflowError, ValueError, TypeError):
        continue

merged_df.loc[merged_df["age"] < 0, "age"] = np.nan
print(f"After patient merge: {merged_df.shape}")

import csv
from collections import defaultdict
import math

print("Aggregating lab events manually (no pandas groupby)...")


lab_means = defaultdict(list)

with open(LABEVENTS_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        try:
            hadm = row["hadm_id"]
            if not hadm:
                continue
            val = float(row["valuenum"])
            if not math.isfinite(val) or abs(val) > 1e6:
                continue
            lab_means[hadm].append(val)
        except (KeyError, ValueError):
            continue
        if i % 5_000_000 == 0 and i > 0:
            print(f"Read {i:,} rows...")


rows = []
for hadm, vals in lab_means.items():
    n = len(vals)
    if n == 0:
        continue
    mean_v = sum(vals) / n
    min_v = min(vals)
    max_v = max(vals)
    var = sum((v - mean_v) ** 2 for v in vals) / n if n > 1 else 0
    std_v = math.sqrt(var)
    rows.append((int(hadm), mean_v, min_v, max_v, std_v))

lab_summary = pd.DataFrame(rows, columns=["hadm_id", "lab_mean", "lab_min", "lab_max", "lab_std"])
print(f"Lab summary ready: {lab_summary.shape}")



print("Merging lab summary with main data...")

merged_df = merged_df.merge(lab_summary, on="hadm_id", how="left")
merged_df.drop_duplicates(subset=["subject_id", "hadm_id"], inplace=True)
merged_df.reset_index(drop=True, inplace=True)

print(f"After lab merge: {merged_df.shape}")

merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"Merged dataset saved successfully at: {OUTPUT_PATH}")
print(f"Final shape: {merged_df.shape}")
