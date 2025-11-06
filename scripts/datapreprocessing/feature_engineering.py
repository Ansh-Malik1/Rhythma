import pandas as pd

df = pd.read_csv(r"dataset\processed\final_data_with_severity.csv")

df["lab_range"] = df["lab_max"] - df["lab_min"]
df["lab_ratio"] = (df["lab_mean"] / (df["lab_std"] + 1e-5)).clip(0, 100)
df["stay_x_severity"] = df["length_of_stay"] * df["severity"]
df["lab_stability"] = (df["lab_std"] / (df["lab_mean"].abs() + 1e-5)).clip(0, 50)

df.to_csv("scaled_enhanced.csv", index=False)
print("Enhanced dataset saved as scaled_enhanced.csv")