import pandas as pd


df_new = pd.read_csv("dataset/processed/new_data.csv")


df_old = pd.read_csv("dataset/processed/heart_failure_patients.csv")
if "icd9_code" in df_old.columns:
    df_new["icd9_code"] = df_old["icd9_code"].astype(str)
else:
    raise ValueError("icd9_code column not found in old dataset!")

print(df_new[["icd9_code"]].head())
df_new.to_csv("dataset/processed/scaled_with_icd9.csv", index=False)
print("ICD9 column successfully added and saved as scaled_with_icd9.csv")

severe_icd9 = ['42821','42823','42833','42843','40491','40493','40291','40413']
moderate_icd9 = ['42820','42822','42830','42831','42832','42840','42841','42842','40211','40411','40401']
mild_icd9 = ['4280','4281','39891','40201']

def map_severity(icd):
    if icd in severe_icd9:
        return 3
    elif icd in moderate_icd9:
        return 2
    elif icd in mild_icd9:
        return 1
    else:
        return 0

df_new["severity"] = df_new["icd9_code"].astype(str).apply(map_severity)
df_new.drop(columns=["icd9_code"], inplace=True)
df_new.drop(columns=["diagnosis"], inplace=True)

df_new.to_csv("dataset/processed/final_data_with_severity.csv", index=False)
print("Severity column added successfully.")