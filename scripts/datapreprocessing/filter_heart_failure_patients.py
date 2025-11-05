import pandas as pd
import os

def filter_heart_failure_patients(admissions_path,diagnoses_path,output_path):
    hf_icd9=[
        '39891','40201','40211','40291','40401','40403','40411','40413',
        '40491','40493','4280','4281','42820','42821','42822','42823',
        '42830','42831','42832','42833','42840','42841','42842','42843','4289'
    ]
    
    admissions = pd.read_csv(admissions_path)
    diagnoses = pd.read_csv(diagnoses_path)
    
    print(f"Loaded {len(admissions)} admissions and {len(diagnoses)} diagnoses")
    
    merged = pd.merge(admissions,diagnoses,on='hadm_id',how='inner')
    
    hf_patients = merged[merged['icd9_code'].astype(str).isin(hf_icd9)]
    print(f"Found {len(hf_patients)} heart failure patients")
    
    hf_patients = hf_patients.drop_duplicates(subset=['hadm_id'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hf_patients.to_csv(output_path, index=False)
    print(f"Saved filtered HF dataset to: {output_path}")
    
    
if __name__ == "__main__":
    admissions_file = r"dataset/processed/admissions_202208161605_with_readmission_flag.csv"
    diagnoses_file = r"dataset/raw/diagnoses_icd_202208161605.csv"
    output_file = r"dataset/processed/heart_failure_patients.csv"
    filter_heart_failure_patients(admissions_file, diagnoses_file, output_file)