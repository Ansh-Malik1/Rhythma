import pandas as pd
import os
import numpy as np
def add_readmission_flag(input_path,output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
   
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df = df.sort_values(by=['subject_id', 'admittime'])
    df['readmission_30days'] = 0
    for patient_id,group in df.groupby('subject_id'):
        group = group.sort_values(by='admittime')
        dischtime = group['dischtime'].values
        admit_times = group['admittime'].values
        
        for i in range(1,len(group)):
            diff_days = (admit_times[i] - dischtime[i - 1]) / np.timedelta64(1, 'D')
            if 0 <= diff_days <= 30:
                df.loc[group.index[i], 'readmission_30days'] = 1
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved file with 30-day readmission flag: {output_path}")
    
    
if __name__ == "__main__":
    input_path = r"dataset/raw/admissions_202208161605.csv"
    output_path = r"dataset/admissions_202208161605_with_readmission_flag.csv"
    add_readmission_flag(input_path, output_path)