import pandas as pd


df = pd.read_csv("encoded_output.csv")

datetime_cols = ['admittime', 'dischtime', 'edregtime', 'edouttime']
for col in datetime_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')  # coerce invalid formats to NaT


df['length_of_stay'] = (df['dischtime'] - df['admittime']).dt.days
df['admit_hour'] = df['admittime'].dt.hour
df['admit_weekday'] = df['admittime'].dt.weekday
df['admit_month'] = df['admittime'].dt.month


df['ed_duration'] = (df['edouttime'] - df['edregtime']).dt.total_seconds() / 3600  # hours

df = df.drop(columns=datetime_cols)


df.to_csv("dataset/processed/date_time_fixed.csv", index=False)

print("Datetime features extracted and file saved successfully.")