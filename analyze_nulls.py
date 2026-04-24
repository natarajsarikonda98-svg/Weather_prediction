import pandas as pd
import sys
import os

path = r'd:\Teeside\Sem-4\weather_ml_upgrade\data\raw\weather_master_dataset.csv'
if not os.path.exists(path):
    print(f"Error: {path} not found")
    sys.exit(1)

print(f"Analyzing {path}...")
# Optimization: only read first 2M rows if it's huge, but we need the whole thing
df = pd.read_csv(path)
print(f"Shape: {df.shape}")
print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal Missing: {missing.sum()}")

print("\nRows per region:")
print(df['region'].value_counts())

print("\nYearly missing values (Sum of all nulls in that year):")
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['year'] = df['datetime'].dt.year
yearly_nulls = df.groupby('year').apply(lambda x: x.isnull().sum().sum())
print(yearly_nulls)
