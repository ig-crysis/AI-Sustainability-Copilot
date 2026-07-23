import pandas as pd
import joblib

df = pd.read_csv('data/raw/owid_co2.csv', usecols=['country','year','co2_per_capita'])
df = df.dropna(subset=['co2_per_capita'])
latest = df.sort_values('year').groupby('country').last().reset_index()

baselines = {}
for _, row in latest.iterrows():
    key = row['country'].lower().strip()
    annual_t = float(row['co2_per_capita'])
    baselines[key] = {
        'per_capita_monthly_kg':    round(annual_t * 1000 / 12, 2),
        'per_capita_annual_tonnes': round(annual_t, 3),
        'year':                     int(row['year']),
    }

joblib.dump(baselines, 'data/processed/owid_baselines.pkl')
print(f'Saved {len(baselines)} country baselines')
print(f'India: {baselines.get("india")}')
print(f'Germany: {baselines.get("germany")}')
print('[OK] owid_baselines.pkl saved to data/processed/')