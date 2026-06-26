"""Convert Porto parquet to trip-level pickle (no slots, no pyarrow on server needed)."""
import pandas as pd
import numpy as np
import pickle, os

PORTO_PARQUET = 'E:/skills_projects/idea1_discovery/data/porto/data/raw/porto_taxi.parquet'
OUTPUT_PKL    = 'E:/skills_projects/idea1_discovery/data/porto_trips.pkl'
N_TAXIS  = 200
MIN_TRIPS = 10
MAX_TRIPS = 80
MIN_POINTS = 4

print("Loading parquet...")
df = pd.read_parquet(PORTO_PARQUET)
print(f"  rows={len(df)}  taxis={df['taxi_id'].nunique()}")

df = df.sort_values(['taxi_id', 'timestamp']).reset_index(drop=True)
taxi_trips = df.groupby('taxi_id')['trip_id'].nunique()
top_taxis  = taxi_trips[taxi_trips >= MIN_TRIPS].nlargest(N_TAXIS).index
df = df[df['taxi_id'].isin(top_taxis)].copy()

print(f"  Using {df['taxi_id'].nunique()} taxis, {len(df)} rows")
print("Building trip-level records (POLYLINE)...")

records = []
for (taxi_id, trip_id), grp in df.groupby(['taxi_id', 'trip_id']):
    grp = grp.sort_values('timestamp')
    pts = list(zip(grp['longitude'].tolist(), grp['latitude'].tolist()))
    if len(pts) < MIN_POINTS:
        continue
    records.append({
        'TAXI_ID':   str(taxi_id),
        'TRIP_ID':   str(trip_id),
        'TIMESTAMP': int(grp['timestamp'].iloc[0]),
        'POLYLINE':  pts,
    })

porto_df = pd.DataFrame(records)
def sample_trips(g):
    return g.sample(min(len(g), MAX_TRIPS), random_state=42)
porto_df = porto_df.groupby('TAXI_ID', group_keys=False).apply(sample_trips).reset_index(drop=True)
print(f"  {len(porto_df)} trips from {porto_df['TAXI_ID'].nunique()} taxis")

# Save as pickle
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(porto_df, f)

size = os.path.getsize(OUTPUT_PKL) / 1e6
print(f"Saved to {OUTPUT_PKL} ({size:.1f} MB)")
