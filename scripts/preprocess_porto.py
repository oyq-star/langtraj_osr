"""
Preprocess Porto taxi parquet data into SemanticTrajectory .pt files,
then build MobDef-Bench benchmark and start training.

Porto data schema: timestamp, trip_id, call_type, origin_call, origin_stand,
                   taxi_id, day_type, speed, longitude, latitude
Each row = one GPS point within a trip.
"""
import sys
import os
import json
import time
import logging
import pickle

import numpy as np
import pandas as pd
import torch

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Step 1: Load Porto parquet and reconstruct trip-level format
# ---------------------------------------------------------------

PORTO_PARQUET = "E:/skills_projects/idea1_discovery/data/porto/data/raw/porto_taxi.parquet"
OUTPUT_DIR    = "E:/skills_projects/idea1_discovery/data/processed_porto"
N_TAXIS       = 200   # number of distinct taxis (proxy for users)
MIN_TRIPS     = 10    # min trips per taxi to include
MAX_TRIPS     = 80    # max trips per taxi (sample if more)
MIN_POINTS    = 4     # min GPS points per trip

logger.info("Loading Porto parquet (%s)...", PORTO_PARQUET)
df = pd.read_parquet(PORTO_PARQUET)
logger.info("  Raw rows: %d, taxis: %d, trips: %d",
            len(df), df['taxi_id'].nunique(), df['trip_id'].nunique())

# Sort by taxi and timestamp
df = df.sort_values(['taxi_id', 'timestamp']).reset_index(drop=True)

# Filter taxis with enough trips
taxi_trips = df.groupby('taxi_id')['trip_id'].nunique()
eligible = taxi_trips[taxi_trips >= MIN_TRIPS].index
logger.info("  Taxis with >= %d trips: %d", MIN_TRIPS, len(eligible))

# Sample top N_TAXIS by trip count
top_taxis = taxi_trips[eligible].nlargest(N_TAXIS).index
df = df[df['taxi_id'].isin(top_taxis)].copy()
logger.info("  Using %d taxis, %d rows", df['taxi_id'].nunique(), len(df))

# Reconstruct trip-level DataFrame with POLYLINE column
logger.info("Reconstructing trip-level data with POLYLINE...")
trip_records = []
for (taxi_id, trip_id), grp in df.groupby(['taxi_id', 'trip_id']):
    grp = grp.sort_values('timestamp')
    pts = list(zip(grp['longitude'].tolist(), grp['latitude'].tolist()))
    if len(pts) < MIN_POINTS:
        continue
    trip_records.append({
        'TRIP_ID':      str(trip_id),
        'TAXI_ID':      str(taxi_id),
        'TIMESTAMP':    int(grp['timestamp'].iloc[0]),
        'POLYLINE':     pts,   # list of [lon, lat]
        'MISSING_DATA': False,
        'DAY_TYPE':     grp['day_type'].iloc[0] if 'day_type' in grp.columns else 'A',
    })

porto_df = pd.DataFrame(trip_records)
logger.info("  Trip-level rows: %d", len(porto_df))

# Per taxi: limit to MAX_TRIPS trips
def sample_trips(group):
    if len(group) > MAX_TRIPS:
        return group.sample(MAX_TRIPS, random_state=42)
    return group

porto_df = porto_df.groupby('TAXI_ID', group_keys=False).apply(sample_trips).reset_index(drop=True)
logger.info("  After sampling: %d trips across %d taxis",
            len(porto_df), porto_df['TAXI_ID'].nunique())

# ---------------------------------------------------------------
# Step 2: Tokenize with TrajectoryTokenizer
# ---------------------------------------------------------------
logger.info("Tokenizing via TrajectoryTokenizer.tokenize_porto()...")
from langtraj_osr.core.tokenizer import TrajectoryTokenizer

# tokenize_porto expects POLYLINE as JSON string or list
tokenizer = TrajectoryTokenizer()

# Patch the method to use our 'TAXI_ID' as user_id
def tokenize_porto_patched(raw_df):
    from langtraj_osr.core.episode import SemanticEpisode, SemanticTrajectory
    import math, hashlib

    R = 6_371_000.0
    def haversine(lat1, lon1, lat2, lon2):
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = (math.sin(dphi/2)**2
             + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2)
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

    def zone_id(lat, lon, res=0.005):
        gl = int(round(lat/res))
        gn = int(round(lon/res))
        h = hashlib.md5(f"{gl},{gn}".encode()).hexdigest()
        return int(h[:8], 16) % (2**31)

    def infer_trans(spd):
        if spd < 2.0: return 0
        if spd < 4.0: return 2
        if spd < 10.0: return 2
        return 1

    trajectories = []
    for _, row in raw_df.iterrows():
        polyline = row['POLYLINE']
        if not polyline or len(polyline) < 2:
            continue
        base_ts = int(row['TIMESTAMP'])
        step_s  = 15
        taxi_id = str(row['TAXI_ID'])
        trip_id = str(row['TRIP_ID'])

        dists = [haversine(polyline[i-1][1], polyline[i-1][0],
                           polyline[i][1],   polyline[i][0])
                 for i in range(1, len(polyline))]
        avg_dist = float(np.mean(dists)) if dists else 1.0

        episodes = []
        subsample = max(1, len(polyline) // 20)
        for idx in range(0, len(polyline), subsample):
            lon, lat = polyline[idx]
            ts = pd.Timestamp(base_ts + idx * step_s, unit='s')
            z  = zone_id(lat, lon)
            tb = ts.hour * 7 + ts.dayofweek
            db = tokenizer._discretize_dwell(subsample * step_s / 60.0)
            if idx > 0:
                pi = max(0, idx - subsample)
                plon, plat = polyline[pi]
                sd = haversine(plat, plon, lat, lon)
                spd = sd / (subsample * step_s) if subsample * step_s > 0 else 0
                trans = infer_trans(spd)
                tlc = sd / avg_dist if avg_dist > 0 else 1.0
            else:
                trans = 1
                tlc   = 1.0
            episodes.append(SemanticEpisode(
                zone_id=z, poi_role=0, time_bin=tb, dwell_bin=db,
                transition_type=trans,
                trip_length_change=round(tlc, 4),
                event_flag=0, companion_flag=0,
            ))
        if episodes:
            trajectories.append(SemanticTrajectory(
                episodes=episodes, user_id=taxi_id,
                trip_id=trip_id, label=0,
            ))
    return trajectories

raw_trajectories = tokenize_porto_patched(porto_df)
logger.info("  Tokenized %d trajectories", len(raw_trajectories))

# ---------------------------------------------------------------
# Step 3: Build MobDef-Bench (split + inject anomalies)
# ---------------------------------------------------------------
logger.info("Building MobDef-Bench benchmark with anomaly injection...")
import tempfile
from langtraj_osr.benchmark.benchmark_builder import MobDefBenchBuilder

builder = MobDefBenchBuilder(data_dir='/tmp/unused', output_dir='/tmp/unused_out', seed=42)

# Directly call internal pipeline (bypassing _load_or_generate)
raw_trajectories = builder._tokenize(raw_trajectories)
train_trajs, val_trajs, test_trajs = builder._split_by_user(raw_trajectories)

# Inject anomalies into each split
benchmark = builder._build_benchmark_from_splits(
    'porto', train_trajs, val_trajs, test_trajs
)

def combine(split):
    return split.normal + split.anomalous

all_train = combine(benchmark.train)
all_val   = combine(benchmark.val)
all_test  = combine(benchmark.test)

logger.info("  train=%d (norm=%d anom=%d), val=%d, test=%d",
            len(all_train), len(benchmark.train.normal), len(benchmark.train.anomalous),
            len(all_val), len(all_test))

# ---------------------------------------------------------------
# Step 4: Save as JSON for MobDefBenchDataModule.load_dataset()
# ---------------------------------------------------------------
import dataclasses

def serialize_traj(t):
    return {
        'user_id': t.user_id,
        'trip_id': t.trip_id,
        'label':   t.label,
        'primitive_labels': t.primitive_labels,
        'episodes': [list(dataclasses.astuple(ep)) for ep in t.episodes],
    }

os.makedirs(OUTPUT_DIR, exist_ok=True)
for split_name, trajs in [('train', all_train), ('val', all_val), ('test', all_test)]:
    path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
    with open(path, 'w') as f:
        json.dump([serialize_traj(t) for t in trajs], f)
    logger.info("  Saved %s: %d trajectories -> %s", split_name, len(trajs), path)

# Save user histories (normal train trajectories)
user_histories = {}
for t in all_train:
    if t.label == 0:
        user_histories.setdefault(t.user_id, []).append(serialize_traj(t))
with open(os.path.join(OUTPUT_DIR, 'user_histories.json'), 'w') as f:
    json.dump(user_histories, f)
logger.info("  Saved user_histories for %d users", len(user_histories))

logger.info("Porto preprocessing complete! Output: %s", OUTPUT_DIR)
logger.info("  train=%d  val=%d  test=%d", len(all_train), len(all_val), len(all_test))
