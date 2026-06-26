"""
NYC failure analysis: compute AUROC stratified by user history length.
Reuses train.py foursquare parsing logic directly.
Run on server: /home/hello/miniconda3/envs/oyq_v01/bin/python nyc_stratify.py
"""

import sys
sys.path.insert(0, '/home/hello/ouyangqi')

import re
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

# ── Config ───────────────────────────────────────────────────────────────────
TRAIN_PARQUET = '/home/hello/ouyangqi/data/foursquare_nyc/train.parquet'
TEST_PARQUET  = '/home/hello/ouyangqi/data/foursquare_nyc/test.parquet'
CKPT_DIR      = Path('/home/hello/ouyangqi/results/foursquare_nyc/seed_42/numosim/seed_42')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

BUCKETS = [
    ('≤10',   0,  10),
    ('11–20', 11, 20),
    ('21–30', 21, 30),
    ('>30',  31, 9999),
]

# ── Foursquare parsing (copied from train.py) ─────────────────────────────
CHECKIN_RE = re.compile(
    r'At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user (\d+) visited POI id (\d+)'
    r' which is a .+? and has Category id (\d+)\.'
)

def _discretize_dwell(dwell_min):
    thresholds = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420, 480]
    for i, t in enumerate(thresholds):
        if dwell_min <= t:
            return i
    return 15

def parse_foursquare(path):
    """Parse foursquare parquet → {user_id: [(date_str, list_of_episodes, label)]}"""
    from langtraj_osr.core.episode import SemanticEpisode as SE

    df = pd.read_parquet(path)
    user_checkins = defaultdict(list)
    for txt in df['inputs']:
        for m in CHECKIN_RE.finditer(str(txt)):
            ts_str, uid, poi_id, cat_id = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            ts = pd.Timestamp(ts_str)
            user_checkins[uid].append((ts, poi_id, cat_id))

    # Get labels per trip_id if available
    label_map = {}
    if 'trip_id' in df.columns and 'label' in df.columns:
        for _, row in df.iterrows():
            label_map[str(row.get('trip_id', ''))] = int(row.get('label', 0))

    trajs = []  # list of (user_id, trip_id, label, episodes_tensor, valid_len)
    for uid, checkins in user_checkins.items():
        checkins.sort(key=lambda x: x[0])
        by_date = defaultdict(list)
        for ts, poi, cat in checkins:
            by_date[ts.date().isoformat()].append((ts, poi, cat))

        for date_str, day_checkins in by_date.items():
            if len(day_checkins) < 3:
                continue
            trip_id = f"{uid}_{date_str}"
            label = label_map.get(trip_id, 0)

            eps_feats = []
            for i, (ts, poi, cat) in enumerate(day_checkins):
                dwell_min = (day_checkins[i+1][0] - ts).total_seconds() / 60.0 \
                            if i + 1 < len(day_checkins) else 30.0
                dwell_min = min(max(dwell_min, 1.0), 480.0)
                trans = 0 if dwell_min < 20 else (2 if dwell_min < 60 else 1)
                zone = int(hashlib.md5(str(poi).encode()).hexdigest()[:8], 16) % (2**31)
                time_bin = ts.hour * 7 + ts.dayofweek
                dwell_bin = _discretize_dwell(dwell_min)
                total_per_day = len(day_checkins)
                avg_per_day = sum(len(v) for v in by_date.values()) / max(len(by_date), 1)
                tlc = min(float(total_per_day) / max(avg_per_day, 1.0), 20.0)
                eps_feats.append([zone, int(cat) % 64, time_bin, dwell_bin, trans, round(tlc, 4), 0, 0])

            # Pad/truncate to 64
            eps_feats = eps_feats[:64]
            vlen = len(eps_feats)
            pad = 64 - vlen
            arr = np.array(eps_feats, dtype=np.float32)
            if pad > 0:
                arr = np.concatenate([arr, np.zeros((pad, 8), dtype=np.float32)], axis=0)
            trajs.append((uid, trip_id, label, arr, vlen))

    return trajs

def main():
    print("Parsing train data (for user history counts)...")
    train_trajs = parse_foursquare(TRAIN_PARQUET)
    # Count normal trips per user in training set
    user_hist = defaultdict(int)
    for uid, _, label, _, _ in train_trajs:
        if label == 0:
            user_hist[uid] += 1
    print(f"  {len(user_hist)} users, normal trips: "
          f"min={min(user_hist.values())}, "
          f"mean={np.mean(list(user_hist.values())):.1f}, "
          f"max={max(user_hist.values())}")

    print("Parsing test data...")
    test_trajs = parse_foursquare(TEST_PARQUET)
    print(f"  {len(test_trajs)} test trajectories")

    # Load model
    print("Loading model checkpoint...")
    from langtraj_osr.models.langtraj_osr import LangTrajOSR
    from langtraj_osr.core.concepts import get_all_definitions

    ckpt = torch.load(CKPT_DIR / 'best_model.pt', map_location=DEVICE)
    concept_defs = get_all_definitions(include_paraphrases=False)
    config = ckpt.get('config', {})

    from langtraj_osr.models.langtraj_osr import LangTrajConfig
    cfg = LangTrajConfig()
    model = LangTrajOSR(cfg).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # Load user prototypes into model
    user_protos = ckpt.get('user_prototypes', {})
    if user_protos:
        model.user_history.load_prototypes(user_protos)
        print(f"  Loaded prototypes for {len(user_protos)} users")
    else:
        print("  WARNING: no user_prototypes in checkpoint")

    # Inference
    print("Running inference on test set...")
    scores = []
    labels = []
    uids = []

    with torch.no_grad():
        for uid, trip_id, label, arr, vlen in test_trajs:
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,64,8)
            pad_mask = torch.zeros(1, 64, dtype=torch.bool, device=DEVICE)
            pad_mask[0, vlen:] = True

            try:
                out = model(x, pad_mask=pad_mask, user_ids=[uid])
                e_norm = out['E_norm'].item()
            except Exception as e:
                e_norm = 0.0

            scores.append(e_norm)
            labels.append(1 if label > 0 else 0)
            uids.append(uid)

    scores = np.array(scores)
    labels = np.array(labels)
    uids = np.array(uids)

    overall_auroc = roc_auc_score(labels, scores)
    print(f"\nOverall AUROC: {overall_auroc:.4f}  (reference: 0.679)")

    # Stratified results
    print("\n=== Stratified by User History Length ===")
    print(f"{'Bucket':10s} {'N Users':>8s} {'N Trajs':>8s} {'N Anom':>8s} {'AUROC':>8s}")
    print("-" * 52)

    results = {}
    for bucket_label, lo, hi in BUCKETS:
        mask = np.array([lo <= user_hist.get(uid, 0) <= hi for uid in uids])
        n_traj = int(mask.sum())
        n_users = len(set(uids[mask]))
        n_anom = int(labels[mask].sum())
        if n_traj == 0 or n_anom == 0 or n_anom == n_traj:
            auroc_val = float('nan')
        else:
            auroc_val = roc_auc_score(labels[mask], scores[mask])
        results[bucket_label] = {
            'n_users': n_users, 'n_trajs': n_traj,
            'n_anom': n_anom, 'auroc': round(auroc_val, 4) if not np.isnan(auroc_val) else None
        }
        print(f"{bucket_label:10s} {n_users:>8d} {n_traj:>8d} {n_anom:>8d} "
              f"{'N/A' if np.isnan(auroc_val) else f'{auroc_val:.4f}':>8s}")

    out = {
        'overall_auroc': round(overall_auroc, 4),
        'user_hist_stats': {
            'total_users': len(user_hist),
            'min': int(min(user_hist.values())),
            'mean': round(float(np.mean(list(user_hist.values()))), 2),
            'median': round(float(np.median(list(user_hist.values()))), 1),
            'max': int(max(user_hist.values())),
        },
        'buckets': results,
    }
    out_path = CKPT_DIR / 'nyc_stratified.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
