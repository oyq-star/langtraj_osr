#!/usr/bin/env python3
"""Porto Natural Anomaly Mining — Case Study for Fix G.

Mine the top-1% highest E_norm trajectories from Porto test set as "natural anomalies".
For each, retrieve the best-matching concept definition and output a qualitative analysis.

Output: JSON with top-50 anomalous trajectories, their concept assignments, scores,
        and trajectory feature summaries for manual analysis.
"""
import sys, json, logging, tempfile, os, math, hashlib
sys.path.insert(0, '/home/hello/ouyangqi')
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir("/home/hello/ouyangqi")

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TEMPERATURE = 0.07
OUT = Path('/home/hello/ouyangqi/results/porto_case_study')
OUT.mkdir(exist_ok=True, parents=True)

np.random.seed(SEED); torch.manual_seed(SEED)

def ep_dict(t, device):
    return {k: t[:,:,i].long().to(device) if i < 5 or i >= 6 else t[:,:,i].float().to(device)
            for i, k in enumerate(['zone_id','poi_role','time_bin','dwell_bin',
                                    'transition_type','trip_length_change','event_flag','companion_flag'])}


def build_porto():
    """Build Porto benchmark data."""
    from langtraj_osr.benchmark.benchmark_builder import (
        MobDefBenchBuilder, Benchmark, CONCEPT_DEFS,
        SPLIT_SEEN, SPLIT_ZS_COMP, SPLIT_ZS_FAMILY, SPLIT_UNKNOWN,
    )
    from langtraj_osr.core.tokenizer import TrajectoryTokenizer
    from langtraj_osr.core.episode import SemanticEpisode, SemanticTrajectory
    import pandas as pd, pickle

    def _hav(lat1, lon1, lat2, lon2):
        R = 6_371_000.0; p1, p2 = math.radians(lat1), math.radians(lat2)
        a = (math.sin(math.radians(lat2-lat1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2)
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))
    def _zone(lat, lon, res=0.005):
        h = hashlib.md5(f"{int(round(lat/res))},{int(round(lon/res))}".encode()).hexdigest()
        return int(h[:8], 16) % (2**31)
    def _trans(spd): return 0 if spd < 2 else (2 if spd < 10 else 1)

    tok = TrajectoryTokenizer()
    with open('data/porto_trips.pkl', 'rb') as f: df = pickle.load(f)
    pdf = (df.groupby('TAXI_ID', group_keys=False)
             .apply(lambda g: g.sample(min(len(g), 80), random_state=42))
             .reset_index(drop=True))
    trajs = []
    for _, row in pdf.iterrows():
        poly = [(lon, lat) for lon, lat in row['POLYLINE'] if lon == lon and lat == lat]
        if len(poly) < 4: continue
        dists = [_hav(poly[i-1][1], poly[i-1][0], poly[i][1], poly[i][0]) for i in range(1, len(poly))]
        dists = [d for d in dists if d == d and math.isfinite(d)]
        avg_d = max(float(np.mean(dists)) if dists else 1.0, 1.0)
        sub = max(1, len(poly) // 20)
        eps = []
        for idx in range(0, len(poly), sub):
            lon, lat = poly[idx]; ts = pd.Timestamp(int(row['TIMESTAMP']) + idx*15, unit='s')
            db = tok._discretize_dwell(sub * 15 / 60.0)
            if idx > 0:
                pl, pa = poly[max(0, idx-sub)]; sd = _hav(pa, pl, lat, lon)
                if not math.isfinite(sd): sd = 0.0
                spd = sd / (sub*15) if sub*15 > 0 else 0; tr = _trans(spd); tlc = min(sd/avg_d, 20.0)
            else: tr = 1; tlc = 1.0
            eps.append(SemanticEpisode(zone_id=_zone(lat, lon), poi_role=0,
                time_bin=ts.hour*7+ts.dayofweek, dwell_bin=db, transition_type=tr,
                trip_length_change=round(float(tlc), 4), event_flag=0, companion_flag=0))
        if eps:
            trajs.append(SemanticTrajectory(episodes=eps, user_id=row['TAXI_ID'],
                trip_id=row['TRIP_ID'], label=0))

    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to2:
        builder = MobDefBenchBuilder(data_dir=td, output_dir=to2, seed=SEED)
        trajs = builder._tokenize(trajs)
        tr, va, te = builder._split_by_user(trajs)
        bench = Benchmark(dataset_name='porto')
        bench.train.normal, bench.val.normal, bench.test.normal = tr, va, te
        sc = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
        zc = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
        zf = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
        uc = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]
        builder._inject_anomalies(bench.train, tr, sc)
        builder._inject_anomalies(bench.val, va, sc + zc)
        builder._inject_anomalies(bench.test, te, sc + zc + zf + uc)
    return bench


def main():
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split, CONCEPT_NAMES
    from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig
    from langtraj_osr.train import fit_user_routines, _tensor_to_episode_dict, _batch_user_prototypes
    from torch.utils.data import DataLoader

    # Build data
    bench = build_porto()
    # For case study, we use ALL normal test trajectories (no injected anomalies)
    # to find naturally unusual ones
    test_normals = bench.test.normal
    test_anomalous = bench.test.anomalous
    all_test = test_normals + test_anomalous
    logger.info(f'Test: {len(test_normals)} normals, {len(test_anomalous)} anomalies')

    all_defs = get_all_definitions(include_paraphrases=True)
    full_ids = sorted(get_concept_ids_for_split('seen')) + \
               sorted(get_concept_ids_for_split('zs_comp')) + \
               sorted(get_concept_ids_for_split('zs_family'))
    def_texts = [all_defs[cid][0] for cid in full_ids]

    train_all = bench.train.normal + bench.train.anomalous
    user_hist = {}
    for t in train_all:
        if t.label == 0:
            user_hist.setdefault(t.user_id, []).append(t)

    # Load model
    ckpt_path = '/home/hello/ouyangqi/results/v6_porto/numosim/seed_42/best_model.pt'
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = LangTrajConfig(**(ckpt.get('config', {})))
    model = LangTrajOSR(config).to(DEVICE)
    model.definition_encoder(["init"])
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # Concept bank
    with torch.no_grad():
        c_bank, _ = model.definition_encoder(def_texts)
        c_bank = F.normalize(c_bank.float(), dim=-1).to(DEVICE)

    # Fit user prototypes on training normals
    train_ds = MobDefBenchDataset(train_all, concept_definitions=all_defs, user_histories=user_hist)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, collate_fn=collate_mobdef)
    user_prototypes = fit_user_routines(model, train_loader, DEVICE)

    # Run inference on NORMAL test trajectories only
    # to find naturally suspicious ones
    normal_ds = MobDefBenchDataset(test_normals, concept_definitions=all_defs, user_histories=user_hist)
    normal_loader = DataLoader(normal_ds, batch_size=256, shuffle=False, collate_fn=collate_mobdef)

    all_scores = []
    all_max_scores = []
    all_user_ids = []
    all_trip_ids = []
    all_ep_features = []

    model.eval()
    idx = 0
    with torch.no_grad():
        for b in normal_loader:
            pad_mask = b['mask'].to(DEVICE)
            ep = ep_dict(b['episode_tensor'], DEVICE)
            protos = _batch_user_prototypes(b['user_id'], user_prototypes, DEVICE)

            # Forward pass
            ep_emb = model.episode_encoder(ep)
            z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask)
            z_norm = F.normalize(z_x.float(), dim=-1)
            scores = (z_norm @ c_bank.T) / TEMPERATURE  # (B, 22)

            max_sc = scores.max(dim=1).values.cpu().numpy()
            all_scores.append(scores.cpu().numpy())
            all_max_scores.append(max_sc)
            all_user_ids.extend(b['user_id'])

            # Extract episode features for analysis
            ep_tensor = b['episode_tensor'].numpy()
            mask_np = b['mask'].numpy()
            batch_size = ep_tensor.shape[0]
            for bi in range(batch_size):
                valid_len = mask_np[bi].sum()
                ep_feats = ep_tensor[bi, :int(valid_len), :]
                # Summarize: time bins, zones, transitions, dwell
                time_bins = ep_feats[:, 2]  # time_bin
                dwell_bins = ep_feats[:, 3]  # dwell_bin
                transitions = ep_feats[:, 4]  # transition_type
                trip_changes = ep_feats[:, 5]  # trip_length_change
                summary = {
                    'n_episodes': int(valid_len),
                    'time_bins': time_bins.tolist(),
                    'dwell_bins': dwell_bins.tolist(),
                    'transitions': transitions.tolist(),
                    'avg_trip_change': float(trip_changes.mean()) if len(trip_changes) > 0 else 0,
                    'max_trip_change': float(trip_changes.max()) if len(trip_changes) > 0 else 0,
                }
                all_ep_features.append(summary)
                all_trip_ids.append(f'normal_{idx}')
                idx += 1

    all_max_scores = np.concatenate(all_max_scores)
    all_scores_np = np.concatenate(all_scores)

    # Find top 1% most anomalous normal trajectories
    n_top = max(50, int(0.01 * len(all_max_scores)))
    top_indices = np.argsort(-all_max_scores)[:n_top]

    logger.info(f'Top {n_top} out of {len(all_max_scores)} normal trajectories')
    logger.info(f'Score range: min={all_max_scores.min():.4f}, max={all_max_scores.max():.4f}')
    logger.info(f'Top-50 score range: {all_max_scores[top_indices[-1]]:.4f} to {all_max_scores[top_indices[0]]:.4f}')

    # Build case study entries
    cases = []
    for rank, idx in enumerate(top_indices):
        scores = all_scores_np[idx]
        top3_concept_idx = np.argsort(-scores)[:3]
        top3 = []
        for ci in top3_concept_idx:
            cid = full_ids[ci]
            top3.append({
                'concept_id': int(cid),
                'concept_name': CONCEPT_NAMES.get(cid, f'C{cid}'),
                'score': float(scores[ci]),
                'definition': def_texts[ci][:100],
            })

        case = {
            'rank': rank + 1,
            'max_score': float(all_max_scores[idx]),
            'user_id': str(all_user_ids[idx]),
            'trip_id': all_trip_ids[idx],
            'top3_concepts': top3,
            'episode_summary': all_ep_features[idx],
        }
        cases.append(case)

    # Save results
    result = {
        'n_normal_test': len(all_max_scores),
        'n_top': n_top,
        'score_stats': {
            'mean': float(all_max_scores.mean()),
            'std': float(all_max_scores.std()),
            'p99': float(np.percentile(all_max_scores, 99)),
            'max': float(all_max_scores.max()),
        },
        'cases': cases[:50],  # Top 50
    }

    with open(OUT / 'case_study.json', 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f'Saved to {OUT / "case_study.json"}')

    # Print top 5 examples
    print('\n' + '='*70)
    print('  TOP 5 NATURALLY SUSPICIOUS PORTO TRAJECTORIES')
    print('='*70)
    for case in cases[:5]:
        print(f"\n  Rank {case['rank']}: user={case['user_id']}, score={case['max_score']:.3f}")
        print(f"  Episodes: {case['episode_summary']['n_episodes']}, avg_trip_change={case['episode_summary']['avg_trip_change']:.2f}")
        for tc in case['top3_concepts']:
            print(f"    -> {tc['concept_name']} (id={tc['concept_id']}, score={tc['score']:.3f})")
            print(f"       \"{tc['definition']}\"")

    # Analyze concept distribution among top-50
    concept_counts = {}
    for case in cases[:50]:
        cid = case['top3_concepts'][0]['concept_id']
        cname = case['top3_concepts'][0]['concept_name']
        key = f'{cid}:{cname}'
        concept_counts[key] = concept_counts.get(key, 0) + 1

    print('\n' + '='*70)
    print('  CONCEPT DISTRIBUTION AMONG TOP-50')
    print('='*70)
    for k, v in sorted(concept_counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v} ({v/50*100:.0f}%)')


if __name__ == '__main__':
    main()
