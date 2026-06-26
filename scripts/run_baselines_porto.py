"""
Run all 5 baselines on Porto real data.
Monkey-patches run_baseline._build_synthetic_datasets to return Porto data.
Usage: /home/hello/miniconda3/envs/oyq_v01/bin/python run_baselines_porto.py [--baseline all|norm_only|...]
"""
import sys, argparse, json, math, hashlib, logging
sys.path.insert(0, '/home/hello/ouyangqi')

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

PORTO_PKL   = '/home/hello/ouyangqi/data/porto_trips.pkl'
OUTPUT_BASE = Path('/home/hello/ouyangqi/results/baselines_porto')
SEED        = 42
BASELINES   = ['norm_only', 'dsl_xl', 'nl2dsl', 'canonical_json', 'lm_tad', 'atrom_ossl']


# ── Porto data builder (mirrors train.py --use_porto_real) ───────────────────

def build_porto_benchmark(seed=42):
    import pickle, math as _math, hashlib as _hash, re as _re
    import pandas as _pd
    from langtraj_osr.core.episode import SemanticEpisode as _SE, SemanticTrajectory as _ST
    from langtraj_osr.core.tokenizer import TrajectoryTokenizer as _TT
    from langtraj_osr.benchmark.benchmark_builder import (
        MobDefBenchBuilder, Benchmark,
        CONCEPT_DEFS, SPLIT_SEEN, SPLIT_ZS_COMP, SPLIT_ZS_FAMILY, SPLIT_UNKNOWN,
    )

    logger.info("Loading Porto pickle: %s", PORTO_PKL)
    with open(PORTO_PKL, 'rb') as f:
        df = pickle.load(f)

    _tok = _TT()

    def _hav(lat1, lon1, lat2, lon2):
        R = 6_371_000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        a = (math.sin(math.radians(lat2-lat1)/2)**2
             + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2)
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

    def _zone(lat, lon, res=0.005):
        h = hashlib.md5(f"{int(round(lat/res))},{int(round(lon/res))}".encode()).hexdigest()
        return int(h[:8], 16) % (2**31)

    def _trans(spd): return 0 if spd < 2 else (2 if spd < 10 else 1)

    # Sample taxis
    import random as _rnd
    _rnd.seed(seed)
    n_taxis = 400
    max_trips = 80
    min_trips = 10
    min_points = 4

    taxi_ids = df['TAXI_ID'].unique().tolist()
    _rnd.shuffle(taxi_ids)
    selected = []
    for tid in taxi_ids:
        sub = df[df['TAXI_ID'] == tid]
        if len(sub) >= min_trips:
            selected.append(tid)
        if len(selected) >= n_taxis:
            break

    logger.info("Selected %d taxis", len(selected))
    trajs = []
    for tid in selected:
        sub = df[df['TAXI_ID'] == tid].copy()
        if len(sub) > max_trips:
            sub = sub.sample(max_trips, random_state=seed)
        sub = sub.sort_values('TIMESTAMP')
        user_speeds = []
        for _, row in sub.iterrows():
            poly = row.get('POLYLINE', [])
            if not isinstance(poly, list) or len(poly) < min_points:
                continue
            pts = [(float(p[1]), float(p[0])) for p in poly]  # (lat, lon)
            speeds = []
            for i in range(1, len(pts)):
                dist = _hav(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])
                spd  = dist / 15.0  # 15 sec interval → m/s
                speeds.append(spd)
            if speeds:
                user_speeds.extend(speeds)
        avg_spd = float(np.mean(user_speeds)) if user_speeds else 5.0

        trip_lens = []
        for _, row in sub.iterrows():
            poly = row.get('POLYLINE', [])
            if not isinstance(poly, list) or len(poly) < min_points:
                continue
            trip_lens.append(len(poly))
        avg_len = float(np.mean(trip_lens)) if trip_lens else 20.0

        for _, row in sub.iterrows():
            poly = row.get('POLYLINE', [])
            if not isinstance(poly, list) or len(poly) < min_points:
                continue
            pts = [(float(p[1]), float(p[0])) for p in poly]
            ts_start = int(row.get('TIMESTAMP', 0))
            eps = []
            for i, (lat, lon) in enumerate(pts):
                dwell_sec = 15.0 * max(1, len(pts) - i - 1) / max(len(pts), 1) * 60
                dwell_sec = min(max(dwell_sec, 1.0), 480.0 * 60)
                dwell_min = dwell_sec / 60.0
                spd = 0.0
                if i > 0:
                    dist = _hav(pts[i-1][0], pts[i-1][1], lat, lon)
                    spd  = dist / 15.0
                zone = _zone(lat, lon)
                import datetime
                dt = datetime.datetime.fromtimestamp(ts_start + i * 15)
                tb = dt.hour * 7 + dt.weekday()
                db = _tok._discretize_dwell(dwell_min)
                tlc = min(len(pts) / max(avg_len, 1.0), 20.0)
                poi = int(hashlib.md5(f"{zone}".encode()).hexdigest()[:2], 16) % 64
                eps.append(_SE(
                    zone_id=zone, poi_role=poi, time_bin=tb, dwell_bin=db,
                    transition_type=_trans(spd),
                    trip_length_change=round(tlc, 4),
                    event_flag=0, companion_flag=0,
                ))
            trajs.append(_ST(
                episodes=eps,
                user_id=str(tid),
                trip_id=f"{tid}_{row.get('TRIP_ID', _)}",
                label=0,
            ))

    logger.info("Tokenized %d trips from %d users", len(trajs), len(selected))

    import tempfile
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to_:
        builder = MobDefBenchBuilder(data_dir=td, output_dir=to_, seed=seed)
        raw = builder._tokenize(trajs)
        tr, va, te = builder._split_by_user(raw)
        bench = Benchmark(dataset_name='porto_real')
        bench.train.normal = tr
        bench.val.normal   = va
        bench.test.normal  = te
        sc = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
        zc = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
        zf = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
        uc = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]
        builder._inject_anomalies(bench.train, tr, sc + zc + zf)
        builder._inject_anomalies(bench.val,   va, sc + zc)
        builder._inject_anomalies(bench.test,  te, sc + zc + zf + uc)

    logger.info("Benchmark: train=%d (norm=%d anom=%d), val=%d, test=%d",
                len(bench.train.normal)+len(bench.train.anomalous),
                len(bench.train.normal), len(bench.train.anomalous),
                len(bench.val.normal)+len(bench.val.anomalous),
                len(bench.test.normal)+len(bench.test.anomalous))
    return bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='all',
                        help='Baseline name or "all"')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args_cli = parser.parse_args()

    baselines_to_run = BASELINES if args_cli.baseline == 'all' else [args_cli.baseline]

    # Build Porto data once
    logger.info("=== Building Porto benchmark ===")
    bench = build_porto_benchmark(args_cli.seed)

    # Wrap in SyntheticBaselineDataset
    from langtraj_osr.baselines.run_baseline import SyntheticBaselineDataset, train_and_evaluate
    import langtraj_osr.baselines.run_baseline as rb
    from langtraj_osr.core.concepts import get_all_definitions

    concept_defs = get_all_definitions(include_paraphrases=True)

    def _combine(s): return s.normal + s.anomalous

    import random as _rnd
    _rng = _rnd.Random(args_cli.seed)
    train_trajs = _combine(bench.train); _rng.shuffle(train_trajs)
    val_trajs   = _combine(bench.val);   _rng.shuffle(val_trajs)
    test_trajs  = _combine(bench.test);  _rng.shuffle(test_trajs)

    train_ds = SyntheticBaselineDataset(train_trajs, concept_defs)
    val_ds   = SyntheticBaselineDataset(val_trajs,   concept_defs)
    test_ds  = SyntheticBaselineDataset(test_trajs,  concept_defs)

    logger.info("Datasets: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    # Monkey-patch _build_synthetic_datasets to return Porto data
    def _porto_datasets(args_inner):
        return train_ds, val_ds, test_ds
    rb._build_synthetic_datasets = _porto_datasets

    # Run each baseline
    all_results = {}
    for bl in baselines_to_run:
        logger.info("\n=== Running baseline: %s ===", bl)
        out_dir = OUTPUT_BASE / bl / f'seed_{args_cli.seed}'
        out_dir.mkdir(parents=True, exist_ok=True)

        run_args = argparse.Namespace(
            baseline=bl,
            dataset='porto_real',
            seed=args_cli.seed,
            use_synthetic=True,
            data_dir=str(OUTPUT_BASE),
            batch_size=args_cli.batch_size,
            epochs=args_cli.epochs,
            lr=args_cli.lr,
            output_dir=str(out_dir),
            gpu=0,
        )

        try:
            result = train_and_evaluate(run_args)
            all_results[bl] = result
            # Save per-baseline result
            with open(out_dir / 'results.json', 'w') as f:
                json.dump(result, f, indent=2)
            logger.info("=== %s DONE: overall=%.4f ===", bl,
                        result.get('test_metrics', {}).get('auroc',
                        result.get('best_val_auroc', 0)))
        except Exception as e:
            import traceback
            logger.error("=== %s FAILED: %s ===", bl, e)
            traceback.print_exc()
            all_results[bl] = {'error': str(e)}

    # Summary
    print("\n=== FINAL RESULTS (Porto Real) ===")
    for bl, res in all_results.items():
        if 'error' in res:
            print(f"{bl}: ERROR — {res['error']}")
        else:
            tm = res.get('test_metrics', {})
            mps = res.get('metrics_per_split', {})
            overall = tm.get('auroc', res.get('best_val_auroc', 0))
            seen  = mps.get('A_seen',   {}).get('auroc', '—')
            zsc   = mps.get('A_zs_comp',{}).get('auroc', '—')
            zsf   = mps.get('A_zs_fam', {}).get('auroc', '—')
            unk   = mps.get('A_unknown',{}).get('auroc', '—')
            print(f"{bl}: overall={overall:.4f} | seen={seen} | zs_comp={zsc} | zs_fam={zsf} | unk={unk}")

    with open(OUTPUT_BASE / f'summary_seed{args_cli.seed}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved → {OUTPUT_BASE}/summary_seed{args_cli.seed}.json")


if __name__ == '__main__':
    main()
