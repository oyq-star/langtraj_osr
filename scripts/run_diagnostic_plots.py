#!/usr/bin/env python3
"""Generate diagnostic plots for LangTraj-OSR paper.

1. Seen-concept confusion matrix (12-class)
2. Stage A calibration curve (E_norm vs empirical anomaly rate)
3. Error-rejection curve (typed accuracy vs coverage as threshold varies)
4. Score distribution histogram for zero-shot concepts

Saves PNG figures to results/diagnostic_plots/
"""
import sys, json, logging, tempfile, os
sys.path.insert(0, '/home/hello/ouyangqi')
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir("/home/hello/ouyangqi")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TEMPERATURE = 0.07
OUT = Path('/home/hello/ouyangqi/results/diagnostic_plots')
OUT.mkdir(exist_ok=True, parents=True)


def ep_dict(t, device):
    return {k: t[:,:,i].long().to(device) if i < 5 or i >= 6 else t[:,:,i].float().to(device)
            for i, k in enumerate(['zone_id','poi_role','time_bin','dwell_bin',
                                    'transition_type','trip_length_change','event_flag','companion_flag'])}


def load_model_and_data(ckpt_path, seed, use_porto):
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
    from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig
    from langtraj_osr.benchmark.benchmark_builder import (
        MobDefBenchBuilder, Benchmark, CONCEPT_DEFS,
        SPLIT_SEEN, SPLIT_ZS_COMP, SPLIT_ZS_FAMILY, SPLIT_UNKNOWN,
    )
    from torch.utils.data import DataLoader

    np.random.seed(seed); torch.manual_seed(seed)

    if use_porto:
        from langtraj_osr.core.tokenizer import TrajectoryTokenizer
        from langtraj_osr.core.episode import SemanticEpisode, SemanticTrajectory
        import math, hashlib, pandas as pd, pickle

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
            builder = MobDefBenchBuilder(data_dir=td, output_dir=to2, seed=seed)
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
    else:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to2:
            builder = MobDefBenchBuilder(data_dir=td, output_dir=to2, seed=seed)
            bench = builder.build(datasets=['numosim'])['numosim']

    test_trajs = bench.test.normal + bench.test.anomalous
    all_defs = get_all_definitions(include_paraphrases=True)
    full_ids = sorted(get_concept_ids_for_split('seen')) + \
               sorted(get_concept_ids_for_split('zs_comp')) + \
               sorted(get_concept_ids_for_split('zs_family'))
    def_texts = [all_defs[cid][0] for cid in full_ids]

    test_ds = MobDefBenchDataset(test_trajs, concept_definitions=all_defs, user_histories={})
    bs = 128 if use_porto else 256
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_mobdef)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get('config', {})
    config = LangTrajConfig(**cfg) if cfg else LangTrajConfig()
    model = LangTrajOSR(config).to(DEVICE)
    model.definition_encoder(["init"])
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    with torch.no_grad():
        c_bank, _ = model.definition_encoder(def_texts)
        c_bank = F.normalize(c_bank.float(), dim=-1).to(DEVICE)

    return model, test_loader, c_bank, full_ids, def_texts


@torch.no_grad()
def collect_all(model, loader, c_bank):
    """Collect scores, labels, and max-score for detection."""
    model.eval()
    all_scores, all_labels = [], []
    for b in loader:
        pad_mask = b['mask'].to(DEVICE)
        ep = ep_dict(b['episode_tensor'], DEVICE)
        ep_emb = model.episode_encoder(ep)
        z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask)
        z_norm = F.normalize(z_x.float(), dim=-1)
        scores = (z_norm @ c_bank.T) / TEMPERATURE
        all_scores.append(scores.cpu().float().numpy())
        all_labels.append(b['label'].numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def plot_confusion_matrix(scores, labels, concept_ids, dataset_name):
    """12-class confusion matrix for seen concepts."""
    from langtraj_osr.core.concepts import get_concept_ids_for_split, CONCEPT_NAMES
    seen_ids = sorted(get_concept_ids_for_split('seen'))
    mask = np.isin(labels, seen_ids)
    if not mask.any(): return

    s_labels = labels[mask]
    s_scores = scores[mask]
    pred_idx = s_scores.argmax(axis=1)
    pred_ids = np.array([concept_ids[p] for p in pred_idx])

    cm = confusion_matrix(s_labels, pred_ids, labels=seen_ids)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # Short names for concepts
    short_names = []
    for cid in seen_ids:
        name = CONCEPT_NAMES.get(cid, f'C{cid}')
        short = name[:15].replace('_', ' ')
        short_names.append(f'{cid}:{short}')

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(seen_ids)))
    ax.set_yticks(range(len(seen_ids)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel('Predicted Concept')
    ax.set_ylabel('True Concept')
    ax.set_title(f'Seen Concept Confusion Matrix ({dataset_name})')
    plt.colorbar(im, ax=ax, label='Proportion')

    # Add text annotations
    for i in range(len(seen_ids)):
        for j in range(len(seen_ids)):
            val = cm_norm[i, j]
            if val > 0.01:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5, color=color)

    plt.tight_layout()
    plt.savefig(OUT / f'confusion_matrix_{dataset_name}.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(OUT / f'confusion_matrix_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved confusion matrix for {dataset_name}')


def plot_calibration_curve(scores, labels, dataset_name):
    """Stage A calibration: max concept score vs empirical anomaly rate."""
    max_scores = scores.max(axis=1)
    binary = (labels > 0).astype(int)

    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(binary, max_scores / max_scores.max(),
                                                  n_bins=10, strategy='uniform')
    except:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: calibration curve
    ax1.plot(prob_pred, prob_true, 'bo-', label='Model')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax1.set_xlabel('Mean Predicted Score (normalized)')
    ax1.set_ylabel('Fraction of Anomalies')
    ax1.set_title(f'Stage A Calibration ({dataset_name})')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: score distributions
    norm_scores = max_scores[labels == 0]
    anom_scores = max_scores[labels > 0]
    bins = np.linspace(min(max_scores.min(), 0), max_scores.max(), 50)
    ax2.hist(norm_scores, bins=bins, alpha=0.6, label=f'Normal (n={len(norm_scores)})', density=True)
    ax2.hist(anom_scores, bins=bins, alpha=0.6, label=f'Anomaly (n={len(anom_scores)})', density=True)
    ax2.set_xlabel('Max Concept Bank Score')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Score Distribution ({dataset_name})')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / f'calibration_{dataset_name}.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(OUT / f'calibration_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved calibration plot for {dataset_name}')


def plot_error_rejection(scores, labels, concept_ids, dataset_name):
    """Error-rejection curve: as we increase confidence threshold, accuracy increases."""
    from langtraj_osr.core.concepts import get_concept_ids_for_split
    seen_ids = sorted(get_concept_ids_for_split('seen'))
    mask = np.isin(labels, seen_ids)
    if not mask.any(): return

    s_labels = labels[mask]
    s_scores = scores[mask]
    pred_idx = s_scores.argmax(axis=1)
    pred_ids = np.array([concept_ids[p] for p in pred_idx])
    max_conf = s_scores.max(axis=1)

    # Sort by confidence (descending)
    order = np.argsort(-max_conf)
    correct = (pred_ids[order] == s_labels[order])

    # Compute cumulative accuracy at each coverage level
    coverages = np.arange(1, len(correct) + 1) / len(correct)
    cum_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages, cum_acc, 'b-', linewidth=1.5)
    ax.set_xlabel('Coverage (fraction of samples scored)')
    ax.set_ylabel('Cumulative Accuracy')
    ax.set_title(f'Error-Rejection Curve — Seen Concepts ({dataset_name})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Add reference lines
    overall_acc = correct.mean()
    ax.axhline(y=overall_acc, color='r', linestyle='--', alpha=0.5,
               label=f'Overall acc={overall_acc:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT / f'error_rejection_{dataset_name}.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(OUT / f'error_rejection_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved error-rejection curve for {dataset_name}')


def plot_zs_score_distribution(scores, labels, concept_ids, dataset_name):
    """Score distribution for zero-shot concepts: correct vs incorrect top-1."""
    from langtraj_osr.core.concepts import get_concept_ids_for_split
    zsc_ids = sorted(get_concept_ids_for_split('zs_comp'))
    zsf_ids = sorted(get_concept_ids_for_split('zs_family'))
    zs_ids = zsc_ids + zsf_ids

    mask = np.isin(labels, zs_ids)
    if not mask.any(): return

    s_labels = labels[mask]
    s_scores = scores[mask]
    pred_idx = s_scores.argmax(axis=1)
    pred_ids = np.array([concept_ids[p] for p in pred_idx])
    max_conf = s_scores.max(axis=1)

    correct = pred_ids == s_labels
    correct_scores = max_conf[correct]
    wrong_scores = max_conf[~correct]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(max_conf.min(), 0), max_conf.max(), 40)
    if len(correct_scores) > 0:
        ax.hist(correct_scores, bins=bins, alpha=0.6, label=f'Correct (n={len(correct_scores)})', color='green', density=True)
    if len(wrong_scores) > 0:
        ax.hist(wrong_scores, bins=bins, alpha=0.6, label=f'Incorrect (n={len(wrong_scores)})', color='red', density=True)
    ax.set_xlabel('Max Concept Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Zero-Shot Score Distribution ({dataset_name})')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / f'zs_scores_{dataset_name}.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(OUT / f'zs_scores_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved ZS score distribution for {dataset_name}')


# =================================================================
from langtraj_osr.core.concepts import get_concept_ids_for_split, CONCEPT_NAMES
from langtraj_osr.models.langtraj_osr import LangTrajConfig

BASE = '/home/hello/ouyangqi/results'
configs = [
    (f'{BASE}/v6_synthetic/numosim/seed_42/best_model.pt', 42, False, 'synthetic'),
    (f'{BASE}/v6_porto/numosim/seed_42/best_model.pt',     42, True,  'porto'),
]

for ckpt_path, seed, use_porto, dname in configs:
    print(f"\n{'='*60}\n  Generating plots for {dname}\n{'='*60}", flush=True)
    if not Path(ckpt_path).exists():
        print('  SKIP'); continue
    try:
        model, loader, c_bank, full_ids, def_texts = load_model_and_data(ckpt_path, seed, use_porto)
        scores, labels = collect_all(model, loader, c_bank)
        logger.info(f'{dname}: scores shape={scores.shape}, labels shape={labels.shape}')

        plot_confusion_matrix(scores, labels, full_ids, dname)
        plot_calibration_curve(scores, labels, dname)
        plot_error_rejection(scores, labels, full_ids, dname)
        plot_zs_score_distribution(scores, labels, full_ids, dname)
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback; traceback.print_exc()

print(f'\nAll plots saved to {OUT}')
print(f'Files: {list(OUT.glob("*.png"))}')
