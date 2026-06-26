"""
Run typed metrics (Top-1, Macro-F1, OSCR) on ablation checkpoints.
Run: /home/hello/miniconda3/envs/oyq_v01/bin/python ablation_typed.py
"""
import sys, json
sys.path.insert(0, '/home/hello/ouyangqi')

import numpy as np
import tempfile
import torch
from pathlib import Path
from sklearn.metrics import f1_score, roc_curve

ABLATIONS = ['full', 'no_repel', 'no_bank', 'no_prim', 'no_para', 'no_orth']
BASE_DIR   = Path('/home/hello/ouyangqi/results')
CKPT_PATHS = {
    'full':     BASE_DIR / 'v6_synthetic/numosim/seed_42',
    'no_repel': BASE_DIR / 'ablation/no_repel/numosim/seed_42',
    'no_bank':  BASE_DIR / 'ablation/no_bank/numosim/seed_42',
    'no_prim':  BASE_DIR / 'ablation/no_prim/numosim/seed_42',
    'no_para':  BASE_DIR / 'ablation/no_para/numosim/seed_42',
    'no_orth':  BASE_DIR / 'ablation/no_orth/numosim/seed_42',
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42


def build_synthetic_data(seed=42):
    """Build synthetic MobDefBench (same as train.py --use_synthetic)."""
    from langtraj_osr.benchmark.benchmark_builder import MobDefBenchBuilder
    from langtraj_osr.core.dataset import MobDefBenchDataModule
    from langtraj_osr.core.concepts import get_all_definitions

    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to_:
        builder = MobDefBenchBuilder(data_dir=td, output_dir=to_, seed=seed)
        benchmarks = builder.build(datasets=['numosim'])
        bench = benchmarks['numosim']

    def _comb(s): return s.normal + s.anomalous
    train_trajs = _comb(bench.train)
    val_trajs   = _comb(bench.val)
    test_trajs  = _comb(bench.test)

    concept_defs = get_all_definitions(include_paraphrases=True)
    user_histories = {}
    for t in train_trajs:
        if t.label == 0:
            user_histories.setdefault(t.user_id, []).append(t)

    data_module = MobDefBenchDataModule(
        trajectories={'train': train_trajs, 'val': val_trajs, 'test': test_trajs},
        concept_definitions=concept_defs,
        user_histories=user_histories,
        batch_size=128,
    )
    return data_module, bench


def compute_typed_metrics(model, data_module, calib, device):
    from langtraj_osr.train import _tensor_to_episode_dict, _batch_user_prototypes
    from langtraj_osr.core.concepts import get_concept_ids_for_split
    from langtraj_osr.core.concepts import get_all_definitions

    q_norm = calib.get('q_norm', 0.5)
    concept_thresholds = calib.get('concept_thresholds', {})
    q_bar = float(np.mean(list(concept_thresholds.values()))) if concept_thresholds else 0.5

    # Concept definitions for scoring (all 22 with definitions)
    all_defs_dict = get_all_definitions(include_paraphrases=False)
    # Build ordered list for concepts 1-22
    if isinstance(all_defs_dict, dict):
        all_defs = [all_defs_dict.get(k, [''])[0] for k in range(1, 23)]
    else:
        all_defs = list(all_defs_dict)[:22]

    # Fit user prototypes from training data
    user_embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch in data_module.train_dataloader():
            ep   = batch['episode_tensor'].to(device)
            mask = batch['mask'].to(device)
            labs = batch['label']
            uids = batch['user_id']
            nm   = labs == 0
            if not nm.any(): continue
            ep_d = _tensor_to_episode_dict(ep[nm])
            emb  = model.episode_encoder(ep_d)
            z, _ = model.trajectory_encoder(emb, ~mask[nm])
            for i, uid in enumerate([u for u,m in zip(uids, nm.tolist()) if m]):
                user_embeddings.setdefault(uid, []).append(z[i].cpu())

    user_protos = {}
    for uid, embs in user_embeddings.items():
        user_protos[uid] = model.user_history.fit_user(torch.stack(embs).to(device))

    # Score test set
    all_enorm, all_cs, all_labels = [], [], []
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            ep   = batch['episode_tensor'].to(device)
            mask = batch['mask'].to(device)
            labs = batch['label']
            uids = batch['user_id']

            ep_d    = _tensor_to_episode_dict(ep)
            protos  = _batch_user_prototypes(uids, user_protos, device)
            out     = model(ep_d, mask, protos, all_defs)

            all_enorm.extend(out['E_norm'].cpu().tolist())
            all_cs.append(out['concept_scores'].cpu().numpy())
            all_labels.extend(labs.tolist())

    enorm  = np.array(all_enorm)
    cs_mat = np.concatenate(all_cs, axis=0)  # (N, 22)
    labels = np.array(all_labels)

    seen_ids = set(get_concept_ids_for_split('seen'))     # 1-12
    unk_ids  = set(get_concept_ids_for_split('unknown'))  # 23-25

    # Top-1 on A_seen anomalies
    seen_mask  = np.isin(labels, list(seen_ids))
    pred_seen  = cs_mat[seen_mask].argmax(axis=1) + 1
    true_seen  = labels[seen_mask]
    top1       = float((pred_seen == true_seen).mean()) if seen_mask.sum() > 0 else 0.0

    # Macro-F1 (3-way: Normal=0, KnownAnomaly=1, Unknown=2)
    pred_3 = []
    true_3 = []
    for i in range(len(labels)):
        if enorm[i] <= q_norm:
            pred_3.append(0)
        else:
            if cs_mat[i].max() > q_bar:
                pred_3.append(1)
            else:
                pred_3.append(2)
        if labels[i] == 0:
            true_3.append(0)
        elif labels[i] in unk_ids:
            true_3.append(2)
        else:
            true_3.append(1)
    mf1 = f1_score(true_3, pred_3, average='macro', zero_division=0)

    # OSCR: A_seen only (normal + seen anomalies)
    so_mask  = (labels == 0) | seen_mask
    s_enorm  = enorm[so_mask]
    s_labels = labels[so_mask]
    s_pred   = cs_mat[so_mask].argmax(axis=1) + 1
    binary   = (s_labels > 0).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        oscr_val = 0.0
    else:
        fprs, tprs, thresholds = roc_curve(binary, s_enorm)
        oscr_vals = []
        for thr in thresholds:
            det  = (s_enorm >= thr) & (s_labels > 0)
            corr = (s_pred == s_labels) & det
            oscr_vals.append(corr.sum() / max((s_labels > 0).sum(), 1))
        oscr_val = float(np.trapz(oscr_vals, fprs))

    return {
        'top1_seen': round(top1, 3),
        'mf1':       round(float(mf1), 3),
        'oscr':      round(abs(oscr_val), 3),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Building synthetic dataset (seed 42)...")
    data_module, bench = build_synthetic_data(SEED)
    print(f"  test={len(bench.test.normal)+len(bench.test.anomalous)} "
          f"(norm={len(bench.test.normal)}, anom={len(bench.test.anomalous)})")

    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig

    print(f"\n{'Variant':15s} {'Top-1 (seen)':>13s} {'Macro-F1':>10s} {'OSCR':>8s}")
    print("-" * 52)

    results = {}
    for variant in ABLATIONS:
        ckpt_dir = CKPT_PATHS[variant]
        if not ckpt_dir.exists():
            print(f"{variant:15s}  MISSING")
            continue
        try:
            ckpt  = torch.load(ckpt_dir / 'best_model.pt', map_location=DEVICE)
            model = LangTrajOSR(LangTrajConfig()).to(DEVICE)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            model.eval()

            with open(ckpt_dir / 'calibrator.json') as f:
                calib = json.load(f)

            m = compute_typed_metrics(model, data_module, calib, DEVICE)
            results[variant] = m
            print(f"{variant:15s} {m['top1_seen']:>13.3f} {m['mf1']:>10.3f} {m['oscr']:>8.3f}")
        except Exception as e:
            import traceback
            print(f"{variant:15s}  ERROR: {e}")
            traceback.print_exc()

    out_path = BASE_DIR / 'ablation_typed_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
