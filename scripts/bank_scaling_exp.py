"""
Exp 3: Definition bank size scaling.
Evaluates AUROC as the number of available concept definitions varies at inference.
Uses validate-style bank scoring (cosine similarity) matching train.py AUROC computation.
Run: python bank_scaling_exp.py
"""
import sys, json, logging, tempfile
sys.path.insert(0, '/home/hello/ouyangqi')

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SEED        = 42
DEVICE      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CKPT        = '/home/hello/ouyangqi/results/v6_synthetic/numosim/seed_42/best_model.pt'
OUTPUT      = Path('/home/hello/ouyangqi/results/bank_scaling_results.json')
TEMPERATURE = 0.07

np.random.seed(SEED); torch.manual_seed(SEED)


def ep_dict(t, device):
    return {k: t[:,:,i].long().to(device) if i < 5 or i >= 6 else t[:,:,i].float().to(device)
            for i, k in enumerate(['zone_id','poi_role','time_bin','dwell_bin',
                                    'transition_type','trip_length_change','event_flag','companion_flag'])}

@torch.no_grad()
def collect_bank_scores(model, loader, c_bank):
    """Collect bank-based cosine similarity scores matching train.py validate."""
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        pad_mask = batch['mask'].to(DEVICE)          # True=valid
        ep = ep_dict(batch['episode_tensor'], DEVICE)
        ep_emb = model.episode_encoder(ep)
        z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask)  # True=padded (correct)
        z_norm = F.normalize(z_x.float(), dim=-1)
        scores = (z_norm @ c_bank.T) / TEMPERATURE   # (B, K)
        all_scores.append(scores.cpu().float().numpy())
        all_labels.append(batch['label'].numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def compute_auroc(scores, labels):
    binary = (labels > 0).astype(int)
    try:
        return float(roc_auc_score(binary, scores.max(axis=1)))
    except Exception:
        return 0.0

def compute_top1(scores, labels, subset_ids):
    anom_mask = np.isin(labels, subset_ids) & (labels > 0)
    if anom_mask.sum() == 0: return 0.0
    s = scores[anom_mask]; l = labels[anom_mask]
    preds = np.array([subset_ids[p] if p < len(subset_ids) else -1
                      for p in s.argmax(axis=1)])
    return float((preds == l).mean())


def main():
    from langtraj_osr.benchmark.benchmark_builder import MobDefBenchBuilder
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
    from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig
    from torch.utils.data import DataLoader

    logger.info('Building benchmark seed=%d...', SEED)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to_:
        builder = MobDefBenchBuilder(data_dir=td, output_dir=to_, seed=SEED)
        benchmarks = builder.build(datasets=['numosim'])
    bench = benchmarks['numosim']
    test_trajs = bench.test.normal + bench.test.anomalous
    logger.info('test=%d (norm=%d anom=%d)', len(test_trajs),
                len(bench.test.normal), len(bench.test.anomalous))

    concept_defs = get_all_definitions(include_paraphrases=False)
    seen_ids  = sorted(get_concept_ids_for_split('seen'))
    zsc_ids   = sorted(get_concept_ids_for_split('zs_comp'))
    zsf_ids   = sorted(get_concept_ids_for_split('zs_family'))
    all_22_ids = seen_ids + zsc_ids + zsf_ids
    def get_def_str(cid):
        v = concept_defs.get(cid, '')
        return v[0] if isinstance(v, list) else v

    logger.info('Loading model...')
    ckpt = torch.load(CKPT, map_location=DEVICE)
    cfg  = ckpt.get('config', {})
    config = LangTrajConfig(**cfg) if cfg else LangTrajConfig()
    model  = LangTrajOSR(config).to(DEVICE)
    model.definition_encoder(["init"])  # Trigger lazy init before loading checkpoint
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    test_ds = MobDefBenchDataset(test_trajs, concept_defs, {}, split='test')
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_mobdef)

    # Scaling configs: K concepts at inference
    scaling_configs = [
        ('K=5  (5 seen)',        seen_ids[:5]),
        ('K=10 (10 seen)',       seen_ids[:10]),
        ('K=12 (12 seen)',       seen_ids),
        ('K=18 (12s+6zsc)',      seen_ids + zsc_ids),
        ('K=22 (all, baseline)', all_22_ids),
    ]

    all_results = {}
    for label, subset_ids in scaling_configs:
        def_texts = [get_def_str(cid) for cid in subset_ids]
        with torch.no_grad():
            c_embs, _ = model.definition_encoder(def_texts)
            c_bank = F.normalize(c_embs.float(), dim=-1).to(DEVICE)

        logger.info('--- %s (%d definitions) ---', label, len(def_texts))
        scores, labels = collect_bank_scores(model, test_loader, c_bank)
        auroc = compute_auroc(scores, labels)
        t1 = compute_top1(scores, labels, subset_ids)
        all_results[label] = {'k': len(subset_ids), 'auroc': round(auroc,4), 'top1': round(t1,4)}
        logger.info('%s: AUROC=%.4f | Top-1=%.3f', label, auroc, t1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info('Saved → %s', OUTPUT)

    print('\n=== BANK SCALING RESULTS ===')
    print(f'{"Config":<25} {"K":>4} {"AUROC":>8} {"Top-1":>8}')
    for name, m in all_results.items():
        print(f'{name:<25} {m["k"]:>4} {m["auroc"]:>8.4f} {m["top1"]:>8.3f}')

if __name__ == '__main__':
    main()
