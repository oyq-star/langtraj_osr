"""
Exp 2: Definition robustness under noise.
Tests AUROC and Top-1 at 5 noise levels applied to definition strings at inference time.
Uses validate-style bank scoring (cosine similarity) matching train.py AUROC computation.
Run: python robustness_exp.py
"""
import sys, json, logging, random, tempfile
sys.path.insert(0, '/home/hello/ouyangqi')

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SEED        = 42
DEVICE      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CKPT        = '/home/hello/ouyangqi/results/v6_synthetic/numosim/seed_42/best_model.pt'
OUTPUT      = Path('/home/hello/ouyangqi/results/robustness_results.json')
TEMPERATURE = 0.07

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── noise functions ──────────────────────────────────────────────────────────

def noise_word_drop(text, rate, rng):
    words = text.split()
    if len(words) <= 2: return text
    kept = [w for w in words if rng.random() > rate]
    return ' '.join(kept) if kept else words[0]

def noise_shuffle(text, rng):
    words = text.split()
    rng.shuffle(words)
    return ' '.join(words)

def noise_truncate(text, keep_frac, rng):
    words = text.split()
    n = max(1, int(len(words) * keep_frac))
    return ' '.join(words[:n])

NOISE_LEVELS = {
    'clean':            lambda txt, rng: txt,
    'drop_20pct':       lambda txt, rng: noise_word_drop(txt, 0.20, rng),
    'drop_40pct':       lambda txt, rng: noise_word_drop(txt, 0.40, rng),
    'truncate_50pct':   lambda txt, rng: noise_truncate(txt, 0.50, rng),
    'shuffle':          lambda txt, rng: noise_shuffle(txt, rng),
}

# ── helpers ───────────────────────────────────────────────────────────────────

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


def compute_metrics(scores, labels, seen_ids, zsc_ids, all_22_ids):
    from sklearn.metrics import roc_auc_score
    binary = (labels > 0).astype(int)
    try:
        auroc = float(roc_auc_score(binary, scores.max(axis=1)))
    except Exception:
        auroc = 0.0

    def top1(subset_ids):
        mask = np.isin(labels, subset_ids)
        if not mask.any(): return 0.0
        s, l = scores[mask], labels[mask]
        preds = np.array([all_22_ids[p] if p < len(all_22_ids) else -1
                          for p in s.argmax(axis=1)])
        return float((preds == l).mean())

    return {'auroc': round(auroc,4),
            'top1_seen': round(top1(seen_ids),4),
            'top1_zsc': round(top1(zsc_ids),4)}


def main():
    from langtraj_osr.benchmark.benchmark_builder import MobDefBenchBuilder
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
    from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig
    from torch.utils.data import DataLoader

    logger.info('Building benchmark seed=%d', SEED)
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
    defs_clean = [get_def_str(cid) for cid in all_22_ids]

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

    all_results = {}
    rng = random.Random(SEED)
    for noise_name, noise_fn in NOISE_LEVELS.items():
        logger.info('--- Noise: %s ---', noise_name)
        defs_noisy = [noise_fn(d, rng) for d in defs_clean]
        logger.info('Sample def[0]: %s', defs_noisy[0][:80])

        # Re-encode noisy definitions each time
        with torch.no_grad():
            c_embs, _ = model.definition_encoder(defs_noisy)
            c_bank = F.normalize(c_embs.float(), dim=-1).to(DEVICE)

        scores, labels = collect_bank_scores(model, test_loader, c_bank)
        m = compute_metrics(scores, labels, seen_ids, zsc_ids, all_22_ids)
        all_results[noise_name] = m
        logger.info('%s: AUROC=%.4f | Top-1 seen=%.3f | Top-1 zsc=%.3f',
                    noise_name, m['auroc'], m['top1_seen'], m['top1_zsc'])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info('Saved to %s', OUTPUT)

    print('\n=== ROBUSTNESS RESULTS ===')
    print(f'{"Noise Level":<20} {"AUROC":>8} {"Top-1 Seen":>12} {"Top-1 ZS-Comp":>14}')
    for name, m in all_results.items():
        print(f'{name:<20} {m["auroc"]:>8.4f} {m["top1_seen"]:>12.3f} {m["top1_zsc"]:>14.3f}')

if __name__ == '__main__':
    main()
