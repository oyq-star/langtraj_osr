"""Download all datasets and models into a self-contained project directory
for transfer to an offline (no-internet) server.

After running this script, copy the entire `idea1_discovery/` folder to
the target server.  Everything will be under:

    data/          — datasets (GeoLife, Porto, Foursquare NYC/Tokyo)
    models_cache/  — sentence-transformers/all-MiniLM-L6-v2

NUMOSIM is synthetic and generated at runtime — no download needed.

Usage:
    # Download everything (needs kaggle CLI for Porto)
    python download_for_offline.py

    # Skip Porto (no Kaggle account)
    python download_for_offline.py --skip_porto

    # Only download the model
    python download_for_offline.py --only_model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models_cache"

# ── Dataset metadata ──────────────────────────────────────────────────────

DATASETS = {
    "geolife": {
        "description": "Microsoft GeoLife GPS Trajectories 1.3 (182 users, Beijing)",
        "url": "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip",
        "filename": "geolife.zip",
        "extract_to": "geolife",
        "method": "http",
        "size_hint": "~298 MB",
    },
    "porto": {
        "description": "Porto Taxi Trip Dataset (1.7M trips, Kaggle)",
        "url": None,
        "filename": "train.csv.zip",
        "extract_to": "porto",
        "method": "kaggle",
        "kaggle_competition": "pkdd-15-predict-taxi-service-trajectory-i",
        "size_hint": "~564 MB",
    },
    "foursquare_nyc": {
        "description": "Foursquare NYC Check-in Dataset (Yang et al.)",
        "url": "https://archive.org/download/foursquare_nyc_raw/dataset_TSMC2014_NYC.txt",
        "filename": "foursquare_nyc.txt",
        "extract_to": "foursquare_nyc",
        "method": "http",
        "size_hint": "~28 MB",
    },
    "foursquare_tokyo": {
        "description": "Foursquare Tokyo Check-in Dataset (Yang et al.)",
        "url": "https://archive.org/download/foursquare_nyc_raw/dataset_TSMC2014_TKY.txt",
        "filename": "foursquare_tokyo.txt",
        "extract_to": "foursquare_tokyo",
        "method": "http",
        "size_hint": "~30 MB",
    },
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ── Helpers ───────────────────────────────────────────────────────────────

def _progress_hook(downloaded: int, block_size: int, total: int) -> None:
    if total > 0:
        pct = min(100, downloaded * block_size * 100 // total)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def download_http(url: str, dest: Path, desc: str = "") -> bool:
    logger.info("Downloading %s  (%s)", desc, dest.name)
    logger.info("  URL: %s", url)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, str(dest), reporthook=_progress_hook)
        print()
        logger.info("  Done: %.1f MB", dest.stat().st_size / 1e6)
        return True
    except Exception as e:
        logger.error("  Download FAILED: %s", e)
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    logger.info("Extracting %s → %s", zip_path.name, extract_to)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        logger.info("  Extracted OK")
        return True
    except Exception as e:
        logger.error("  Extract FAILED: %s", e)
        return False


# ── Dataset downloaders ───────────────────────────────────────────────────

def download_geolife() -> str:
    info = DATASETS["geolife"]
    dest_dir = DATA_DIR / info["extract_to"]
    marker = dest_dir / ".done"
    if marker.exists():
        logger.info("[geolife] Already present, skipping.")
        return "DONE"

    zip_path = DATA_DIR / info["filename"]
    if not zip_path.exists():
        if not download_http(info["url"], zip_path, "GeoLife"):
            return "FAILED"

    dest_dir.mkdir(parents=True, exist_ok=True)
    if not extract_zip(zip_path, dest_dir):
        return "FAILED"

    marker.touch()
    return "DONE"


def download_porto(skip: bool = False) -> str:
    info = DATASETS["porto"]
    dest_dir = DATA_DIR / info["extract_to"]
    marker = dest_dir / ".done"
    if marker.exists():
        logger.info("[porto] Already present, skipping.")
        return "DONE"
    if skip:
        logger.info("[porto] Skipped by --skip_porto.")
        return "SKIPPED"

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / info["filename"]

    if not zip_path.exists():
        logger.info("[porto] Trying kaggle CLI ...")
        try:
            subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
            result = subprocess.run(
                ["kaggle", "competitions", "download",
                 "-c", info["kaggle_competition"],
                 "-p", str(dest_dir)],
                capture_output=False, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("kaggle CLI returned non-zero")
        except Exception as e:
            logger.warning("[porto] Kaggle download failed: %s", e)
            logger.warning(
                "[porto] Please download manually:\n"
                "  1. Go to https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data\n"
                "  2. Download train.csv.zip\n"
                "  3. Place at: %s\n"
                "  4. Re-run this script.",
                zip_path,
            )
            return "MANUAL_REQUIRED"

    if zip_path.exists():
        if extract_zip(zip_path, dest_dir):
            marker.touch()
            return "DONE"

    return "FAILED"


def download_foursquare(city: str) -> str:
    name = f"foursquare_{city}"
    info = DATASETS[name]
    dest_dir = DATA_DIR / info["extract_to"]
    dest_file = dest_dir / info["filename"]
    marker = dest_dir / ".done"
    if marker.exists():
        logger.info("[%s] Already present, skipping.", name)
        return "DONE"

    dest_dir.mkdir(parents=True, exist_ok=True)
    if not download_http(info["url"], dest_file, f"Foursquare {city.upper()}"):
        return "FAILED"

    marker.touch()
    return "DONE"


# ── Model downloader ─────────────────────────────────────────────────────

def download_model() -> str:
    """Download sentence-transformers model into models_cache/ for offline use."""
    local_path = MODEL_DIR / "all-MiniLM-L6-v2"
    marker = local_path / ".done"
    if marker.exists():
        logger.info("[model] Already present at %s, skipping.", local_path)
        return "DONE"

    logger.info("[model] Downloading %s → %s", MODEL_NAME, local_path)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("[model] sentence-transformers not installed.  pip install sentence-transformers")
        return "FAILED"

    try:
        model = SentenceTransformer(MODEL_NAME)
        local_path.mkdir(parents=True, exist_ok=True)
        model.save(str(local_path))
        # Verify it loads from local path
        _ = SentenceTransformer(str(local_path))
        logger.info("[model] Saved & verified at %s", local_path)
        marker.touch()
        return "DONE"
    except Exception as e:
        logger.error("[model] Failed: %s", e)
        return "FAILED"


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all assets for offline LangTraj-OSR experiments"
    )
    parser.add_argument("--skip_porto", action="store_true",
                        help="Skip Porto Taxi (requires Kaggle account)")
    parser.add_argument("--skip_foursquare", action="store_true",
                        help="Skip Foursquare NYC/Tokyo")
    parser.add_argument("--only_model", action="store_true",
                        help="Only download the text encoder model")
    parser.add_argument("--only_datasets", action="store_true",
                        help="Only download datasets, skip model")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LangTraj-OSR Offline Asset Downloader")
    logger.info("  Data dir:  %s", DATA_DIR)
    logger.info("  Model dir: %s", MODEL_DIR)
    logger.info("=" * 60)

    status = {}

    # ── Datasets ──
    if not args.only_model:
        logger.info("")
        logger.info("--- Datasets ---")
        logger.info("[numosim] Synthetic — generated at runtime, no download needed.")
        status["numosim"] = "SYNTHETIC"
        status["geolife"] = download_geolife()
        status["porto"] = download_porto(skip=args.skip_porto)
        if not args.skip_foursquare:
            status["foursquare_nyc"] = download_foursquare("nyc")
            status["foursquare_tokyo"] = download_foursquare("tokyo")
        else:
            status["foursquare_nyc"] = "SKIPPED"
            status["foursquare_tokyo"] = "SKIPPED"

    # ── Model ──
    if not args.only_datasets:
        logger.info("")
        logger.info("--- Model ---")
        status["model:all-MiniLM-L6-v2"] = download_model()

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary:")
    for key, val in status.items():
        icon = {
            "DONE": "OK", "SYNTHETIC": "OK", "SKIPPED": "--",
        }.get(val, "!!")
        logger.info("  [%s]  %-35s  %s", icon, key, val)

    # Write status file
    status_path = ROOT / "download_status.json"
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    logger.info("")
    logger.info("Status saved to %s", status_path)

    failed = [k for k, v in status.items() if v == "FAILED"]
    manual = [k for k, v in status.items() if v == "MANUAL_REQUIRED"]

    if failed:
        logger.error("FAILED: %s", ", ".join(failed))
    if manual:
        logger.warning("MANUAL download needed: %s", ", ".join(manual))

    # ── Offline instructions ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("To use on offline server:")
    logger.info("  1. Copy the entire idea1_discovery/ folder to server")
    logger.info("  2. Set text_encoder in config to local path:")
    logger.info("       models_cache/all-MiniLM-L6-v2")
    logger.info("  3. Or use --text_encoder flag:")
    logger.info("       python -m langtraj_osr.train --text_encoder models_cache/all-MiniLM-L6-v2 ...")
    logger.info("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
