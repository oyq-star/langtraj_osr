"""Download all datasets and pre-cache models for LangTraj-OSR experiments.

Datasets:
  - GeoLife (Microsoft): auto-download
  - Porto Taxi (Kaggle): requires kaggle credentials or manual download
  - Foursquare NYC/Tokyo: auto-download from public mirror
  - NUMOSIM: synthetic, generated in code — no download needed

Models:
  - sentence-transformers/all-MiniLM-L6-v2: pre-cached via HuggingFace

Usage:
    python -m langtraj_osr.scripts.download_assets --data_dir data/
    python -m langtraj_osr.scripts.download_assets --data_dir data/ --skip_porto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASETS = {
    "geolife": {
        "description": "Microsoft GeoLife GPS Trajectories 1.3 (182 users, Beijing, 4yr)",
        "url": "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip",
        "filename": "geolife.zip",
        "extract_to": "geolife",
        "md5": None,  # verify manually if needed
        "method": "http",
        "size_hint": "~298 MB",
    },
    "porto": {
        "description": "Porto Taxi Trip Dataset (1.7M trips, Kaggle)",
        "url": None,  # requires Kaggle API
        "filename": "train.csv.zip",
        "extract_to": "porto",
        "method": "kaggle",
        "kaggle_dataset": "c/pkdd-15-predict-taxi-service-trajectory-i",
        "size_hint": "~564 MB",
        "manual_note": (
            "1. Go to https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data\n"
            "2. Download train.csv.zip\n"
            "3. Place it at: {data_dir}/porto/train.csv.zip\n"
            "4. Re-run this script to extract."
        ),
    },
    "foursquare_nyc": {
        "description": "Foursquare NYC Check-in Dataset (Yang et al., 2015)",
        "url": "https://archive.org/download/foursquare_nyc_raw/dataset_TSMC2014_NYC.txt",
        "filename": "foursquare_nyc.txt",
        "extract_to": "foursquare_nyc",
        "method": "http",
        "size_hint": "~28 MB",
        "alt_url": (
            "https://sites.google.com/site/yangdingqi/home/foursquare-dataset"
        ),
    },
    "foursquare_tokyo": {
        "description": "Foursquare Tokyo Check-in Dataset (Yang et al., 2015)",
        "url": "https://archive.org/download/foursquare_nyc_raw/dataset_TSMC2014_TKY.txt",
        "filename": "foursquare_tokyo.txt",
        "extract_to": "foursquare_tokyo",
        "method": "http",
        "size_hint": "~30 MB",
    },
}

MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "description": "Sentence Transformers MiniLM-L6-v2 (text encoder)",
        "cache_dir": None,  # uses HuggingFace default cache
    },
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _progress_hook(downloaded: int, block_size: int, total: int) -> None:
    if total > 0:
        pct = min(100, downloaded * block_size * 100 // total)
        done = pct // 5
        bar = "#" * done + "-" * (20 - done)
        print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def download_http(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file via HTTP with progress."""
    logger.info("Downloading %s → %s", desc or url, dest)
    logger.info("  URL: %s", url)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, dest, reporthook=_progress_hook)
        print()  # newline after progress bar
        logger.info("  Downloaded: %.1f MB", dest.stat().st_size / 1e6)
        return True
    except Exception as e:
        logger.error("  Download failed: %s", e)
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip archive."""
    logger.info("Extracting %s → %s", zip_path.name, extract_to)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        logger.info("  Extracted successfully")
        return True
    except Exception as e:
        logger.error("  Extraction failed: %s", e)
        return False


def try_kaggle_download(dataset: str, dest_dir: Path) -> bool:
    """Try to download a Kaggle dataset using the kaggle CLI."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("  kaggle CLI not found. Install with: pip install kaggle")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Running: kaggle competitions download -c %s -p %s", dataset, dest_dir)
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", dataset.split("/")[-1], "-p", str(dest_dir)],
        capture_output=False,
        text=True,
    )
    return result.returncode == 0


def write_download_instructions(data_dir: Path, name: str, info: dict) -> None:
    """Write a MANUAL_DOWNLOAD.txt for datasets that cannot be auto-downloaded."""
    instr_path = data_dir / info["extract_to"] / "MANUAL_DOWNLOAD.txt"
    instr_path.parent.mkdir(parents=True, exist_ok=True)
    note = info.get("manual_note", "").format(data_dir=data_dir)
    instr_path.write_text(
        f"# Manual Download Required: {name}\n\n"
        f"Description: {info['description']}\n"
        f"Expected size: {info.get('size_hint', 'unknown')}\n\n"
        f"Instructions:\n{note}\n"
    )
    logger.info("  Manual download instructions written to: %s", instr_path)


def write_status(data_dir: Path, status: dict) -> None:
    """Write download status JSON."""
    status_path = data_dir / "download_status.json"
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    logger.info("Download status saved to %s", status_path)


# ---------------------------------------------------------------------------
# Per-dataset handlers
# ---------------------------------------------------------------------------

def download_geolife(data_dir: Path) -> str:
    info = DATASETS["geolife"]
    dest_dir = data_dir / info["extract_to"]
    marker = dest_dir / ".done"

    if marker.exists():
        logger.info("[geolife] Already downloaded, skipping.")
        return "DONE"

    zip_path = data_dir / info["filename"]
    if not zip_path.exists():
        ok = download_http(info["url"], zip_path, "GeoLife dataset")
        if not ok:
            return "DOWNLOAD_FAILED"

    dest_dir.mkdir(parents=True, exist_ok=True)
    ok = extract_zip(zip_path, dest_dir)
    if not ok:
        return "EXTRACT_FAILED"

    marker.touch()
    logger.info("[geolife] Ready at %s", dest_dir)
    return "DONE"


def download_porto(data_dir: Path, skip: bool = False) -> str:
    info = DATASETS["porto"]
    dest_dir = data_dir / info["extract_to"]
    marker = dest_dir / ".done"

    if marker.exists():
        logger.info("[porto] Already downloaded, skipping.")
        return "DONE"

    if skip:
        logger.info("[porto] Skipped by user flag.")
        return "SKIPPED"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try Kaggle CLI first
    zip_path = dest_dir / info["filename"]
    if not zip_path.exists():
        logger.info("[porto] Attempting Kaggle CLI download...")
        ok = try_kaggle_download(info["kaggle_dataset"], dest_dir)
        if not ok:
            logger.warning("[porto] Kaggle auto-download failed.")
            write_download_instructions(data_dir, "porto", info)
            return "MANUAL_REQUIRED"

    # Extract if zip exists
    if zip_path.exists():
        ok = extract_zip(zip_path, dest_dir)
        if ok:
            marker.touch()
            return "DONE"

    write_download_instructions(data_dir, "porto", info)
    return "MANUAL_REQUIRED"


def download_foursquare(data_dir: Path, city: str) -> str:
    """Download Foursquare NYC or Tokyo dataset."""
    name = f"foursquare_{city}"
    info = DATASETS[name]
    dest_dir = data_dir / info["extract_to"]
    dest_file = dest_dir / info["filename"]
    marker = dest_dir / ".done"

    if marker.exists():
        logger.info("[%s] Already downloaded, skipping.", name)
        return "DONE"

    dest_dir.mkdir(parents=True, exist_ok=True)
    ok = download_http(info["url"], dest_file, f"Foursquare {city.upper()}")

    if not ok:
        # Try alternative: write instructions
        alt = info.get("alt_url", "")
        instr = dest_dir / "MANUAL_DOWNLOAD.txt"
        instr.write_text(
            f"Auto-download failed for Foursquare {city.upper()}.\n"
            f"Please download manually from:\n  {alt}\n"
            f"Place the .txt file at: {dest_file}\n"
        )
        logger.warning("[%s] Download failed. See %s", name, instr)
        return "DOWNLOAD_FAILED"

    marker.touch()
    logger.info("[%s] Ready at %s", name, dest_file)
    return "DONE"


def precache_text_model(model_name: str, cache_dir: Optional[str] = None) -> str:
    """Pre-download the sentence-transformer model to avoid first-run delays."""
    logger.info("[model] Pre-caching: %s", model_name)
    try:
        from sentence_transformers import SentenceTransformer
        kwargs = {"cache_folder": cache_dir} if cache_dir else {}
        model = SentenceTransformer(model_name, **kwargs)
        # Warm up with a test sentence
        model.encode(["test sentence"], show_progress_bar=False)
        logger.info("[model] %s is ready.", model_name)
        return "DONE"
    except ImportError:
        logger.error("[model] sentence-transformers not installed. Run: pip install sentence-transformers")
        return "INSTALL_REQUIRED"
    except Exception as e:
        logger.error("[model] Failed to cache model: %s", e)
        return "FAILED"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LangTraj-OSR assets")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root directory for datasets")
    parser.add_argument("--model_cache", type=str, default=None,
                        help="Directory to cache HuggingFace models (default: HF default)")
    parser.add_argument("--skip_porto", action="store_true",
                        help="Skip Porto Taxi (requires Kaggle account)")
    parser.add_argument("--skip_foursquare", action="store_true",
                        help="Skip Foursquare NYC/Tokyo downloads")
    parser.add_argument("--only_model", action="store_true",
                        help="Only pre-cache the text encoder model")
    parser.add_argument("--only_datasets", action="store_true",
                        help="Skip model pre-caching")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LangTraj-OSR Asset Downloader")
    logger.info("Data directory: %s", data_dir)
    logger.info("=" * 60)

    status = {}

    # ---- Datasets ----
    if not args.only_model:
        logger.info("\n--- Datasets ---")
        logger.info("[numosim] Synthetic — generated at runtime, no download needed.")
        status["numosim"] = "SYNTHETIC"

        status["geolife"] = download_geolife(data_dir)
        status["porto"] = download_porto(data_dir, skip=args.skip_porto)

        if not args.skip_foursquare:
            status["foursquare_nyc"] = download_foursquare(data_dir, "nyc")
            status["foursquare_tokyo"] = download_foursquare(data_dir, "tokyo")
        else:
            status["foursquare_nyc"] = "SKIPPED"
            status["foursquare_tokyo"] = "SKIPPED"

    # ---- Models ----
    if not args.only_datasets:
        logger.info("\n--- Models ---")
        for model_name in MODELS:
            status[f"model:{model_name}"] = precache_text_model(
                model_name, cache_dir=args.model_cache
            )

    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("Download Summary:")
    for key, val in status.items():
        icon = "✓" if val in ("DONE", "SYNTHETIC") else ("?" if "MANUAL" in val or val == "SKIPPED" else "✗")
        logger.info("  %s  %-35s  %s", icon, key, val)

    needs_manual = [k for k, v in status.items() if "MANUAL" in v]
    if needs_manual:
        logger.warning("\n⚠  Manual download required for: %s", ", ".join(needs_manual))
        logger.warning("   Check MANUAL_DOWNLOAD.txt files inside each dataset directory.")

    write_status(data_dir, status)

    failed = [k for k, v in status.items() if "FAILED" in v]
    if failed:
        logger.error("\n✗ Some downloads failed: %s", failed)
        sys.exit(1)
    else:
        logger.info("\n✓ All available assets ready. Proceed with experiments.")


if __name__ == "__main__":
    main()
