"""
fetch_dataset.py — Autonomous Weather Data Fetcher
====================================================
Fetches missing Open-Meteo historical data in 25s windows / 60s waits.

Rules:
  - Fetches in 25-second bursts, then waits 60 seconds (rate limit safe)
  - Saves staging CSV after every burst — no data lost on crash
  - Progress tracker (fetch_progress.json) enables seamless resume
  - No fake/forward-filled data — failed chunks are skipped & logged
  - Auto-merges into master dataset after full fetch completes
  - Self-restarts on unexpected crashes (Ctrl+C exits cleanly)

Usage:
  python fetch_dataset.py
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR      = BASE_DIR / "outputs" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE  = RAW_DIR / "weather_master_dataset.csv"
STAGING_FILE = RAW_DIR / "weather_staging_dataset.csv"
PROGRESS_FILE = RAW_DIR / "fetch_progress.json"

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_END_DATE      = pd.Timestamp("2026-04-18")
FETCH_WINDOW_SECONDS = 25    # Active fetch window
WAIT_SECONDS         = 60    # Pause between windows

REGIONS = {
    "London":       {"lat": 51.5085, "lon": -0.1257},
    "Manchester":   {"lat": 53.4809, "lon": -2.2374},
    "Birmingham":   {"lat": 52.4814, "lon": -1.8998},
    "Leeds":        {"lat": 53.7964, "lon": -1.5478},
    "Glasgow":      {"lat": 55.8651, "lon": -4.2576},
    "Southampton":  {"lat": 50.9039, "lon": -1.4042},
    "Liverpool":    {"lat": 53.4105, "lon": -2.9779},
    "Newcastle":    {"lat": 54.9732, "lon": -1.6139},
    "Sheffield":    {"lat": 53.3829, "lon": -1.4659},
    "Middlesbrough":{"lat": 54.5762, "lon": -1.2348},
}

HOURLY_VARS = (
    "temperature_2m,apparent_temperature,precipitation,rain,snowfall,"
    "wind_speed_10m,wind_direction_10m,relative_humidity_2m,"
    "dew_point_2m,pressure_msl,cloud_cover"
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "fetch_dataset.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_start_date():
    """Find the day after the last datetime in the master dataset."""
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_FILE}")
    df = pd.read_csv(MASTER_FILE, usecols=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    last_dt = df["datetime"].max()
    start = last_dt.normalize() + timedelta(days=1)
    log.info(f"Master dataset ends at: {last_dt.date()}. Will fetch from: {start.date()}")
    return start


def build_chunks(start_date, end_date):
    """Split the date range into monthly city chunks."""
    chunks = []
    cur = start_date
    while cur <= end_date:
        if cur.month == 12:
            next_month = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            next_month = cur.replace(month=cur.month + 1, day=1)
        chunk_end = min(next_month - timedelta(days=1), end_date)
        for city in REGIONS:
            chunks.append({
                "city":  city,
                "start": cur.strftime("%Y-%m-%d"),
                "end":   chunk_end.strftime("%Y-%m-%d"),
                "key":   f"{city}_{cur.strftime('%Y-%m')}",
            })
        cur = next_month
    return chunks


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"done": [], "missing": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def fetch_chunk(city, start_str, end_str):
    """Fetch one city-month chunk. Returns DataFrame or None on failure."""
    coords = REGIONS[city]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  coords["lat"],
        "longitude": coords["lon"],
        "start_date": start_str,
        "end_date":   end_str,
        "hourly":     HOURLY_VARS,
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if "hourly" not in data or not data["hourly"].get("time"):
                log.warning(f"  [{city}] Empty response for {start_str}→{end_str}")
                return None
            df = pd.DataFrame(data["hourly"])
            df.rename(columns={"time": "datetime"}, inplace=True)
            df["region"] = city
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        except Exception as e:
            log.warning(f"  [{city}] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(5)
    log.error(f"  [{city}] FAILED after 3 attempts — chunk MISSING, not filled.")
    return None


def flush_buffer(buffer):
    """Append in-memory buffer to the staging CSV and clear it."""
    if not buffer:
        return
    new_df = pd.concat(buffer, ignore_index=True)
    if STAGING_FILE.exists():
        new_df.to_csv(STAGING_FILE, mode="a", header=False, index=False)
    else:
        new_df.to_csv(STAGING_FILE, index=False)
    log.info(f"  [SAVED] {len(new_df):,} rows flushed to staging CSV.")


# ──────────────────────────────────────────────────────────────────────────────
# MERGE
# ──────────────────────────────────────────────────────────────────────────────

def auto_merge():
    """Integrity-check then merge staging CSV into master dataset."""
    log.info("=" * 60)
    log.info("AUTO-MERGE: Merging staging into master dataset...")
    log.info("=" * 60)

    if not STAGING_FILE.exists():
        log.error("Staging file not found — nothing to merge.")
        return

    staging = pd.read_csv(STAGING_FILE)
    staging["datetime"] = pd.to_datetime(staging["datetime"], format="mixed")
    log.info(f"Staging dataset: {len(staging):,} rows")

    # ── Integrity check ────────────────────────────────────────────────────────
    null_pct = staging.isnull().mean() * 100
    bad_cols = null_pct[null_pct > 5]
    if not bad_cols.empty:
        log.warning("WARNING — Columns exceeding 5% nulls in staging:")
        for col, pct in bad_cols.items():
            log.warning(f"  {col}: {pct:.1f}%")
        log.warning("Proceeding with merge but flagging these columns.")
    else:
        log.info("Integrity check PASSED — all columns within acceptable null range.")

    # Drop completely null rows (garbage rows, not missing chunks)
    before = len(staging)
    staging = staging.dropna(how="all")
    if len(staging) < before:
        log.info(f"Dropped {before - len(staging)} fully-null rows.")

    # ── Merge ──────────────────────────────────────────────────────────────────
    log.info("Loading master dataset...")
    master = pd.read_csv(MASTER_FILE)
    master["datetime"] = pd.to_datetime(master["datetime"], format="mixed")
    rows_before = len(master)

    combined = pd.concat([master, staging], ignore_index=True)
    combined.drop_duplicates(subset=["datetime", "region"], keep="last", inplace=True)
    combined.sort_values(by=["region", "datetime"], inplace=True)
    combined.to_csv(MASTER_FILE, index=False)

    rows_after = len(combined)
    log.info(f"Merge complete: {rows_before:,} → {rows_after:,} rows (+{rows_after - rows_before:,})")

    # ── Rebuild features ───────────────────────────────────────────────────────
    log.info("Rebuilding feature vectors (this may take a few minutes)...")
    sys.path.insert(0, str(BASE_DIR / "src"))
    import weather_ml
    weather_ml.step_feature_engineering()
    log.info("Feature rebuild complete. Extended dataset is ready for retraining.")
    log.info("=" * 60)
    log.info("ALL DONE. Run 'python run.py' to retrain with the full extended dataset.")
    log.info("You can safely delete: fetch_dataset.py, fetch_progress.json, weather_staging_dataset.csv")
    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN FETCH LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("WEATHER ML — AUTONOMOUS DATASET FETCHER")
    log.info(f"Target: {TARGET_END_DATE.date()} | Cycle: {FETCH_WINDOW_SECONDS}s fetch / {WAIT_SECONDS}s wait")
    log.info("=" * 60)

    start_date = get_start_date()
    if start_date > TARGET_END_DATE:
        log.info("Master dataset is already up to date. Nothing to fetch.")
        return

    all_chunks = build_chunks(start_date, TARGET_END_DATE)
    progress   = load_progress()
    done_keys  = set(progress["done"])
    pending    = [c for c in all_chunks if c["key"] not in done_keys]

    total     = len(all_chunks)
    remaining = len(pending)
    n_months  = total // len(REGIONS)
    # ~3s per request estimate
    est_hours = (remaining / max(FETCH_WINDOW_SECONDS // 3, 1)) * (FETCH_WINDOW_SECONDS + WAIT_SECONDS) / 3600

    log.info(f"Chunks: {total} total ({n_months} months × {len(REGIONS)} cities)")
    log.info(f"Already done: {len(done_keys)} | Remaining: {remaining}")
    log.info(f"Estimated time: ~{est_hours:.1f} hours")
    log.info("=" * 60)

    buffer       = []
    window_start = time.time()

    for i, chunk in enumerate(pending):
        # ── 25s window elapsed? Save + wait ───────────────────────────────────
        elapsed = time.time() - window_start
        if elapsed >= FETCH_WINDOW_SECONDS:
            flush_buffer(buffer)
            buffer = []
            done_pct = len(done_keys) / total * 100
            log.info(f"Progress: {len(done_keys)}/{total} ({done_pct:.1f}%) — Waiting {WAIT_SECONDS}s...")
            time.sleep(WAIT_SECONDS)
            window_start = time.time()

        city      = chunk["city"]
        start_str = chunk["start"]
        end_str   = chunk["end"]
        log.info(f"[{len(done_keys)+1}/{total}] {city} | {start_str} → {end_str}")

        df = fetch_chunk(city, start_str, end_str)

        if df is not None:
            buffer.append(df)
            progress["done"].append(chunk["key"])
            done_keys.add(chunk["key"])
        else:
            if chunk["key"] not in progress["missing"]:
                progress["missing"].append(chunk["key"])

        save_progress(progress)

    # Final flush
    flush_buffer(buffer)

    log.info("=" * 60)
    log.info(f"FETCH COMPLETE — {len(progress['done'])}/{total} chunks fetched successfully.")
    if progress["missing"]:
        log.warning(f"{len(progress['missing'])} chunks could not be fetched (no fake data inserted):")
        for key in progress["missing"]:
            log.warning(f"  MISSING: {key}")

    # Auto-merge
    auto_merge()


# ──────────────────────────────────────────────────────────────────────────────
# SELF-RESTARTING ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    while True:
        try:
            main()
            break  # Completed successfully — exit clean
        except KeyboardInterrupt:
            log.info("\n[STOPPED] Ctrl+C detected. Progress saved. Re-run script to resume.")
            sys.exit(0)
        except Exception as e:
            log.error(f"[CRASH] Unexpected error: {e}. Restarting in 30 seconds...")
            time.sleep(30)
            log.info("[RESTART] Resuming from last saved progress...")
