"""Download every photo into data/img_cache so the live session never waits on wifi.

Run once, before the talk:  uv run python prewarm_images.py
Safe to re-run: it skips anything already cached.
"""
import pandas as pd, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DATA = Path("data")
CACHE = DATA / "img_cache"; CACHE.mkdir(exist_ok=True)
photos = pd.read_parquet(DATA / "london5k_index.parquet")

todo = [r for _, r in photos.iterrows() if not (CACHE / r["photos"]).exists()]
print(f"{len(photos) - len(todo):,} already cached, {len(todo):,} to fetch")

ok = fail = 0


def get(row):
    global ok, fail
    try:
        r = requests.get(row["url"], timeout=20,
                         headers={"User-Agent": "beautiful-places-exercise"})
        r.raise_for_status()
        (CACHE / row["photos"]).write_bytes(r.content)
        ok += 1
    except Exception:
        fail += 1
    if (ok + fail) % 250 == 0:
        print(f"  {ok + fail:,}/{len(todo):,}  ok={ok:,} failed={fail:,}", flush=True)


with ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(get, todo))

print(f"done: {ok:,} downloaded, {fail:,} failed, "
      f"{len(list(CACHE.iterdir())):,} images cached")
