#!/usr/bin/env python3
"""Download the Acevedo PBC dataset from Mendeley Data (external validation set).

Dataset:
    Acevedo A, Merino A, Alferez S, Molina A, Boldu L, Rodellar J (2020).
    A dataset of microscopic peripheral blood cell images for development of
    automatic recognition systems. Data in Brief 30:105474.
    Mendeley Data: https://data.mendeley.com/datasets/snkd93bnjr/1
    DOI: 10.17632/snkd93bnjr.1

Used by 07_external_validation.py to test cross-institution / cross-instrument
generalisation. Re-running is idempotent: the zip is skipped if already present
and its sha256 matches, and extraction is skipped if the images are already
unpacked.

The Mendeley public API returns the file manifest (filename, size, sha256,
download_url) from:
    GET /public-api/datasets/{slug}/files?folder_id=root&version={v}

Stdlib only — no conda env needed (mirrors 00_download_data.py).

Usage:
    python scripts/00b_download_acevedo.py                 # -> <repo>/data/acevedo/
    python scripts/00b_download_acevedo.py --data-dir /rds/user/cyy36/hpc-work/w5/data/acevedo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

MENDELEY_SLUG = "snkd93bnjr"
DEFAULT_VERSION = 1
# Folder that the zip unpacks to (07's --acevedo-dir points here).
DATASET_FOLDER = "PBC_dataset_normal_DIB"
_UA = "Mozilla/5.0 (compatible; HDS-W5-downloader/1.0)"


def _urlopen(url: str):
    """urlopen with a browser-ish User-Agent (Mendeley rejects the default)."""
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": _UA}))


def fetch_manifest(version: int) -> list[dict]:
    """Return [{name, size, sha256, url}, ...] for the dataset's root files.

    The Mendeley response is a JSON array of file objects; the download URL,
    sha256, and size live under each object's `content_details`."""
    api = (f"https://data.mendeley.com/public-api/datasets/{MENDELEY_SLUG}"
           f"/files?folder_id=root&version={version}")
    with _urlopen(api) as response:
        raw = response.read()
    payload = json.loads(raw)
    files = []
    for f in payload if isinstance(payload, list) else []:
        cd = f.get("content_details") or {}
        url, name = cd.get("download_url"), f.get("filename")
        if not url or not name:
            continue  # skip folders / incomplete entries
        files.append({
            "name": name,
            "size": int(f.get("size") or cd.get("size") or 0),
            "sha256": cd.get("sha256_hash", ""),
            "url": url,
        })
    if not files:
        # Fail loudly with the raw response so we never silently guess the shape.
        raise RuntimeError(
            f"no downloadable files parsed from manifest; raw response begins: {raw[:500]!r}"
        )
    return files


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TB"


def download(url: str, dest: Path, expected_size: int) -> None:
    """Stream-download `url` to `dest` via a `.part` file for atomicity.

    The Mendeley download_url redirects to file storage; urllib follows it."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with _urlopen(url) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            pct = (written / expected_size * 100) if expected_size else 0.0
            print(f"\r  {human_bytes(written)} / {human_bytes(expected_size)} ({pct:5.1f}%)",
                  end="", flush=True)
    print()
    tmp.rename(dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Acevedo PBC dataset from Mendeley Data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "acevedo",
        help="Target directory (default: <repo>/data/acevedo/). 07 expects "
             f"{{this}}/{DATASET_FOLDER}/.",
    )
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION,
                        help="Mendeley dataset version (default: 1).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    image_root = data_dir / DATASET_FOLDER
    if image_root.exists() and any(image_root.iterdir()):
        print(f"Acevedo already extracted at {image_root}. Nothing to do.")
        return 0

    print(f"Mendeley dataset: {MENDELEY_SLUG} (version {args.version})")
    print(f"Target directory: {data_dir}\n")

    print("Fetching file manifest from Mendeley public API...")
    try:
        manifest = fetch_manifest(args.version)
    except Exception as exc:
        print(f"  Failed to reach Mendeley API: {exc}", file=sys.stderr)
        print("  Manual download: https://data.mendeley.com/datasets/snkd93bnjr/1", file=sys.stderr)
        return 1
    if not manifest:
        print("  ERROR: manifest empty (no downloadable files).", file=sys.stderr)
        return 1

    total = sum(f["size"] for f in manifest)
    print(f"  {len(manifest)} file(s), {human_bytes(total)} total\n")

    for f in manifest:
        dest = data_dir / f["name"]
        print(f"[{f['name']}]  {human_bytes(f['size'])}")

        if dest.exists() and f["sha256"]:
            if sha256_of(dest) == f["sha256"]:
                print("  Already present, sha256 matches. Skipping download.")
            else:
                print("  Present but sha256 mismatch — redownloading.")
                dest.unlink()
        if not dest.exists():
            download(f["url"], dest, f["size"])
            if f["sha256"]:
                actual = sha256_of(dest)
                if actual != f["sha256"]:
                    print(f"  sha256 mismatch! expected {f['sha256']}, got {actual}", file=sys.stderr)
                    dest.unlink()
                    return 1
                print(f"  sha256 OK ({actual[:16]}...)")

    # ── Extract the dataset zip ─────────────────────────────────────────
    zip_path = data_dir / f"{DATASET_FOLDER}.zip"
    if zip_path.exists() and not (image_root.exists() and any(image_root.iterdir())):
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
        if not image_root.exists():
            print(f"  WARNING: expected {image_root} after extraction — check the zip's "
                  f"internal layout (07 --acevedo-dir must point at the class-folder root).",
                  file=sys.stderr)
        else:
            print(f"Extracted to {image_root}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
