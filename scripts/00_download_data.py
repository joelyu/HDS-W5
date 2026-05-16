#!/usr/bin/env python3
"""Download the KU-Optofil PBC dataset from Zenodo.

Dataset:
    Yarıkan AE et al. (2026). A Large-Scale Peripheral Blood Cell Dataset
    for Automated Hematological Analysis. Scientific Data 13:417.
    Zenodo: https://doi.org/10.5281/zenodo.17333317

Re-running is idempotent: files that already exist and match their published
md5 are skipped.

Usage:
    python scripts/00_download_data.py                                  # everything → <repo>/data/raw/
    python scripts/00_download_data.py --metadata-only                  # ~4 MB, schema check
    python scripts/00_download_data.py --data-dir /rds/user/cyy36/hpc-work/w5/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD_ID = "17333317"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
METADATA_FILENAMES = {"metadata.csv", "metadata_with_patient_level_splits.csv"}


def fetch_manifest() -> list[dict]:
    """Return [{name, size, md5, url}, ...] for every file in the Zenodo record."""
    with urllib.request.urlopen(ZENODO_API) as response:
        payload = json.load(response)
    return [
        {
            "name": f["key"],
            "size": f["size"],
            "md5": f["checksum"].removeprefix("md5:"),
            "url": f["links"]["self"],
        }
        for f in payload["files"]
    ]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
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
    """Stream-download `url` to `dest`, writing through a `.part` file for atomicity."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            pct = (written / expected_size * 100) if expected_size else 0.0
            print(
                f"\r  {human_bytes(written)} / {human_bytes(expected_size)} ({pct:5.1f}%)",
                end="",
                flush=True,
            )
    print()
    tmp.rename(dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the KU-Optofil PBC dataset from Zenodo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Target directory (default: <repo>/data/raw/)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip the 565 MB dataset.zip; only fetch the two CSV files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Zenodo record:    {ZENODO_RECORD_ID}")
    print(f"Target directory: {data_dir}")
    print()

    print("Fetching file manifest from Zenodo API...")
    try:
        manifest = fetch_manifest()
    except Exception as exc:
        print(f"  Failed to reach Zenodo API: {exc}", file=sys.stderr)
        return 1

    if args.metadata_only:
        manifest = [f for f in manifest if f["name"] in METADATA_FILENAMES]

    total = sum(f["size"] for f in manifest)
    print(f"  {len(manifest)} file(s), {human_bytes(total)} total")
    print()

    for f in manifest:
        dest = data_dir / f["name"]
        print(f"[{f['name']}]  {human_bytes(f['size'])}")

        if dest.exists():
            actual = md5_of(dest)
            if actual == f["md5"]:
                print("  Already present, md5 matches. Skipping.")
                continue
            print("  Present but md5 mismatch — redownloading.")
            dest.unlink()

        download(f["url"], dest, f["size"])

        actual = md5_of(dest)
        if actual != f["md5"]:
            print(
                f"  md5 mismatch after download! "
                f"expected {f['md5']}, got {actual}",
                file=sys.stderr,
            )
            dest.unlink()
            return 1
        print(f"  md5 OK ({actual[:16]}...)")

    # ── Unzip dataset if needed ──────────────────────────────────────────
    zip_path = data_dir / "dataset.zip"
    image_root = data_dir / "dataset"
    if zip_path.exists() and not (image_root.exists() and any(image_root.iterdir())):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
        print(f"Extracted to {image_root}")
    elif image_root.exists():
        print("Images already extracted.")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
