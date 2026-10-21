"""Download and extract the official UCI HAR Dataset archive."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


# Dataset page: https://archive.ics.uci.edu/dataset/240/
UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_DIR = RAW_DIR / "UCI HAR Dataset"
ARCHIVE_PATH = RAW_DIR / "uci_har.zip"
NESTED_ARCHIVE_PATH = RAW_DIR / "UCI HAR Dataset.zip"

REQUIRED_PATHS = (
    "train/Inertial Signals/body_acc_x_train.txt",
    "train/subject_train.txt",
    "train/y_train.txt",
    "test/Inertial Signals/body_acc_x_test.txt",
    "test/subject_test.txt",
    "test/y_test.txt",
)


def dataset_is_complete(dataset_dir: Path = DATASET_DIR) -> bool:
    return all((dataset_dir / relative_path).is_file() for relative_path in REQUIRED_PATHS)


def _download(url: str, destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mamba-course-project/1.0"})

    with urllib.request.urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\rDownloading UCI HAR: {100 * downloaded / total:5.1f}%", end="")

    print()
    temporary_path.replace(destination)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != destination and destination not in member_path.parents:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
        archive.extractall(destination)


def download_uci_har() -> Path:
    """Return the local dataset directory, downloading it when necessary."""
    if dataset_is_complete():
        print(f"UCI HAR is ready: {DATASET_DIR}")
        return DATASET_DIR

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.is_file():
        _download(UCI_HAR_URL, ARCHIVE_PATH)

    print("Extracting UCI HAR...")
    _safe_extract(ARCHIVE_PATH, RAW_DIR)

    # The current UCI endpoint wraps the original dataset ZIP inside another ZIP.
    if not dataset_is_complete() and NESTED_ARCHIVE_PATH.is_file():
        _safe_extract(NESTED_ARCHIVE_PATH, RAW_DIR)

    if not dataset_is_complete():
        raise RuntimeError(
            "The archive was extracted, but the expected UCI HAR files were not found. "
            f"Check {RAW_DIR}."
        )

    ARCHIVE_PATH.unlink(missing_ok=True)
    NESTED_ARCHIVE_PATH.unlink(missing_ok=True)
    shutil.rmtree(RAW_DIR / "__MACOSX", ignore_errors=True)
    print(f"UCI HAR is ready: {DATASET_DIR}")
    return DATASET_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    download_uci_har()


if __name__ == "__main__":
    main()

