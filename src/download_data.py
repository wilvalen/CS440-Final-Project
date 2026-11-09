"""Download skillscope_data.db from the official Hugging Face dataset.

Usage: python src/download_data.py
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://huggingface.co/datasets/NLBSE/SkillCompetition/resolve/main/skillscope_data.zip"
SHA256 = "c75b035e503553e8dae015653cd12aefe920e63618e1db9c123291970134fd18"


def main():
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    zip_path = data_dir / "skillscope_data.zip"

    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Checksum mismatch: got {digest}, expected {SHA256}")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extract("skillscope_data.db", data_dir)
    zip_path.unlink()
    print(f"Saved {data_dir / 'skillscope_data.db'}")


if __name__ == "__main__":
    main()
