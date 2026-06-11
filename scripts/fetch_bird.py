"""Download + unpack the BIRD dev set into data/bird/.

    python scripts/fetch_bird.py

Tries the official OSS mirror; if the URL has moved, prints manual steps
(https://bird-bench.github.io → Dev set). Expected result:
    data/bird/dev.json
    data/bird/dev_databases/<db_id>/<db_id>.sqlite
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
import zipfile

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
DEST = "data/bird"


def _flatten(root: str) -> None:
    """The zip nests everything under dev_20240627/ (and databases in a second
    zip). Normalise to data/bird/dev.json + data/bird/dev_databases/."""
    # snapshot the walks before mutating the tree underneath them
    for dirpath, _, files in list(os.walk(root)):
        for f in files:
            if f == "dev.json" and dirpath != root:
                shutil.move(os.path.join(dirpath, f), os.path.join(root, f))
            if f == "dev_databases.zip":
                with zipfile.ZipFile(os.path.join(dirpath, f)) as z:
                    z.extractall(root)
    for dirpath, dirs, _ in list(os.walk(root)):
        for d in dirs:
            if d == "dev_databases" and dirpath != root:
                target = os.path.join(root, d)
                if not os.path.exists(target):
                    shutil.move(os.path.join(dirpath, d), target)


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    zip_path = os.path.join(DEST, "dev.zip")
    if not os.path.exists(zip_path):
        print(f"downloading {URL} (~1.2GB)…")
        try:
            urllib.request.urlretrieve(URL, zip_path)
        except Exception as e:
            sys.exit(
                f"download failed ({e}).\nManual fallback: get the dev set from "
                "https://bird-bench.github.io and unzip so that data/bird/dev.json "
                "and data/bird/dev_databases/ exist."
            )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DEST)
    _flatten(DEST)
    ok = os.path.exists(os.path.join(DEST, "dev.json")) and os.path.isdir(
        os.path.join(DEST, "dev_databases")
    )
    print("ok: data/bird ready" if ok else
          "unzipped, but layout unexpected — arrange manually (see docstring)")


if __name__ == "__main__":
    main()
