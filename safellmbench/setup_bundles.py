"""Download and install the SafeLLMBench pretrained bundles from Google Drive."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from . import config


def _run_gdown(file_id: str, out_path: Path) -> None:
    url = f"https://drive.google.com/uc?id={file_id}"
    cmd = [sys.executable, "-m", "gdown", url, "-O", str(out_path)]
    print(f"[setup] downloading {out_path.name} ...")
    subprocess.check_call(cmd)


def _extract(zip_path: Path, dest: Path) -> None:
    print(f"[setup] extracting {zip_path.name} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def install(force: bool = False) -> None:
    """Download `trained_models.zip` and `classifier_bundle.zip` into `~/.safellmbench/`.

    Layout after install:

        ~/.safellmbench/
        ├── jailbreak_generator/   (LoRA adapter + Qwen tokenizer)
        ├── transformer_classifier.pt
        └── bpe_tokenizer.json
    """
    config.ensure_home()
    if not force and config.bundles_installed():
        print(f"[setup] bundles already present at {config.HOME_DIR}")
        return

    tmp = config.HOME_DIR / "_downloads"
    tmp.mkdir(exist_ok=True)

    trained_zip = tmp / "trained_models.zip"
    clf_zip = tmp / "classifier_bundle.zip"

    if force or not trained_zip.exists():
        _run_gdown(config.TRAINED_MODELS_GDRIVE_ID, trained_zip)
    if force or not clf_zip.exists():
        _run_gdown(config.CLASSIFIER_GDRIVE_ID, clf_zip)

    # trained_models.zip contains a top-level `jailbreak_generator/` folder.
    _extract(trained_zip, config.HOME_DIR)
    # classifier_bundle.zip contains loose files at root.
    _extract(clf_zip, config.HOME_DIR)

    # Cleanup: drop the heavy `checkpoint-*` subfolder — the adapter at the top level
    # is sufficient for inference, and this saves ~250MB of disk.
    for sub in config.GENERATOR_DIR.glob("checkpoint-*"):
        if sub.is_dir():
            shutil.rmtree(sub, ignore_errors=True)

    # Cleanup: remove downloaded zips (kept around only during the run).
    shutil.rmtree(tmp, ignore_errors=True)

    if not config.bundles_installed():
        raise RuntimeError(
            f"Bundle install failed — expected files not found under {config.HOME_DIR}"
        )
    print(f"[setup] done. bundles installed at {config.HOME_DIR}")
