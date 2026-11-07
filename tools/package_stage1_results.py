"""Package only approved Stage 1 inputs and outputs for GitHub download."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/stage1"
WORK = ROOT / ".cache/stage1_upload"
PART_BYTES = 48 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_paths(include_features):
    paths = []
    for folder in ["stage1", "tests", "results/stage1", "figures/stage1", "reports/stage1", "notebooks", "tools"]:
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and ".ipynb_checkpoints" not in p.parts)
    paths.extend(ROOT / name for name in ["STAGE1_TASK.md", "requirements-stage1.txt", "docs/CLOUD_SETUP.md"])
    data = ROOT / "data/DSAI4205_unified_cleaning_seed42_v3_stemming"
    paths.extend(data / name for name in ["train.csv", "fold_assignments.csv", "code/unified_data.py",
                                         "dataset_config.json", "README.md", "SHA256SUMS.json"])
    if include_features:
        paths.extend(p for p in (ROOT / "features/stage1").rglob("*") if p.is_file())
    return sorted(set(paths))


def build_zip(path, files):
    manifest = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for p in files:
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.writestr("BUNDLE_CONTENTS.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files) + 1
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "file_count": len(files)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    small = build_zip(OUTPUT / "stage1_results.zip", collect_paths(False))
    full_path = WORK / "stage1_complete.zip"
    full = build_zip(full_path, collect_paths(True))
    parts = []
    with full_path.open("rb") as stream:
        number = 1
        while chunk := stream.read(PART_BYTES):
            path = OUTPUT / f"stage1_complete.zip.part{number:02d}"
            path.write_bytes(chunk)
            parts.append({"filename": path.name, "bytes": len(chunk), "sha256": sha256(path)})
            number += 1
    metadata = {"results_zip": small, "complete_zip": full, "parts": parts,
                "part_bytes": PART_BYTES, "includes_features": True,
                "excluded": ["test and audit data", "original dataset ZIP", "model downloads", "virtual environment", "credentials"]}
    (OUTPUT / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
