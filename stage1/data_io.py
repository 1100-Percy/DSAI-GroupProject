"""Allowlisted input access, immutable Phrase fingerprints, and grouped folds."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

# Dynamic loading of the original normalizer must never write into DATA_DIR.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold


def project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    here = Path.cwd().resolve()
    return here if (here / "STAGE1_TASK.md").exists() else here.parent


ROOT = project_root()
CONFIG = yaml.safe_load((ROOT / "stage1/config.yaml").read_text(encoding="utf-8"))
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DATA = ROOT / CONFIG["data_dir"]
FEATURES = ROOT / "features/stage1"
RESULTS = ROOT / "results/stage1"
FIGURES = ROOT / "figures/stage1"
REPORTS = ROOT / "reports/stage1"
LABELS = [0, 1, 2, 3]
CLASS_NAMES = ["Negative", "Positive", "Irrelevant", "Neutral"]


def prepare_dirs():
    for path in (FEATURES, RESULTS, FIGURES, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def phrase_sha(phrases):
    # Length framing preserves every byte and makes row boundaries unambiguous.
    digest = hashlib.sha256()
    for phrase in phrases:
        value = phrase.encode("utf-8")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def load_data(stage="read", require_check=True):
    prepare_dirs()
    frame = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig", keep_default_na=False)
    folds = pd.read_csv(DATA / "fold_assignments.csv", encoding="utf-8-sig", keep_default_na=False)
    assert list(frame.columns) == ["Phrase", "Sentiment"]
    assert len(frame) == len(folds) == 563
    assert np.array_equal(folds.row_index.to_numpy(), np.arange(563))
    assert folds.sample_id.is_unique and folds.group_id.ne("").all()
    assert np.array_equal(frame.Sentiment.to_numpy(), folds.Sentiment.to_numpy())
    assert set(frame.Sentiment) == set(LABELS)
    assert set(folds.validation_fold) == set(range(1, 6))
    assert all(isinstance(p, str) and p and "" not in p.split(" ") for p in frame.Phrase)
    assert int((folds.groupby("group_id").size() > 1).sum()) == 13
    assert folds.groupby("group_id").validation_fold.nunique().max() == 1
    value = phrase_sha(frame.Phrase)
    check_path = RESULTS / "check.json"
    if require_check:
        if not check_path.exists():
            raise RuntimeError("Run check before using the data")
        check = read_json(check_path)
        assert value == check["phrase_sha256"], "Phrase changed since check"
        for name in ("train.csv", "fold_assignments.csv", "code/unified_data.py"):
            assert file_sha(DATA / name) == check["verified_hashes"][name]
    write_json(RESULTS / f"phrase_{stage}.json", {"stage": stage, "phrase_sha256": value, "n_rows": len(frame)})
    return frame, folds


def make_repeats(frame, assignments):
    records = []
    for repeat in CONFIG["repeats"]:
        allocated = np.zeros(len(frame), dtype=int)
        if repeat == 0:
            allocated[:] = assignments.validation_fold
        else:
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=repeat)
            for fold, (_, valid) in enumerate(splitter.split(frame.Phrase, frame.Sentiment, assignments.group_id), 1):
                allocated[valid] = fold
        assert set(allocated) == set(range(1, 6))
        assert pd.DataFrame({"group": assignments.group_id, "fold": allocated}).groupby("group").fold.nunique().max() == 1
        records.extend({"row_index": i, "repeat": repeat, "fold": int(f)} for i, f in enumerate(allocated))
    return pd.DataFrame(records)


def check_inputs():
    frame, assignments = load_data("check", require_check=False)
    manifest = read_json(DATA / "SHA256SUMS.json")
    allowed = ["train.csv", "fold_assignments.csv", "code/unified_data.py", "dataset_config.json", "README.md"]
    verified = {}
    for name in allowed:
        actual = file_sha(DATA / name)
        assert actual == manifest[name], f"Hash mismatch: {name}"
        verified[name] = actual
    repeats = make_repeats(frame, assignments)
    repeats.to_csv(RESULTS / "folds_repeats.csv", index=False, encoding="utf-8")
    check = {"phrase_sha256": phrase_sha(frame.Phrase), "n_rows": len(frame),
             "verified_hashes": verified, "unverified_manifest_entries": sorted(set(manifest) - set(allowed)),
             "scope": "Allowlisted extracted files only; forbidden and archived files never opened",
             "groups": int(assignments.group_id.nunique()), "multirow_groups": 13,
             "folds_sha256": file_sha(RESULTS / "folds_repeats.csv")}
    write_json(RESULTS / "check.json", check)
    print(f"CHECK passed in allowed scope: {len(frame)} rows, 25 grouped folds, {len(verified)} hashes", flush=True)
    return check


def load_repeats(frame, assignments):
    path = RESULTS / "folds_repeats.csv"
    assert file_sha(path) == read_json(RESULTS / "check.json")["folds_sha256"]
    repeats = pd.read_csv(path, encoding="utf-8")
    pd.testing.assert_frame_equal(repeats, make_repeats(frame, assignments))
    return repeats


def record_environment(command):
    prepare_dirs()
    import importlib.metadata
    packages = ["numpy", "scipy", "scikit-learn", "gensim", "torch", "transformers", "nltk"]
    versions = {p: importlib.metadata.version(p) for p in packages}
    info = {"command": command, "machine": platform.node(), "platform": platform.platform(),
            "python": sys.version, "versions": versions,
            "environment": {key: os.environ.get(key) for key in ["PYTHONHASHSEED", "PYTHONUTF8", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM"]}}
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8")
    (RESULTS / f"env_{platform.node()}.txt").write_text(json.dumps(info, indent=2) + "\n\n" + freeze, encoding="utf-8")
    write_json(RESULTS / f"environment_{command}.json", info)


def cache_identity():
    return {"phrase_sha256": read_json(RESULTS / "check.json")["phrase_sha256"],
            "config_sha256": file_sha(ROOT / "stage1/config.yaml"),
            "embeddings_sha256": file_sha(ROOT / "stage1/embeddings.py"),
            "normalizer_sha256": file_sha(DATA / "code/unified_data.py")}
