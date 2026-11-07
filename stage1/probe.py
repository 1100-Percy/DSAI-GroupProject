"""Grouped nested selection with fold-local transforms and fixed linear probes."""
import itertools

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baseline_ref import baseline_features
from .data_io import (CONFIG, FEATURES, LABELS, REPORTS, RESULTS, file_sha, load_data,
                      load_repeats, read_json, write_json)
from .embeddings import fold_word_features, load_pretrained


def select_c(candidates):
    best = max(row["inner_f1_mean"] for row in candidates)
    return min(row["C"] for row in candidates if best - row["inner_f1_mean"] < CONFIG["tie_tolerance"])


def feature_transform(config):
    if config == "baseline_ref":
        return baseline_features()
    steps = []
    if config == "bertweet_pca100":
        steps.append(("pca", PCA(n_components=100, svd_solver="full", random_state=42)))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def linear_probe(c):
    return LogisticRegression(C=float(c), class_weight="balanced", solver="lbfgs",
                              max_iter=5000, random_state=42)


def metrics(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {"macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            "acc": accuracy_score(y_true, y_pred),
            "macro_p": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            "macro_r": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            **dict(zip(["f1_neg", "f1_pos", "f1_irr", "f1_neu"], per_class))}


def nested_fold(config, x, y, groups, train, valid):
    assert not set(groups[train]) & set(groups[valid])
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    scores = {float(c): [] for c in CONFIG["C_grid"]}
    for inner_train, inner_valid in inner.split(x[train], y[train], groups[train]):
        fit_rows, tune_rows = train[inner_train], train[inner_valid]
        assert not set(groups[fit_rows]) & set(groups[tune_rows])
        assert set(y[fit_rows]) == set(LABELS)
        transform = feature_transform(config)
        a = transform.fit_transform(x[fit_rows])
        b = transform.transform(x[tune_rows])
        for c in scores:
            model = linear_probe(c).fit(a, y[fit_rows])
            scores[c].append(metrics(y[tune_rows], model.predict(b))["macro_f1"])
    candidates = [{"C": c, "inner_f1_mean": float(np.mean(values)),
                   "inner_f1_sd": float(np.std(values, ddof=1))} for c, values in scores.items()]
    chosen = select_c(candidates)
    transform = feature_transform(config)
    a = transform.fit_transform(x[train])
    b = transform.transform(x[valid])
    model = linear_probe(chosen).fit(a, y[train])
    assert np.array_equal(model.classes_, LABELS)
    return chosen, model.predict(b), model.predict_proba(b), candidates


def pairwise_comparisons(scores):
    rows = []
    for a, b in itertools.combinations(CONFIG["configs"], 2):
        joined = scores[scores.config == a].merge(scores[scores.config == b], on=["repeat", "fold"],
                                                 suffixes=("_a", "_b"), validate="one_to_one")
        delta = joined.macro_f1_a - joined.macro_f1_b
        assert len(delta) == 25
        rows.append({"config_a": a, "config_b": b, "mean_delta": delta.mean(),
                     "sd_delta": delta.std(ddof=1), "wins_a": int((delta > 0).sum()), "n_folds": len(delta)})
    return pd.DataFrame(rows)


def comparison_detail(scores, a, b):
    av = scores[scores.config == a].set_index(["repeat", "fold"]).macro_f1
    bv = scores[scores.config == b].set_index(["repeat", "fold"]).macro_f1
    delta = av - bv
    mean, wins_a, wins_b = delta.mean(), int((delta > 0).sum()), int((delta < 0).sum())
    decision = "无法区分"
    if abs(mean) >= 0.01:
        if mean > 0 and wins_a >= 18:
            decision = "更好"
        elif mean < 0 and wins_b >= 18:
            decision = "更差"
    return {"config_a": a, "config_b": b, "mean_delta": mean, "sd_delta": delta.std(ddof=1),
            "wins_a": wins_a, "wins_b": wins_b, "decision": decision}


def run_probe(recompute=False):
    frame, assignments = load_data("probe")
    repeats = load_repeats(frame, assignments)
    pretrained = load_pretrained()
    y, groups = frame.Sentiment.to_numpy(), assignments.group_id.to_numpy()
    score_rows, tuning_rows, coverage_rows = [], [], []
    oof = {config: [] for config in CONFIG["configs"]}
    for (repeat, fold), rows in repeats.groupby(["repeat", "fold"], sort=True):
        valid = rows.row_index.to_numpy()
        train = np.setdiff1d(np.arange(len(frame)), valid)
        assert not np.intersect1d(train, valid).size
        for config in CONFIG["configs"]:
            if config in ("w2v", "ft"):
                x, coverage = fold_word_features(config, frame, train, valid, int(repeat), int(fold), recompute)
                coverage_rows.append(coverage)
            elif config == "baseline_ref":
                x = frame.Phrase.to_numpy()
            else:
                x = pretrained["bertweet" if config == "bertweet_pca100" else config]
            c, predictions, probabilities, candidates = nested_fold(config, x, y, groups, train, valid)
            score_rows.append({"config": config, "repeat": repeat, "fold": fold, "C": c,
                               **metrics(y[valid], predictions)})
            tuning_rows.extend({"config": config, "repeat": repeat, "fold": fold, **row} for row in candidates)
            if repeat == 0:
                oof[config].extend({"row_index": int(i), "sample_id": assignments.sample_id.iloc[i],
                                    "fold": int(fold), "y_true": int(y[i]), "y_pred": int(pred),
                                    **{f"p{k}": float(prob[k]) for k in LABELS}}
                                   for i, pred, prob in zip(valid, predictions, probabilities))
            print(f"R{repeat} F{fold} {config}: F1={score_rows[-1]['macro_f1']:.5f}, C={c}", flush=True)
    scores, tuning = pd.DataFrame(score_rows), pd.DataFrame(tuning_rows)
    assert len(scores) == 175 and len(tuning) == 1575
    assert tuning.groupby(["config", "repeat", "fold"]).size().eq(9).all()
    scores.to_csv(RESULTS / "fold_scores.csv", index=False, encoding="utf-8")
    tuning.to_csv(RESULTS / "tuning.csv", index=False, encoding="utf-8")
    for config, rows in oof.items():
        table = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
        assert np.array_equal(table.row_index, np.arange(563))
        table.to_csv(RESULTS / f"oof_{config}.csv", index=False, encoding="utf-8")
    pairwise_comparisons(scores).to_csv(RESULTS / "pairwise_delta.csv", index=False, encoding="utf-8")
    coverage = pd.concat([pd.DataFrame(coverage_rows), pd.read_csv(RESULTS / "coverage_pretrained.csv", encoding="utf-8")], ignore_index=True)
    coverage.to_csv(RESULTS / "coverage.csv", index=False, encoding="utf-8")
    boundary = scores.assign(boundary=scores.C.isin([0.001, 10])).groupby("config").boundary.sum()
    write_json(RESULTS / "grid_boundaries.json", {k: int(v) for k, v in boundary.items()})
    return scores


def verify_reproducibility():
    # Retrain every fold-local word model and refit every nested probe on each run.
    run_probe(recompute=True)
    first = file_sha(RESULTS / "fold_scores.csv")
    first_bytes = (RESULTS / "fold_scores.csv").read_bytes()
    run_probe(recompute=True)
    second = file_sha(RESULTS / "fold_scores.csv")
    assert first_bytes == (RESULTS / "fold_scores.csv").read_bytes(), "Same-machine rerun differs"
    write_json(RESULTS / "reproducibility.json", {"first_sha256": first, "second_sha256": second,
               "byte_identical": True, "fresh_word_models_each_run": True,
               "fresh_nested_fits_each_run": True, "pretrained_features": "cached frozen extraction"})
