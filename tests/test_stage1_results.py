"""Validate completed experiment artifacts against the prespecified protocol."""
import numpy as np
import pandas as pd
import pytest

from stage1.data_io import CONFIG, FEATURES, RESULTS, file_sha, load_data, read_json
from stage1.probe import pairwise_comparisons, select_c

pytestmark = pytest.mark.skipif(not (RESULTS / "fold_scores.csv").exists(), reason="Full experiment not yet run")


def test_all_scores_tuning_candidates_and_selected_c():
    scores = pd.read_csv(RESULTS / "fold_scores.csv", encoding="utf-8")
    tuning = pd.read_csv(RESULTS / "tuning.csv", encoding="utf-8")
    assert len(scores) == 175 and len(tuning) == 1575
    assert set(scores.config) == set(CONFIG["configs"])
    assert not scores.duplicated(["config", "repeat", "fold"]).any()
    assert not tuning.duplicated(["config", "repeat", "fold", "C"]).any()
    for keys, part in tuning.groupby(["config", "repeat", "fold"]):
        assert set(part.C) == set(CONFIG["C_grid"])
        chosen = scores.set_index(["config", "repeat", "fold"]).loc[keys, "C"]
        assert chosen == select_c(part.to_dict("records"))
    columns = ["macro_f1", "acc", "macro_p", "macro_r", "f1_neg", "f1_pos", "f1_irr", "f1_neu"]
    assert scores[columns].ge(0).all().all() and scores[columns].le(1).all().all()


def test_oof_probabilities_labels_and_pairwise_scores():
    frame, assignments = load_data("acceptance")
    for config in CONFIG["configs"]:
        oof = pd.read_csv(RESULTS / f"oof_{config}.csv", encoding="utf-8")
        assert len(oof) == 563 and np.array_equal(oof.row_index, np.arange(563))
        assert oof.sample_id.tolist() == assignments.sample_id.tolist()
        assert oof.y_true.tolist() == frame.Sentiment.tolist()
        assert oof.fold.tolist() == assignments.validation_fold.tolist()
        probabilities = oof[["p0", "p1", "p2", "p3"]].to_numpy()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), oof.y_pred)
    actual = pd.read_csv(RESULTS / "pairwise_delta.csv", encoding="utf-8")
    expected = pairwise_comparisons(pd.read_csv(RESULTS / "fold_scores.csv", encoding="utf-8"))
    pd.testing.assert_frame_equal(actual, expected)


def test_freezing_token_alignment_and_reproduction_evidence():
    metadata = read_json(FEATURES / "bertweet.json")
    assert metadata["state_sha256_before"] == metadata["state_sha256_after"]
    assert metadata["all_requires_grad_false"] and metadata["input_byte_assertions"] == 563
    tokens = np.load(FEATURES / "bertweet_tokens.npz")
    assert tokens["vectors"].dtype == np.float16
    assert tokens["vectors"].shape[1] == 768 and len(tokens["offsets"]) == 564
    assert tokens["offsets"][-1] == len(tokens["token_ids"]) == len(tokens["vectors"])
    assert np.diff(tokens["offsets"]).min() > 0
    reproduction = read_json(RESULTS / "reproducibility.json")
    assert reproduction["byte_identical"]
    assert reproduction["first_sha256"] == reproduction["second_sha256"] == file_sha(RESULTS / "fold_scores.csv")
