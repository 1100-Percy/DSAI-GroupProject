"""Checks for protocol failures that could invalidate the comparison."""
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from stage1.data_io import CONFIG, ROOT, load_data, make_repeats
from stage1.embeddings import mean_vectors, pool_bertweet, train_word_vectors
from stage1.probe import comparison_detail, feature_transform, select_c
from stage1.build_notebook import notebook_source


def test_group_integrity_and_phrase_preservation():
    frame, assignments = load_data("tests")
    original = frame.Phrase.tolist()
    repeated = make_repeats(frame, assignments)
    assert len(repeated) == 2815
    for _, part in repeated.groupby("repeat"):
        joined = part.merge(assignments[["row_index", "group_id"]], on="row_index", validate="one_to_one")
        assert joined.groupby("group_id").fold.nunique().eq(1).all()
    assert frame.Phrase.tolist() == original


def test_c_tolerance_is_relative_to_global_best():
    candidates = [{"C": c, "inner_f1_mean": score} for c, score in [(0.001, .5), (.003, .5009), (.01, .5018)]]
    assert select_c(candidates) == .003
    assert select_c([{"C": .001, "inner_f1_mean": .5}, {"C": 10, "inner_f1_mean": .502}]) == 10


def test_real_token_and_last_four_layer_pooling():
    hidden = tuple(torch.full((1, 4, 2), float(i)) for i in range(13))
    hidden = tuple(x + torch.tensor([0., 2., 4., 99.]).view(1, 4, 1) for x in hidden)
    sentence, tokens, mask = pool_bertweet(hidden, torch.tensor([[1, 1, 1, 0]]), torch.tensor([[1, 0, 1, 1]]))
    assert mask.tolist() == [[False, True, False, False]]
    assert sentence.tolist() == [[12.5, 12.5]]


def test_word_models_never_train_on_validation_rows():
    phrases = ["apple apple fruit", "pear fruit", "validationexclusive"]
    vectors, vocabulary = train_word_vectors("w2v", phrases, np.array([0, 1]), np.array([2]))
    assert "validationexclusive" not in vocabulary
    assert (vectors[2] == 0).all()
    ft_vectors, ft_vocab = train_word_vectors("ft", phrases, np.array([0, 1]), np.array([2]))
    assert "validationexclusive" not in ft_vocab
    assert np.any(ft_vectors[2] != 0)
    with pytest.raises(AssertionError):
        train_word_vectors("w2v", phrases, np.array([0, 1]), np.array([1, 2]))


def test_fold_transforms_fit_only_supplied_rows():
    x = np.arange(120 * 110, dtype=float).reshape(120, 110)
    transform = feature_transform("bertweet_pca100")
    a = transform.fit_transform(x[:110])
    assert a.shape == (110, 100)
    assert transform.named_steps["pca"].n_samples_ == 110
    np.testing.assert_allclose(transform.named_steps["pca"].mean_, x[:110].mean(axis=0))
    assert transform.transform(x[110:]).shape == (10, 100)


def test_decision_uses_favored_direction_and_does_not_count_ties():
    scores = pd.DataFrame([{"config": config, "repeat": i // 5, "fold": i % 5 + 1, "macro_f1": value}
                           for i in range(25) for config, value in [("a", .3), ("b", .4)]])
    assert comparison_detail(scores, "a", "b")["decision"] == "更差"
    assert comparison_detail(scores, "b", "a")["decision"] == "更好"
    assert comparison_detail(scores, "a", "a")["decision"] == "无法区分"


def test_source_restrictions_and_notebook_inlining():
    for path in (ROOT / "stage1").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.name != "baseline_ref.py":
            assert "Tfidf" not in source
        assert "test.csv" not in source and "audit.csv" not in source
        inline = notebook_source(path)
        tree = ast.parse(inline)
        assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree))
        assert not any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith("stage1") for n in ast.walk(tree))
