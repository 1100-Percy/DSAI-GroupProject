"""The sole location of the sparse TF-IDF reference representation."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def space_tokens(text):
    return text.split(" ")


def baseline_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", tokenizer=space_tokens,
                                token_pattern=None, lowercase=False, ngram_range=(1, 1),
                                sublinear_tf=True, min_df=2, norm="l2")),
        ("char", TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=(3, 5),
                                sublinear_tf=True, min_df=2, norm="l2")),
    ])
