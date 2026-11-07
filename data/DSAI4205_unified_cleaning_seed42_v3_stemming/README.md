# Unified cleaned dataset v3 with stemming

Frozen dataset only. No training or embedding extraction.
563 training / 397 official test; same members, labels, order and five group folds
as DSAI4205_unified_cleaning_seed42_v2. Model input is the final lowercase Phrase in the two-column
train.csv/test.csv and supplied folds. v2 normalization is followed by Snowball
English stemming after WordNet lemmatization. Mention identity, hostname tokens
and protected negation/pronouns bypass stemming; stopwords are not removed.

Read CLEANING_REPORT.md for effects, limits and token-level audits, and
PARENT_CLEANING_REPORT.md for inherited source-ID, residual encoding damage,
exclusions and external-provenance limitations. No external source was re-read
or restored in this extension. v2's provenance metadata is inherited unchanged.
This release does not claim better model accuracy than v2.

Reproduce in the existing project environment with the frozen v2 parent present:
python scripts/data/freeze_stemmed_dataset.py
Add --freeze to publish this version; existing releases refuse overwrite.
Verify: python scripts/data/verify_unified_dataset.py outputs/DSAI4205_unified_cleaning_seed42_v3_stemming
The package includes raw course inputs, all row-level audit records, frozen fold
indices, source/linguistic resource hashes and code snapshots. Every other package
file is hashed by SHA256SUMS.json. Future rule changes require a new release.
