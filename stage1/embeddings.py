"""Mean-pooled representations; no labels used for representation fitting."""
import gc
import importlib.util
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from .data_io import (CONFIG, DATA, FEATURES, RESULTS, ROOT, cache_identity, file_sha,
                      load_data, read_json, write_json)


def mean_vectors(tokens, lookup, dimension):
    vectors = [lookup(token) for token in tokens]
    vectors = [vector for vector in vectors if vector is not None]
    return np.mean(vectors, axis=0).astype(np.float32) if vectors else np.zeros(dimension, np.float32)


def lexical_coverage(phrases, vocabulary, vectors, embedding, repeat=-1, fold=-1, **extra):
    tokens = [token for phrase in phrases for token in phrase.split(" ")]
    unique = set(tokens)
    missing = [token for token in tokens if token not in vocabulary]
    return {"embedding": embedding, "repeat": repeat, "fold": fold,
            "token_count": len(tokens), "type_count": len(unique),
            "unseen_token_rate": len(missing) / len(tokens),
            "unseen_type_rate": len(set(missing)) / len(unique),
            "oov_token_rate": len(missing) / len(tokens),
            "oov_type_rate": len(set(missing)) / len(unique),
            "zero_rows": int(np.all(vectors == 0, axis=1).sum()), **extra}


def train_word_vectors(kind, phrases, train_indices, valid_indices):
    assert os.environ.get("PYTHONHASHSEED") == "0", "Set PYTHONHASHSEED=0 before process startup"
    from gensim.models import FastText, Word2Vec
    train_indices = np.asarray(train_indices)
    valid_indices = np.asarray(valid_indices)
    assert len(np.intersect1d(train_indices, valid_indices)) == 0
    assert len(set(train_indices.tolist())) == len(train_indices)
    sentences = [phrases[int(i)].split(" ") for i in train_indices]
    assert len(sentences) == len(train_indices)
    kwargs = dict(CONFIG["word_vectors"])
    cls = Word2Vec if kind == "w2v" else FastText
    if kind == "ft":
        kwargs.update(CONFIG["fasttext"])
    model = cls(sentences=sentences, **kwargs)
    vocabulary = set(model.wv.key_to_index)
    def lookup(token):
        return model.wv[token] if kind == "ft" or token in vocabulary else None
    vectors = np.stack([mean_vectors(p.split(" "), lookup, 100) for p in phrases])
    assert vectors.shape == (len(phrases), 100) and np.isfinite(vectors).all()
    return vectors, vocabulary


def fold_word_features(kind, frame, train, valid, repeat, fold, recompute=False):
    path = FEATURES / f"{kind}_r{repeat}_f{fold}.npz"
    metadata_path = path.with_suffix(".json")
    identity = {**cache_identity(), "train_indices": train.tolist(), "valid_indices": valid.tolist(),
                "embedding": kind, "repeat": int(repeat), "fold": int(fold)}
    if path.exists() and not recompute:
        metadata = read_json(metadata_path)
        assert metadata["identity"] == identity
        assert metadata["sha256"] == file_sha(path)
        return np.load(path)["vectors"], metadata["coverage"]
    vectors, vocabulary = train_word_vectors(kind, frame.Phrase.tolist(), train, valid)
    coverage = lexical_coverage(frame.Phrase.iloc[valid], vocabulary, vectors[valid], kind, repeat, fold)
    if kind == "ft":
        coverage.update(oov_token_rate=0.0, oov_type_rate=0.0)
    np.savez_compressed(path, vectors=vectors)
    write_json(metadata_path, {"identity": identity, "sha256": file_sha(path), "coverage": coverage})
    return vectors, coverage


def load_normalizer():
    import nltk
    nltk.data.path.insert(0, str(ROOT / ".nltk_data"))
    source = DATA / "code/unified_data.py"
    spec = importlib.util.spec_from_file_location("stage1_original_normalizer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.initialize_nltk(ROOT)
    return module.normalize_input


def extract_glove(frame):
    import gensim.downloader as api
    identity = cache_identity()
    metadata_path = FEATURES / "glove.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        assert metadata["identity"] == identity
        for name, expected in metadata["hashes"].items():
            assert file_sha(FEATURES / name) == expected
        return metadata["coverage"]
    print("Loading glove-twitter-100 via gensim.downloader", flush=True)
    glove = api.load(CONFIG["glove"])
    direct = np.stack([mean_vectors(p.split(" "), lambda t: glove[t] if t in glove else None, 100)
                       for p in frame.Phrase])
    normalize = load_normalizer()
    map_path = FEATURES / "glove_stem_map.pkl"
    map_meta = FEATURES / "glove_stem_map.json"
    if map_path.exists() and map_meta.exists():
        mapping_metadata = read_json(map_meta)
        assert mapping_metadata["identity"] == identity
        assert file_sha(map_path) == mapping_metadata["sha256"]
        with map_path.open("rb") as stream:
            mapping = pickle.load(stream)
    else:
        sums, counts, transformed = {}, Counter(), []
        skipped_empty, skipped_multi = 0, 0
        # No vocabulary prefilter: every pretrained entry receives the original function.
        for i, token in enumerate(glove.index_to_key):
            try:
                output, _ = normalize(token, morphology="lemma_stem")
            except ValueError as exc:
                if "Cleaning produced an empty input" not in str(exc):
                    raise
                output = ""
                skipped_empty += 1
            if len(output.split(" ")) != 1 or not output:
                skipped_multi += bool(output)
                transformed.append(None)
                continue
            transformed.append(output)
            if output not in sums:
                sums[output] = glove[token].copy()
            else:
                sums[output] += glove[token]
            counts[output] += 1
            if (i + 1) % 100000 == 0:
                print(f"GloVe normalized {i + 1}/{len(glove)}", flush=True)
        mapping = {key: vector / np.float32(counts[key]) for key, vector in sums.items()}
        top = sorted(counts, key=lambda key: (-counts[key], key))[:20]
        top_set = set(top)
        members = {key: [] for key in top}
        for original, output in zip(glove.index_to_key, transformed):
            if output in top_set:
                members[output].append(original)
        pd.DataFrame([{"v3_form": key, "merged_words": counts[key],
                       "source_words": " | ".join(members[key])} for key in top]).to_csv(
                           RESULTS / "glove_merges_top20.csv", index=False, encoding="utf-8")
        with map_path.open("wb") as stream:
            pickle.dump(mapping, stream, protocol=pickle.HIGHEST_PROTOCOL)
        mapping_metadata = {"identity": identity, "sha256": file_sha(map_path),
                            "pretrained_words": len(glove), "aligned_forms": len(mapping),
                            "merged_forms": sum(n > 1 for n in counts.values()),
                            "skipped_empty": skipped_empty, "skipped_multiword": int(skipped_multi),
                            "prefilter": False, "aggregation_dtype": "float32"}
        write_json(map_meta, mapping_metadata)
        del sums, counts, transformed
        gc.collect()
    stem = np.stack([mean_vectors(p.split(" "), mapping.get, 100) for p in frame.Phrase])
    coverage = [lexical_coverage(frame.Phrase, glove.key_to_index, direct, "glove_direct"),
                lexical_coverage(frame.Phrase, mapping, stem, "glove_stem")]
    rescued = Counter(t for p in frame.Phrase for t in p.split(" ") if t not in glove and t in mapping)
    pd.DataFrame(rescued.most_common(20), columns=["word", "count"]).to_csv(
        RESULTS / "glove_rescued_top20.csv", index=False, encoding="utf-8")
    np.save(FEATURES / "glove_direct.npy", direct)
    np.save(FEATURES / "glove_stem.npy", stem)
    hashes = {name: file_sha(FEATURES / name) for name in ["glove_direct.npy", "glove_stem.npy", "glove_stem_map.pkl"]}
    write_json(metadata_path, {"identity": identity, "hashes": hashes, "coverage": coverage})
    del glove, mapping
    gc.collect()
    return coverage


def tensor_state_sha(model):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pool_bertweet(hidden_states, attention_mask, special_tokens_mask):
    import torch
    assert len(hidden_states) == 13
    token_features = torch.stack(hidden_states[-4:]).mean(dim=0)
    mask = attention_mask.bool() & ~special_tokens_mask.bool()
    assert mask.sum(dim=1).gt(0).all()
    sentences = (token_features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return sentences, token_features, mask


def extract_bertweet(frame):
    import torch
    from transformers import AutoModel, AutoTokenizer
    metadata_path = FEATURES / "bertweet.json"
    identity = cache_identity()
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        assert metadata["identity"] == identity
        for name, expected in metadata["hashes"].items():
            assert file_sha(FEATURES / name) == expected
        assert metadata["state_sha256_before"] == metadata["state_sha256_after"]
        return metadata["coverage"]
    cfg = CONFIG["bertweet"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"], normalization=False)
    assert tokenizer.normalization is False
    model = AutoModel.from_pretrained(cfg["checkpoint"], torch_dtype=torch.float32)
    model.requires_grad_(False)
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    model.eval().to(device)
    before = tensor_state_sha(model)
    phrases = frame.Phrase.tolist()
    sentence_vectors, token_vectors, token_ids, offsets = [], [], [], [0]
    truncated_rows = unk_tokens = unk_rows = 0
    word_counts = Counter(t for p in phrases for t in p.split(" "))
    pieces = {word: len(tokenizer.tokenize(word)) for word in word_counts}
    for start in range(0, len(phrases), cfg["batch_size"]):
        texts = phrases[start:start + cfg["batch_size"]]
        for i, text in enumerate(texts, start):
            assert text.encode("utf-8") == frame.Phrase.iloc[i].encode("utf-8")
        lengths = [len(tokenizer.encode(text, truncation=False)) for text in texts]
        truncated_rows += sum(n > cfg["max_length"] for n in lengths)
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=cfg["max_length"],
                            return_special_tokens_mask=True, return_tensors="pt")
        special = encoded.pop("special_tokens_mask").to(device)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)
            sentences, tokens, mask = pool_bertweet(output.hidden_states, encoded["attention_mask"], special)
        sentence_vectors.append(sentences.cpu().numpy().astype(np.float32))
        for row in range(len(texts)):
            vectors = tokens[row][mask[row]].cpu().numpy().astype(np.float16)
            ids = encoded["input_ids"][row][mask[row]].cpu().numpy()
            n_unk = int((ids == tokenizer.unk_token_id).sum())
            unk_tokens += n_unk
            unk_rows += n_unk > 0
            token_vectors.append(vectors)
            token_ids.append(ids)
            offsets.append(offsets[-1] + len(ids))
        print(f"BERTweet {min(start + len(texts), len(phrases))}/{len(phrases)} on {device}", flush=True)
    after = tensor_state_sha(model)
    assert before == after
    assert all(not p.requires_grad for p in model.parameters())
    vectors = np.concatenate(sentence_vectors)
    assert vectors.shape == (563, 768) and np.isfinite(vectors).all()
    np.save(FEATURES / "bertweet.npy", vectors)
    np.savez_compressed(FEATURES / "bertweet_tokens.npz", vectors=np.concatenate(token_vectors),
                        token_ids=np.concatenate(token_ids), offsets=np.asarray(offsets),
                        row_index=np.arange(len(frame)))
    n_words = sum(word_counts.values())
    coverage = {"embedding": "bertweet", "repeat": -1, "fold": -1,
                "subwords_per_word": sum(pieces[w] * n for w, n in word_counts.items()) / n_words,
                "split_word_rate": sum(n for w, n in word_counts.items() if pieces[w] >= 2) / n_words,
                "unk_tokens": unk_tokens, "unk_rows": int(unk_rows), "truncated_rows": truncated_rows,
                "zero_rows": int(np.all(vectors == 0, axis=1).sum())}
    hashes = {name: file_sha(FEATURES / name) for name in ["bertweet.npy", "bertweet_tokens.npz"]}
    write_json(metadata_path, {"identity": identity, "hashes": hashes, "coverage": coverage,
               "state_sha256_before": before, "state_sha256_after": after, "device": device,
               "all_requires_grad_false": True, "input_byte_assertions": len(phrases),
               "model_revision": model.config._commit_hash, "dtype": "float32",
               "pooling": "mean layers 9-12 then mean non-special non-padding tokens"})
    del model
    gc.collect()
    return coverage


def extract_all():
    frame, _ = load_data("extract")
    coverage = extract_glove(frame)
    coverage.append(extract_bertweet(frame))
    pd.DataFrame(coverage).to_csv(RESULTS / "coverage_pretrained.csv", index=False, encoding="utf-8")


def load_pretrained():
    identity = cache_identity()
    for source in ["glove.json", "bertweet.json"]:
        metadata = read_json(FEATURES / source)
        assert metadata["identity"] == identity
        for name, expected in metadata["hashes"].items():
            assert file_sha(FEATURES / name) == expected
    return {name: np.load(FEATURES / f"{name}.npy") for name in ["glove_stem", "glove_direct", "bertweet"]}
