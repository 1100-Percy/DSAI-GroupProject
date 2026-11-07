"""Versioned common-input cleaning; independent of historical experiment profiles.

No classifier fitting, network requests, external labels or source-text recovery.
"""

from __future__ import annotations

import hashlib
import html
import re
import string
import unicodedata
from collections import defaultdict
from pathlib import Path

import emoji
import nltk
import pandas as pd
from ftfy import fix_text
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from rapidfuzz import fuzz, process

RULE_VERSION = "unified_course_v2"
HOST = r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\s*\.\s*)+[a-z]{2,24}"
URL_RE = re.compile(
    r"(?<![\w@])(?P<scheme>https?\s*:\s*/\s*/\s*)?"
    r"(?P<host>" + HOST + r")"
    r"(?P<path>(?:\s*/\s*[^\s?<>\[\]{}()]+)*)"
    r"(?P<query>\?\s*(?:[a-z0-9_.%-]+\s*=\s*[^\s&<>\[\]{}()]+"
    r"(?:\s*&\s*[a-z0-9_.%-]+\s*=\s*[^\s&<>\[\]{}()]+)*"
    r"|[^\s<>\[\]{}()]+))?",
    re.IGNORECASE,
)
MENTION_RE = re.compile(r"(?<!\w)@\s*([A-Za-z0-9_]+)")
KNOWN_SUFFIXES = frozenset(
    ["com", "org", "net", "edu", "gov", "io", "co", "in", "tv", "be", "ly", "it", "gg", "me", "biz", "news", "futbol", "uk", "ca", "de", "us", "au", "info", "app", "dev", "ai", "live", "online", "site", "xyz", "link", "cc", "to", "fm", "eu", "ru", "jp", "cn", "hk", "network"]
)
KNOWN_BARE_HOSTS = frozenset(
    ["twitter.com", "pic.twitter.com", "t.co", "twitch.tv", "youtube.com", "youtu.be", "amazon.com", "amazon.in", "facebook.com", "instagram.com"]
)
PROTECTED_WORDS = frozenset(
    ["i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs", "not", "no", "nor", "never", "url", "rhandlerr", "unk"]
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_text(text: str) -> str:
    return fix_text(html.unescape(str(text)), normalization="NFC")


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", repair_text(text)).casefold().split())


def initialize_nltk(root: Path) -> None:
    local = str(root / ".nltk_data")
    if local not in nltk.data.path:
        nltk.data.path.insert(0, local)
    # Fail explicitly rather than download resources or silently skip lemmatization.
    nltk.data.find("corpora/wordnet.zip")
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")


def normalize_input(text: str, *, morphology: str = "lemma") -> tuple[str, dict]:
    """Return lowercase common input plus an audit of lossy input transformations.

    Punctuation is replaced with boundaries, not concatenated. Known English
    negative contractions expand before punctuation removal; lost apostrophes
    are never guessed. Mentions/domains are exempt from both morphology steps.
    lemma_stem applies Snowball English after WordNet; lemma preserves v2 behaviour.
    """
    if morphology not in {"lemma", "lemma_stem"}:
        raise ValueError(f"Unknown morphology pipeline: {morphology}")
    value = unicodedata.normalize("NFKC", str(text))
    urls, mentions, protected, suspicious_urls = [], [], {}, []

    def protect(words: list[str]) -> str:
        key = f"zzreservedtoken{len(protected)}zz"
        if key in value.lower():
            raise ValueError("Input collides with reserved protection token")
        protected[key] = words
        return f" {key} "

    def url_match(match: re.Match) -> str:
        raw_host = match.group("host")
        prefix = ""
        # A sentence-ending word must not become an invented subdomain merely
        # because the next sentence begins with a URL: 'broken. In.tt/abc'.
        boundary = re.search(r"\.\s+(?=[A-Z][a-z][^.]*\.)", raw_host)
        if boundary:
            prefix, raw_host = raw_host[:boundary.end()], raw_host[boundary.end():]
        host = re.sub(r"\s+", "", raw_host).lower()
        has_path = bool(match.group("path") or match.group("query"))
        suffix = host.rsplit(".", 1)[-1]
        accepted = (len(suffix) == 2 or suffix in KNOWN_SUFFIXES) and (
            match.group("scheme") or host.startswith("www.") or has_path
            or host in KNOWN_BARE_HOSTS
        )
        if not accepted:
            if match.group("scheme") or has_path:
                suspicious_urls.append({"matched_text": match.group(),
                                        "reason": "unrecognized_domain_kept_without_guessing"})
            return match.group()
        raw = match.group()[len(prefix):]
        # Punctuation outside a matched URL is irrelevant to this punctuation-free
        # input, but remains intact in repaired_text and in this audit record.
        urls.append({"matched_text": raw, "hostname": host,
                     "damaged": "\ufffd" in raw,
                     "truncated": raw.rstrip().endswith(("...", "\ufffd")),
                     "start": match.start() + len(prefix), "end": match.end()})
        words = ["url", *re.findall(r"[a-z0-9]+", host)]
        return prefix + protect(words)

    value = URL_RE.sub(url_match, value)
    for match in re.finditer(r"\bhttps?\s*:\s*/\s*/[^\n]{0,80}", value, re.IGNORECASE):
        suspicious_urls.append({"matched_text": match.group(),
                                "reason": "incomplete_scheme_or_host_kept_without_guessing"})

    def mention_match(match: re.Match) -> str:
        handle = match.group(1).lower()
        words = [part for part in handle.split("_") if part]
        mentions.append({"matched_text": match.group(), "handle": handle,
                         "tokens": words, "ambiguous_short_handle": handle in {"me", "user"}})
        return protect(words)

    value = MENTION_RE.sub(mention_match, value)
    value = emoji.demojize(value, delimiters=(" ", " "))
    value = value.lower().replace("\u2019", "'")
    value = re.sub(r"\b(can't|won't|shan't)\b", lambda m: {
        "can't": "can not", "won't": "will not", "shan't": "shall not"
    }[m.group()], value)
    value = re.sub(r"\b([a-z]+)n't\b", r"\1 not", value)
    # Preserve unknown replacement characters and non-punctuation symbols; the
    # audit records residual damage. Do not invent <CORRUPT> or missing letters.
    value = "".join(" " if c in string.punctuation or unicodedata.category(c).startswith("P")
                    else c for c in value)
    tokens = re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE)
    tag_tokens = [t for t in tokens if t not in protected]
    tags = iter(nltk.pos_tag(tag_tokens, lang="eng"))
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    result, lemma_result, changes, stem_changes = [], [], [], []
    for token in tokens:
        if token in protected:
            result.extend(protected[token])
            lemma_result.extend(protected[token])
            continue
        _, tag = next(tags)
        pos = {"J": "a", "V": "v", "N": "n", "R": "r"}.get(tag[:1])
        lemma = token
        if pos and token.isalpha() and len(token) > 2 and token not in PROTECTED_WORDS:
            lemma = lemmatizer.lemmatize(token, pos=pos)
        lemma_result.append(lemma)
        stem = lemma
        if morphology == "lemma_stem" and re.fullmatch(r"[a-z]{3,}", lemma) and (
            lemma not in PROTECTED_WORDS
        ):
            stem = stemmer.stem(lemma)
        if stem != lemma:
            stem_changes.append({"token": token, "lemma": lemma, "stem": stem,
                                 "pos": tag, "output_token_index": len(result)})
        result.append(stem)
        if lemma != token:
            changes.append({"token": token, "lemma": lemma, "pos": tag})
    output = " ".join(result).lower()
    if not output.strip():
        raise ValueError("Cleaning produced an empty input; manual review required")
    assert output == output.lower()
    assert not any(unicodedata.category(c).startswith("P") for c in output)
    assert not any(c in string.punctuation for c in output)
    details = {"urls": urls, "mentions": mentions, "lemma_changes": changes,
               "suspicious_urls": suspicious_urls}
    if morphology == "lemma_stem":
        details.update({"lemmatized_text": " ".join(lemma_result).lower(),
                        "stem_changes": stem_changes})
    return output, details


def read_course(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if set(frame.columns) != {"Phrase", "Sentiment"}:
        raise ValueError("Course input must contain exactly Phrase and Sentiment")
    if frame.Phrase.map(lambda x: not isinstance(x, str) or not x.strip()).any():
        raise ValueError("Missing or empty Phrase")
    frame["Sentiment"] = pd.to_numeric(frame.Sentiment, errors="raise")
    if not frame.Sentiment.isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid target")
    frame["Sentiment"] = frame.Sentiment.astype(int)
    frame.insert(0, "sample_id", [f"{split}:{i:04d}" for i in range(len(frame))])
    frame.insert(1, "source_row_index", range(len(frame)))
    frame["split"] = split
    frame["repaired_text"] = frame.Phrase.map(repair_text)
    frame["canonical_text"] = frame.Phrase.map(canonical)
    frame["replacement_count"] = frame.repaired_text.str.count("\ufffd")
    frame["repair_status"] = [
        "partially_repaired" if "\ufffd" in repaired else
        "fully_repaired" if raw != repaired else "unchanged"
        for raw, repaired in zip(frame.Phrase, frame.repaired_text, strict=True)
    ]
    frame["raw_question_count"] = frame.Phrase.str.count(r"\?")
    frame["repaired_question_count"] = frame.repaired_text.str.count(r"\?")
    frame["uppercase_word_count"] = frame.repaired_text.map(
        lambda s: len(re.findall(r"\b[A-Z]{2,}\b", s)))
    return frame


def provenance_audit(course: pd.DataFrame, external_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Unique source-ID evidence, never copy external text/labels into model input.

    Damage-compatible matches require every retained character to agree and only
    let existing U+FFFD spans match punctuation. Fuzzy scores generate candidates
    but never alone establish a source ID.
    """
    keys_to_ids: dict[str, set[int]] = defaultdict(set)
    hashes = {}
    for name in ("training", "validation"):
        path = external_dir / f"twitter_{name}.csv"
        hashes[path.name] = sha256(path)
        external = pd.read_csv(path, header=None, usecols=[0, 3], names=["id", "text"])
        for row in external.dropna(subset=["text"]).itertuples(index=False):
            keys_to_ids[canonical(row.text)].add(int(row.id))
    choices = sorted(keys_to_ids)
    rows = []
    for row in course.itertuples(index=False):
        key = row.canonical_text
        ids = set(keys_to_ids.get(key, set()))
        method = "exact_canonical" if ids else "unmatched"
        evidence = []
        if not ids and "\ufffd" in key:
            parts = re.split(r"\s*\ufffd+\s*", key)
            pattern = re.compile(r"\s*[^\w\s]{1,6}\s*".join(re.escape(p) for p in parts))
            for candidate, score, _ in process.extract(
                key, choices, scorer=fuzz.ratio, limit=100, score_cutoff=70
            ):
                if pattern.fullmatch(candidate):
                    ids.update(keys_to_ids[candidate])
                    evidence.append({"candidate_sha256": hashlib.sha256(
                        candidate.encode("utf-8")).hexdigest(), "similarity": score})
            if ids:
                method = "damage_compatible_retained_characters"
        unique = len(ids) == 1
        rows.append({"sample_id": row.sample_id, "source_row_index": row.source_row_index,
                     "split": row.split, "match_status": "unique" if unique else
                     "ambiguous" if ids else "unmatched", "match_method": method,
                     "source_group_id": f"source:{next(iter(ids))}" if unique else "",
                     "candidate_source_ids": sorted(ids), "damage_match_evidence": evidence})
    return pd.DataFrame(rows), hashes


class Components:
    """Deterministic union-find for provenance and identical-input grouping."""

    def __init__(self, values: list[str]):
        self.parent = {v: v for v in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def assign_groups(frame: pd.DataFrame) -> pd.Series:
    groups = Components(frame.sample_id.tolist())
    for column in ("source_group_id", "canonical_text", "model_text"):
        for value, rows in frame.groupby(column, sort=True):
            if value == "":
                continue
            members = rows.sample_id.tolist()
            for member in members[1:]:
                groups.union(members[0], member)
    return frame.sample_id.map(lambda x: "group:" + groups.find(x))
