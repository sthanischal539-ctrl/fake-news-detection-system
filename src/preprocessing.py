"""Text preprocessing for the classical (TF-IDF) pipeline.

Operates on the FULL article text. Steps: lowercase, strip URLs/HTML/emails and
source datelines, tokenize, remove stopwords, and lemmatize.

NOTE: the transformer path uses its own subword tokenizer and only needs light
cleaning, so it calls `light_clean` rather than `preprocess`.
"""
from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --------------------------------------------------------------------------- #
# One-time NLTK resource download (quiet, cached after first run)
# --------------------------------------------------------------------------- #
_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


import warnings

def ensure_nltk() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for path, pkg in _NLTK_RESOURCES:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)


ensure_nltk()

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Regexes compiled once
_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_EMAIL_RE = re.compile(r"\S+@\S+")
# Source datelines like "WASHINGTON (Reuters) -" that leak the label in ISOT.
_DATELINE_RE = re.compile(r"^[A-Z][A-Za-z\s.,/]*\(reuters\)\s*-\s*", re.IGNORECASE)
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def light_clean(text: str) -> str:
    """Minimal cleaning suitable for the transformer tokenizer."""
    text = str(text)
    text = _DATELINE_RE.sub("", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", text).strip()


def preprocess(text: str) -> str:
    """Full cleaning for TF-IDF: clean -> tokenize -> stopwords -> lemmatize."""
    text = light_clean(text).lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOPWORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def preprocess_series(texts) -> list[str]:
    """Vectorized convenience wrapper for a pandas Series / iterable of strings."""
    return [preprocess(t) for t in texts]


if __name__ == "__main__":
    sample = "WASHINGTON (Reuters) - Officials said the <b>economy</b> grew. Visit http://x.com now!"
    print("RAW :", sample)
    print("PREP:", preprocess(sample))
