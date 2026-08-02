"""
Module B1 (Text Preprocessing) + B2 (Sentiment Analysis).

clean_text() is the shared preprocessing function used both at training
time (in 03_sentiment_model_training.ipynb) and at inference time here —
keeping it in one place avoids train/serve skew.
"""
import pickle
import re
import string
from typing import Tuple

from app import config


_lemmatizer = None
_stopwords = None


def _ensure_nltk_resources():
    """Lazily pull in NLTK's lemmatizer + stopword list on first use."""
    global _lemmatizer, _stopwords
    if _lemmatizer is not None:
        return
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    _lemmatizer = WordNetLemmatizer()
    _stopwords = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Standard NLP cleaning pipeline:
    lowercase -> strip punctuation/digits -> tokenize -> remove stopwords
    -> lemmatize -> rejoin.
    """
    _ensure_nltk_resources()

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                          # digits
    tokens = text.split()
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 1]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


class ModelNotTrainedError(RuntimeError):
    pass


class SentimentAnalyzer:
    """TF-IDF + Logistic Regression sentiment classifier (baseline model)."""

    def __init__(self):
        self._model = None
        self._vectorizer = None

    def _load(self):
        if self._model is not None:
            return
        if not config.SENTIMENT_MODEL_PATH.exists() or not config.VECTORIZER_PATH.exists():
            raise ModelNotTrainedError(
                "sentiment_model.pkl / vectorizer.pkl not found. Run "
                "notebooks/03_sentiment_model_training.ipynb first."
            )
        with open(config.SENTIMENT_MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        with open(config.VECTORIZER_PATH, "rb") as f:
            self._vectorizer = pickle.load(f)

    def predict(self, text: str) -> Tuple[str, float, str]:
        self._load()
        cleaned = clean_text(text)
        vec = self._vectorizer.transform([cleaned])
        label = self._model.predict(vec)[0]

        # Not every sklearn classifier exposes predict_proba (e.g. plain LinearSVC).
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(vec)[0]
            confidence = float(max(proba))
        else:
            confidence = 1.0  # decision-function-only models: report full confidence

        return str(label), confidence, cleaned


sentiment_analyzer = SentimentAnalyzer()
