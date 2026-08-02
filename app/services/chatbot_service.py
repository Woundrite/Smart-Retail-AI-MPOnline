"""
Module B3 — Chatbot basics.

Hybrid design:
1. RULE PASS — exact/substring keyword matching against `patterns` in
   intents.json. Fast, deterministic, zero training needed. Handles the
   bulk of FAQ traffic.
2. ML FALLBACK — if no rule matches confidently, a TF-IDF + Logistic
   Regression intent classifier (trained on the same intents.json patterns)
   predicts the closest intent. This model is trained lazily on first use
   and cached to chatbot_model.pkl, so there's no separate notebook step
   required — it "just works" the first time /chatbot is called.

Both paths return a random response from the matched intent's `responses`
list, plus which method + confidence was used (useful for the A/B testing
stretch goal and for logging).
"""
import json
import pickle
import random
from typing import Dict, List, Tuple

from app import config
from app.services.nlp_service import clean_text


class ChatbotService:
    def __init__(self):
        self._intents: List[Dict] = []
        self._vectorizer = None
        self._classifier = None
        self._loaded = False

    # ---------------------------------------------------------------
    def _load_intents(self):
        with open(config.INTENTS_PATH) as f:
            data = json.load(f)
        self._intents = data["intents"]

    def _train_ml_fallback(self):
        """Train (or load cached) TF-IDF + LogisticRegression intent classifier."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        if config.CHATBOT_MODEL_PATH.exists():
            with open(config.CHATBOT_MODEL_PATH, "rb") as f:
                cached = pickle.load(f)
            self._vectorizer, self._classifier = cached["vectorizer"], cached["classifier"]
            return

        texts, labels = [], []
        for intent in self._intents:
            for pattern in intent["patterns"]:
                texts.append(clean_text(pattern))
                labels.append(intent["tag"])

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X, labels)

        self._vectorizer, self._classifier = vectorizer, classifier
        with open(config.CHATBOT_MODEL_PATH, "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "classifier": classifier}, f)

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._load_intents()
        self._train_ml_fallback()
        self._loaded = True

    # ---------------------------------------------------------------
    def _rule_match(self, message: str) -> Tuple[Dict, float]:
        """Simple substring/keyword scoring — no training required."""
        cleaned = clean_text(message)
        tokens = set(cleaned.split())
        best_intent, best_score = None, 0.0

        for intent in self._intents:
            for pattern in intent["patterns"]:
                pattern_tokens = set(clean_text(pattern).split())
                if not pattern_tokens:
                    continue
                overlap = len(tokens & pattern_tokens) / len(pattern_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best_intent = intent

        return best_intent, best_score

    def get_reply(self, message: str) -> Dict:
        self._ensure_loaded()

        # 1. Try the rule pass first.
        intent, score = self._rule_match(message)
        method = "rule"

        # 2. Fall back to the ML classifier if the rule match is weak.
        if intent is None or score < 0.5:
            cleaned = clean_text(message)
            vec = self._vectorizer.transform([cleaned])
            predicted_tag = self._classifier.predict(vec)[0]
            proba = self._classifier.predict_proba(vec)[0]
            score = float(max(proba))
            intent = next((i for i in self._intents if i["tag"] == predicted_tag), None)
            method = "ml_fallback"

        if intent is None:
            return {
                "reply": "Sorry, I didn't quite catch that — could you rephrase, or ask about "
                         "orders, returns, store hours, or product info?",
                "matched_intent": "fallback",
                "method": method,
                "confidence": 0.0,
            }

        self._log_query(message, intent["tag"], method)

        return {
            "reply": random.choice(intent["responses"]),
            "matched_intent": intent["tag"],
            "method": method,
            "confidence": round(float(score), 3),
        }

    @staticmethod
    def _log_query(message: str, tag: str, method: str):
        import time
        entry = {"message": message, "intent": tag, "method": method, "timestamp": time.time()}
        chatbot_log_path = config.MODELS_DIR / "chatbot_logs.jsonl"
        with open(chatbot_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


chatbot_service = ChatbotService()
