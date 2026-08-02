"""
Endpoint tests using FastAPI's TestClient.

Vision/NLP model-dependent tests are skipped automatically if the
corresponding artifacts haven't been trained yet (fresh clone before
running the notebooks) — this keeps `pytest` green out of the box while
still giving full coverage once models exist.
"""
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app

client = TestClient(app)
HEADERS = {"x-api-key": config.API_KEY}


def test_health_no_auth_required():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_protected_endpoint_rejects_missing_key():
    resp = client.post("/analyze-sentiment", json={"text": "great product"})
    assert resp.status_code == 422  # missing required header


def test_protected_endpoint_rejects_wrong_key():
    resp = client.post(
        "/analyze-sentiment",
        json={"text": "great product"},
        headers={"x-api-key": "wrong-key"},
    )
    assert resp.status_code == 401


@pytest.mark.skipif(
    not (config.SENTIMENT_MODEL_PATH.exists() and config.VECTORIZER_PATH.exists()),
    reason="Sentiment model not trained yet — run notebooks/03_sentiment_model_training.ipynb",
)
def test_analyze_sentiment_positive_text():
    resp = client.post(
        "/analyze-sentiment",
        json={"text": "Absolutely love this, best purchase ever!"},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"positive", "negative", "neutral"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_chatbot_greeting():
    resp = client.post("/chatbot", json={"message": "hello there"}, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched_intent"] == "greeting"
    assert body["reply"]


def test_chatbot_return_policy_intent():
    resp = client.post(
        "/chatbot", json={"message": "how do returns work"}, headers=HEADERS
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched_intent"] == "return_policy"


def test_dashboard_stats_shape():
    resp = client.get("/dashboard/stats", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "total_visits",
        "unique_customers",
        "sentiment_breakdown",
        "total_chatbot_queries",
        "top_intents",
    ):
        assert key in body


@pytest.mark.skipif(
    not config.PRODUCT_CLASSIFIER_PATH.exists(),
    reason="Product classifier not trained yet — run notebooks/01_image_classifier_training.ipynb",
)
def test_classify_product_requires_valid_image():
    resp = client.post(
        "/classify-product",
        files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
        headers=HEADERS,
    )
    assert resp.status_code == 400


@pytest.mark.skipif(
    not config.FACE_DB_PATH.exists(),
    reason="Face recognizer not trained yet — run notebooks/02_face_recognition_setup.ipynb",
)
def test_recognize_face_requires_valid_image():
    resp = client.post(
        "/recognize-face",
        files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
        headers=HEADERS,
    )
    assert resp.status_code == 400


# ---------- Online enrollment (new product types / new people on the fly) ----------

def test_enroll_product_rejects_too_few_images():
    resp = client.post(
        "/enroll-product",
        data={"label": "test-widget"},
        files=[("files", ("a.jpg", b"not-an-image", "image/jpeg"))],
        headers=HEADERS,
    )
    assert resp.status_code == 400


def test_enroll_product_rejects_builtin_class_name():
    files = [("files", (f"{i}.jpg", b"not-an-image", "image/jpeg")) for i in range(3)]
    resp = client.post(
        "/enroll-product",
        data={"label": "Sneaker"},  # already a built-in Fashion-MNIST class
        files=files,
        headers=HEADERS,
    )
    assert resp.status_code in (400, 503)  # 503 if the base model itself isn't trained yet


def test_enroll_face_rejects_too_few_images():
    resp = client.post(
        "/enroll-face",
        data={"person_name": "test-person"},
        files=[("files", ("a.jpg", b"not-an-image", "image/jpeg"))],
        headers=HEADERS,
    )
    assert resp.status_code == 400


def test_enroll_face_rejects_empty_name():
    files = [("files", (f"{i}.jpg", b"not-an-image", "image/jpeg")) for i in range(3)]
    resp = client.post(
        "/enroll-face",
        data={"person_name": "  "},
        files=files,
        headers=HEADERS,
    )
    assert resp.status_code == 400
