"""
Central configuration for the Smart Retail platform.
All paths and settings are read from here so services/routers never
hardcode paths, and everything can be overridden with environment
variables when deployed (Docker / cloud).
"""
import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent          # app/
PROJECT_ROOT = BASE_DIR.parent                       # smart-retail-ai/
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = PROJECT_ROOT / "data"
FACES_DIR = DATA_DIR / "faces"
LOGS_PATH = MODELS_DIR / "logs.jsonl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

# --- Model artifact filenames ---
PRODUCT_CLASSIFIER_PATH = MODELS_DIR / "product_classifier.h5"
CLASS_LABELS_PATH = MODELS_DIR / "class_labels.json"

FACE_DB_PATH = MODELS_DIR / "face_db.pkl"          # LBPH recognizer (cv2.face)
FACE_LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
HAAR_CASCADE_PATH = None  # resolved lazily via cv2.data.haarcascades in cv_utils.py

# Prototype store for online product-class enrollment (nearest-centroid classifier
# built on frozen MobileNetV2 embeddings — see cv_service.ProductClassifier.enroll).
PRODUCT_PROTOTYPES_PATH = MODELS_DIR / "product_prototypes.pkl"

# --- Online-learning tunables ---
MIN_ENROLL_IMAGES = 3            # minimum photos required per enrollment call
FACE_ENROLL_SIZE = (200, 200)    # must match notebooks/02_face_recognition_setup.ipynb


SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

CHATBOT_MODEL_PATH = MODELS_DIR / "chatbot_model.pkl"
INTENTS_PATH = DATA_DIR / "intents.json"

REVIEWS_CSV_PATH = DATA_DIR / "reviews.csv"
VISITS_LOG_PATH = MODELS_DIR / "customer_visits.jsonl"

# --- API auth (simulated production security) ---
# In real deployment set this via environment variable, never commit real keys.
API_KEY = os.environ.get("API_KEY", "dev-secret-key")

# --- Misc ---
IMAGE_SIZE = (128, 128)          # product classifier input size
FACE_RECOGNITION_CONFIDENCE_THRESHOLD = 70.0  # LBPH: LOWER distance = more confident; above this -> "unknown"
