# Smart Retail & Customer Intelligence Platform

An AI-powered platform that recognizes returning customers via face recognition,
classifies product images, analyzes review/chat sentiment, answers FAQs via a
hybrid chatbot, and exposes everything through a single FastAPI service.

```
Client (dashboard / Postman / webcam) --REST--> FastAPI Gateway --> CV / NLP / Chatbot services --> pickled models
```

## 1. Folder structure

```
smart-retail-ai/
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── config.py              # settings + API key auth
│   ├── schemas.py             # Pydantic request/response models
│   ├── routers/
│   │   ├── vision.py          # /classify-product, /recognize-face
│   │   ├── nlp.py             # /analyze-sentiment
│   │   └── chatbot.py         # /chatbot, /dashboard/stats
│   ├── services/
│   │   ├── cv_utils.py        # OpenCV basics (grayscale/resize/blur/canny/Haar)
│   │   ├── cv_service.py      # image classifier + face recognition logic
│   │   ├── nlp_service.py     # text cleaning + sentiment
│   │   └── chatbot_service.py # rule-based + ML-fallback chatbot
│   └── models/                # trained .h5 / .pkl artifacts land here
├── notebooks/
│   ├── 01_image_classifier_training.ipynb
│   ├── 02_face_recognition_setup.ipynb
│   └── 03_sentiment_model_training.ipynb
├── data/
│   ├── reviews.csv            # real NLTK movie_reviews corpus (Pang & Lee), swap for Kaggle Women's E-Commerce Reviews if you want retail-domain text
│   ├── intents.json           # 25 chatbot intents (FAQ)
│   └── faces/                 # one sub-folder per person, put 5-10 photos each
├── tests/
│   └── test_endpoints.py
├── requirements.txt
├── Dockerfile
└── .github/workflows/deploy.yml
```

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 3. Train the models (run once)

Every notebook trains on a **real, public dataset** — nothing here is
synthetic or placeholder data. Each notebook is also runnable as a script.
Order matters because the API expects the `.pkl` / `.h5` files to already
exist in `app/models/`.

| Step | Notebook | Real dataset used | Produces |
|---|---|---|---|
| 1 | `01_image_classifier_training.ipynb` | **Fashion-MNIST** (Zalando Research) — 70,000 real labeled clothing/footwear images across 10 categories, downloaded via `tf.keras.datasets.fashion_mnist` (needs internet on first run) | `app/models/product_classifier.h5`, `app/models/class_labels.json` |
| 2 | `02_face_recognition_setup.ipynb` | Your own real photos in `data/faces/<person>/*.jpg` if present, otherwise **LFW — Labeled Faces in the Wild**, downloaded via `sklearn.datasets.fetch_lfw_people` (real public-figure photographs, needs internet on first run) | `app/models/face_db.pkl`, `app/models/label_map.json` |
| 3 | `03_sentiment_model_training.ipynb` | The real **NLTK `movie_reviews` corpus** (Pang & Lee, 2000 real human-written movie reviews, Positive/Negative) — already exported to `data/reviews.csv` | `app/models/sentiment_model.pkl`, `app/models/vectorizer.pkl` |

The chatbot's intent-classification fallback trains itself automatically on
first run from `data/intents.json` (a real, hand-authored FAQ set — building
a custom intents file is standard practice for retail chatbots) and caches
to `app/models/chatbot_model.pkl`.

> **Swapping in retail-domain data**: notebook 1 has a note on switching to a
> real Kaggle *Retail Product Checkout Dataset* organized as
> `data/product_images/<class>/*.jpg`; notebook 3 has a note on swapping to
> the real Kaggle *Women's E-Commerce Clothing Reviews* dataset (same
> `review_text` / `sentiment` columns) for retail-specific text and a real
> Neutral class.

## 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Swagger docs: http://localhost:8000/docs

All endpoints (except `/health`) require a header:

```
x-api-key: dev-secret-key
```

(change this in `app/config.py` / the `API_KEY` env var before deploying)

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/classify-product` | image file (multipart) | category + confidence |
| POST | `/enroll-product` | `label` + ≥3 image files (multipart) | new/updated product type, enrolled instantly |
| POST | `/recognize-face` | image file (multipart) | customer id / "unknown" + visit logged |
| POST | `/enroll-face` | `person_name` + ≥3 image files (multipart) | new/updated person, enrolled instantly |
| POST | `/analyze-sentiment` | `{"text": "..."}` | sentiment label + confidence |
| POST | `/chatbot` | `{"message": "..."}` | bot reply + matched intent |
| GET  | `/dashboard/stats` | — | aggregate visit + sentiment counts (JSON) |
| GET  | `/health` | — | liveness check (no auth) |

## 5. Tests

```bash
pytest tests/ -v
```

## 6. Docker

```bash
docker build -t smart-retail-ai .
docker run -p 8000:8000 -e API_KEY=dev-secret-key smart-retail-ai
```

## 7. Online enrollment (add product types / people without retraining)

Both `/classify-product` and `/recognize-face` support adding new classes at
runtime, no notebook re-run or redeploy required:

```bash
# Add a brand-new product type from a handful of sample photos
curl -X POST http://localhost:8000/enroll-product \
  -H "x-api-key: dev-secret-key" \
  -F "label=umbrellas" \
  -F "files=@sample1.jpg" -F "files=@sample2.jpg" -F "files=@sample3.jpg"

# Enroll a new customer/person from a few clear, front-facing photos
curl -X POST http://localhost:8000/enroll-face \
  -H "x-api-key: dev-secret-key" \
  -F "person_name=alex" \
  -F "files=@face1.jpg" -F "files=@face2.jpg" -F "files=@face3.jpg"
```

How each works under the hood, and why they don't need a full retrain:

- **Faces** — the LBPH recognizer (`cv2.face.LBPHFaceRecognizer`) supports a
  native incremental `.update()` call. `enroll-face` detects the face in each
  submitted photo, and folds the new labeled samples straight into the
  existing `face_db.pkl` in place (a one-time `.train()` only happens the
  very first time, before any model exists on disk).
- **Products** — the trained softmax head is fixed-size (10 Fashion-MNIST
  classes) and *can't* grow new output classes without retraining, so new
  product types don't go through it. Instead, `enroll-product` embeds each
  sample photo with the same frozen MobileNetV2 backbone the classifier
  already uses, and stores a running average ("centroid") embedding per new
  class in `app/models/product_prototypes.pkl`. `/classify-product` then
  checks incoming photos against both the original softmax classes *and*
  these prototype centroids (nearest-centroid / cosine similarity) and
  returns whichever is the best match. Re-enrolling the same label just
  folds more photos into its centroid — accuracy improves with more samples,
  and existing classes are untouched.
- A few caveats worth knowing: prototype-class confidence scores are cosine
  similarities (temperature-scaled into 0–1), not calibrated probabilities
  like the softmax scores — treat them as good for *ranking*, not as exact
  percentages. Each enrollment call needs at least `config.MIN_ENROLL_IMAGES`
  (3 by default) usable photos, and product/person names can't collide with
  an existing label.

## 8. Ethics & privacy note (face recognition module)

This project uses face recognition for a "returning customer" loyalty demo.
By default the included notebook trains on the real **LFW (Labeled Faces in
the Wild)** dataset — real photos of real public figures, used strictly as a
practice/benchmark dataset (exactly the convention the original project spec
recommends). In a real deployment this raises issues worth writing up in the
report:

- **Consent**: customers must opt in before their face is enrolled. This now
  applies to `/enroll-face` too — it's the runtime on-ramp for adding new
  people, so it's the point where a real deployment needs to gate on explicit
  opt-in (and ideally its own audit log), not just to the initial notebook
  training set. Before using this for an actual retail demo, put real,
  consenting photos in `data/faces/<person_name>/*.jpg` — the notebook
  automatically prefers that real local data over LFW whenever it finds
  photos there.
- **Data minimization & retention**: store face *embeddings*, not raw images,
  and define a retention/deletion policy.
- **Bias**: face recognition accuracy is known to vary across demographics;
  any production system needs bias evaluation before deployment.
- **Security**: embeddings should be encrypted at rest; the API key auth in
  this repo is a minimal stand-in for real auth (OAuth2 / mTLS in production).

## 9. Stretch goals (already scaffolded to extend)

- Swap `sentiment_model.pkl` (TF-IDF + LogisticRegression) for a fine-tuned
  DistilBERT — the `nlp_service.py` interface is model-agnostic.
- Add a `/ws/recognize-face` WebSocket endpoint reusing `cv_service.py`.
- Log every prediction's confidence to `app/models/logs.jsonl` for drift
  monitoring (hook already present in `main.py`).
