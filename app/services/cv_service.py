"""
Module A2 (Image Classification) + A3 (Face Recognition).

- ProductClassifier wraps a Keras MobileNetV2 transfer-learning model
  trained by notebooks/01_image_classifier_training.ipynb, PLUS an online,
  incrementally-updatable nearest-centroid "prototype" classifier built on
  the same frozen MobileNetV2 backbone. New product types can be enrolled
  at runtime (a few sample photos in, a new class out) without retraining
  or touching the original softmax head.
- FaceRecognitionService wraps an OpenCV LBPH recognizer trained by
  notebooks/02_face_recognition_setup.ipynb, and logs each recognized
  visit to a JSONL file for the /dashboard/stats endpoint. LBPH supports
  incremental `.update()`, so new people can be enrolled at runtime too.

Both classes lazy-load their model files so the API can boot even before
training has been run (with a clear error raised only when the endpoint
is actually called).

NOTE on the two different "confidence" spaces for products: scores for the
original 10 softmax classes are true probabilities from the trained head.
Scores for prototype (online-enrolled) classes are cosine similarities to a
class centroid, temperature-scaled into a 0-1 range so they sort sensibly
together in `all_scores`. They are comparable for *ranking* purposes but are
not calibrated probabilities the way the softmax scores are — don't treat
e.g. 0.62 on each as meaning literally the same thing statistically.
"""
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app import config
from app.services import cv_utils


class ModelNotTrainedError(RuntimeError):
    pass


class EnrollmentError(ValueError):
    pass


# ---------------------------------------------------------------------------
# A2 — Product image classifier (MobileNetV2 transfer learning)
# ---------------------------------------------------------------------------

class ProductClassifier:
    def __init__(self):
        self._model = None
        self._embedder = None          # sub-model: image -> 1280-d MobileNetV2 embedding
        self._labels = None            # softmax head classes, {"0": "T-shirt/ltop", ...}
        self._prototypes: Dict[str, np.ndarray] = {}   # online classes: label -> centroid (1280-d, L2-normalized)
        self._prototype_counts: Dict[str, int] = {}    # label -> number of images averaged into the centroid

    def _load(self):
        if self._model is not None:
            return
        if not config.PRODUCT_CLASSIFIER_PATH.exists():
            raise ModelNotTrainedError(
                "product_classifier.h5 not found. Run "
                "notebooks/01_image_classifier_training.ipynb first."
            )
        # Imported lazily: tensorflow is heavy and only needed for this feature.
        import tensorflow as tf

        self._model = tf.keras.models.load_model(str(config.PRODUCT_CLASSIFIER_PATH))
        with open(config.CLASS_LABELS_PATH) as f:
            self._labels = json.load(f)  # {"0": "shoes", "1": "bags", ...}

        # Sub-model that stops right after GlobalAveragePooling2D (the frozen
        # MobileNetV2 backbone's output) — this is the embedding we use for
        # online (prototype) classes, since it never needs retraining. Built by
        # replaying the first two layers on a fresh Input rather than reusing
        # `self._model.input`, since a loaded Sequential .h5 model doesn't
        # reliably expose that attribute in Keras 3.
        inputs = tf.keras.Input(shape=(*config.IMAGE_SIZE, 3))
        x = self._model.layers[0](inputs)
        x = self._model.layers[1](x)
        self._embedder = tf.keras.Model(inputs=inputs, outputs=x)

        self._load_prototypes()

    def _load_prototypes(self):
        self._prototypes = {}
        self._prototype_counts = {}
        if config.PRODUCT_PROTOTYPES_PATH.exists():
            with open(config.PRODUCT_PROTOTYPES_PATH, "rb") as f:
                store = pickle.load(f)
            self._prototypes = store.get("centroids", {})
            self._prototype_counts = store.get("counts", {})

    def _save_prototypes(self):
        config.PRODUCT_PROTOTYPES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.PRODUCT_PROTOTYPES_PATH, "wb") as f:
            pickle.dump({"centroids": self._prototypes, "counts": self._prototype_counts}, f)

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = cv_utils.bytes_to_image(image_bytes)
        img = cv_utils.resize(img, config.IMAGE_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv_utils.normalize(img)

    def _embed(self, image_bytes: bytes) -> np.ndarray:
        """Return an L2-normalized 1280-d embedding for one image."""
        arr = self._preprocess(image_bytes)
        batch = np.expand_dims(arr, axis=0)
        emb = self._embedder.predict(batch, verbose=0)[0]
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def predict(self, image_bytes: bytes) -> Tuple[str, float, Dict[str, float]]:
        self._load()
        arr = self._preprocess(image_bytes)
        batch = np.expand_dims(arr, axis=0)

        preds = self._model.predict(batch, verbose=0)[0]  # shape (num_classes,)
        all_scores = {self._labels[str(i)]: float(p) for i, p in enumerate(preds)}

        if self._prototypes:
            embedding = self._embedder.predict(batch, verbose=0)[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            sims = {
                label: float(np.dot(embedding, centroid))  # cosine similarity, both unit-norm
                for label, centroid in self._prototypes.items()
            }
            # Temperature-scaled softmax over the prototype similarities so they
            # land in a comparable 0-1 range to the softmax scores above.
            temperature = 10.0
            exp_scores = {label: np.exp(sim * temperature) for label, sim in sims.items()}
            total = sum(exp_scores.values())
            for label, e in exp_scores.items():
                all_scores[label] = float(e / total)

        best_label = max(all_scores, key=all_scores.get)
        best_conf = all_scores[best_label]
        return best_label, best_conf, all_scores

    def enroll(self, label: str, image_bytes_list: List[bytes]) -> Dict:
        """
        Add (or extend) a product class on the fly — no retraining required.

        Embeds each sample photo with the frozen MobileNetV2 backbone and
        folds them into a running centroid for `label`. Existing samples
        aren't kept around (only their contribution to the mean), so this
        stays cheap regardless of how many times a class gets re-enrolled.
        """
        label = label.strip()
        if not label:
            raise EnrollmentError("label must not be empty")
        if len(image_bytes_list) < config.MIN_ENROLL_IMAGES:
            raise EnrollmentError(
                f"Need at least {config.MIN_ENROLL_IMAGES} sample photos to enroll a class "
                f"reliably (got {len(image_bytes_list)})."
            )
        if label in (self._labels or {}).values():
            raise EnrollmentError(
                f"'{label}' is already one of the built-in classifier classes; "
                "choose a different name for the new product type."
            )

        self._load()
        embeddings = [self._embed(b) for b in image_bytes_list]
        new_sum = np.sum(embeddings, axis=0)
        new_count = len(embeddings)

        if label in self._prototypes:
            prev_count = self._prototype_counts.get(label, 0)
            prev_sum = self._prototypes[label] * prev_count  # undo the previous normalization
            combined = prev_sum + new_sum
            total_count = prev_count + new_count
        else:
            combined = new_sum
            total_count = new_count

        centroid = combined / total_count
        norm = np.linalg.norm(centroid)
        centroid = centroid / norm if norm > 0 else centroid

        self._prototypes[label] = centroid
        self._prototype_counts[label] = total_count
        self._save_prototypes()

        return {
            "label": label,
            "images_used": new_count,
            "total_images_for_label": total_count,
            "total_classes": len(self._labels) + len(self._prototypes),
        }


# ---------------------------------------------------------------------------
# A3 — Face recognition & returning-customer visit logging
# ---------------------------------------------------------------------------

class FaceRecognitionService:
    def __init__(self):
        self._recognizer = None
        self._label_map = None  # {int_id: "customer_name"}

    def _load(self):
        if self._recognizer is not None:
            return
        if not config.FACE_DB_PATH.exists():
            raise ModelNotTrainedError(
                "face_db.pkl not found. Run "
                "notebooks/02_face_recognition_setup.ipynb first."
            )
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._recognizer.read(str(config.FACE_DB_PATH))
        with open(config.FACE_LABEL_MAP_PATH) as f:
            raw_map = json.load(f)
        self._label_map = {int(k): v for k, v in raw_map.items()}

    def recognize(self, image_bytes: bytes) -> Dict:
        self._load()
        img = cv_utils.bytes_to_image(image_bytes)
        boxes = cv_utils.detect_faces(img)

        if not boxes:
            return {
                "recognized": False,
                "customer_id": None,
                "confidence": 0.0,
                "visit_logged": False,
                "visit_count": 0,
            }

        # Use the largest detected face (closest to camera).
        box = max(boxes, key=lambda b: b[2] * b[3])
        face = cv_utils.crop_face(img, box, padding=0.0)
        gray = cv_utils.to_grayscale(face)
        gray = cv_utils.resize(gray, (200, 200))

        label_id, distance = self._recognizer.predict(gray)
        # LBPH: LOWER distance == better match.
        recognized = distance <= config.FACE_RECOGNITION_CONFIDENCE_THRESHOLD
        customer_id = self._label_map.get(label_id, "unknown") if recognized else None

        visit_count = 0
        visit_logged = False
        if recognized:
            visit_count = self._log_visit(customer_id)
            visit_logged = True

        return {
            "recognized": recognized,
            "customer_id": customer_id,
            "confidence": float(distance),
            "visit_logged": visit_logged,
            "visit_count": visit_count,
        }

    def enroll(self, person_name: str, image_bytes_list: List[bytes]) -> Dict:
        """
        Add a new person on the fly using LBPH's native incremental update.

        Unlike the product classifier's softmax head, LBPH has no fixed output
        size — `recognizer.update()` folds new labeled faces into the existing
        model in place, so this never needs a full retrain from
        notebooks/02_face_recognition_setup.ipynb.
        """
        person_name = person_name.strip()
        if not person_name:
            raise EnrollmentError("person_name must not be empty")
        if len(image_bytes_list) < config.MIN_ENROLL_IMAGES:
            raise EnrollmentError(
                f"Need at least {config.MIN_ENROLL_IMAGES} sample photos to enroll a person "
                f"reliably (got {len(image_bytes_list)})."
            )

        faces = []
        for image_bytes in image_bytes_list:
            img = cv_utils.bytes_to_image(image_bytes)
            boxes = cv_utils.detect_faces(img)
            if not boxes:
                continue  # skip photos where no face was detected
            box = max(boxes, key=lambda b: b[2] * b[3])
            face = cv_utils.crop_face(img, box, padding=0.0)
            gray = cv_utils.to_grayscale(face)
            gray = cv_utils.resize(gray, config.FACE_ENROLL_SIZE)
            faces.append(gray)

        if len(faces) < config.MIN_ENROLL_IMAGES:
            raise EnrollmentError(
                f"Only detected a usable face in {len(faces)} of the submitted photos; "
                f"need at least {config.MIN_ENROLL_IMAGES}. Try clearer, front-facing shots."
            )

        # A trained model already on disk means LBPH's .update() is safe to use;
        # otherwise this is the very first enrollment and needs a one-time .train().
        model_already_trained = config.FACE_DB_PATH.exists()

        # Load any existing recognizer/label_map so we extend rather than overwrite.
        try:
            self._load()
        except ModelNotTrainedError:
            self._recognizer = cv2.face.LBPHFaceRecognizer_create()
            self._label_map = {}

        existing_id = next(
            (lid for lid, name in self._label_map.items() if name == person_name), None
        )
        if existing_id is not None:
            label_id = existing_id
        else:
            label_id = max(self._label_map.keys(), default=-1) + 1
            self._label_map[label_id] = person_name

        labels = np.array([label_id] * len(faces))

        if model_already_trained:
            self._recognizer.update(faces, labels)  # incremental — no full retrain
        else:
            self._recognizer.train(faces, labels)   # first-ever enrollment

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._recognizer.write(str(config.FACE_DB_PATH))
        with open(config.FACE_LABEL_MAP_PATH, "w") as f:
            json.dump({str(k): v for k, v in self._label_map.items()}, f, indent=2)

        return {
            "person_name": person_name,
            "label_id": label_id,
            "images_used": len(faces),
            "total_people": len(self._label_map),
        }

    @staticmethod
    def _log_visit(customer_id: str) -> int:
        """Append a visit event and return the customer's running visit count."""
        entry = {"customer_id": customer_id, "timestamp": time.time()}
        with open(config.VISITS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

        count = 0
        if config.VISITS_LOG_PATH.exists():
            with open(config.VISITS_LOG_PATH) as f:
                for line in f:
                    try:
                        if json.loads(line)["customer_id"] == customer_id:
                            count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        return count


# Module-level singletons — models load once at process startup, not per-request.
product_classifier = ProductClassifier()
face_recognition_service = FaceRecognitionService()
