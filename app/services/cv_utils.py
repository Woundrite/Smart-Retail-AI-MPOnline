"""
Module A1 — OpenCV Basics.

Reusable, dependency-light image preprocessing functions used by both the
product classifier and the face recognition pipeline. Kept separate from
cv_service.py so these primitives can be unit-tested and reused on their
own (e.g. from a notebook or a webcam demo script).
"""
from typing import List, Tuple

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """Read an image from disk as a BGR numpy array (OpenCV's native format)."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at path: {path}")
    return img


def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw uploaded bytes (from FastAPI's UploadFile) into a BGR image."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes — is this a valid image file?")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """size = (width, height)"""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def gaussian_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # kernel size must be odd
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def canny_edges(img: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    gray = to_grayscale(img) if len(img.shape) == 3 else img
    return cv2.Canny(gray, low, high)


def normalize(img: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] float32 — standard pre-CNN normalization."""
    return img.astype(np.float32) / 255.0


_face_cascade = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Lazily load OpenCV's bundled Haar cascade (avoids import-time file IO)."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return _face_cascade


def detect_faces(img: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces with the classic Viola-Jones Haar cascade detector.
    Returns a list of (x, y, w, h) bounding boxes in the original image.
    """
    gray = to_grayscale(img) if len(img.shape) == 3 else img
    faces = _get_face_cascade().detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(60, 60)
    )
    return [tuple(map(int, f)) for f in faces]


def crop_face(img: np.ndarray, box: Tuple[int, int, int, int], padding: float = 0.15) -> np.ndarray:
    """Crop a detected face with a small margin of padding on each side."""
    x, y, w, h = box
    pad_w, pad_h = int(w * padding), int(h * padding)
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad_w), max(0, y - pad_h)
    x1, y1 = min(W, x + w + pad_w), min(H, y + h + pad_h)
    return img[y0:y1, x0:x1]


def draw_boxes(img: np.ndarray, boxes: List[Tuple[int, int, int, int]], label: str = "") -> np.ndarray:
    """Draw bounding boxes on a copy of the image — useful for demo/debug output."""
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if label:
            cv2.putText(out, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out
