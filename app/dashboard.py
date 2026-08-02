"""
Gradio dashboard (mounted at "/") — a human-friendly front end for every
endpoint exposed by app/main.py.

This module never imports the services directly. It talks to the API the
same way Postman/curl/a real client would: real HTTP requests, with the
`x-api-key` header attached. That keeps the dashboard honest as a live demo
of the actual API surface (auth included) rather than a shortcut around it.

Configure via environment variables:
  API_KEY       - must match the API_KEY the FastAPI app was started with
                   (defaults to "SecretPassword" to match the project setup)
  API_BASE_URL  - base URL of the FastAPI app itself (defaults to
                   "http://127.0.0.1:8000" since the dashboard is mounted
                   into the same process/port that serves the API)
"""
import io
import os

import gradio as gr
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "SecretPassword")
TIMEOUT = 30


def _headers():
    return {"x-api-key": API_KEY}


def _error_detail(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


def _call(method: str, path: str, **kwargs):
    """Shared request wrapper. Returns (ok, data_or_error_string)."""
    try:
        resp = requests.request(
            method, f"{API_BASE_URL}{path}", headers=_headers(), timeout=TIMEOUT, **kwargs
        )
    except requests.exceptions.RequestException as e:
        return False, f"Could not reach the API at {API_BASE_URL}{path} — {e}"

    if resp.status_code == 401:
        return False, "401 Unauthorized — the x-api-key sent by the dashboard doesn't match API_KEY on the server."
    if not resp.ok:
        return False, f"{resp.status_code} error — {_error_detail(resp)}"
    return True, resp.json()


# --------------------------------------------------------------------------- #
# Computer Vision
# --------------------------------------------------------------------------- #

def classify_product(image):
    if image is None:
        return "Upload a product image first.", None

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    buf.seek(0)

    ok, data = _call(
        "POST", "/classify-product", files={"file": ("upload.jpg", buf, "image/jpeg")}
    )
    if not ok:
        return f"❌ {data}", None

    summary = f"**Predicted category:** {data['category']}  \n**Confidence:** {data['confidence']:.1%}"
    return summary, data.get("all_scores", {})


def recognize_face(image):
    if image is None:
        return "Upload or capture a photo first.", None

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    buf.seek(0)

    ok, data = _call(
        "POST", "/recognize-face", files={"file": ("upload.jpg", buf, "image/jpeg")}
    )
    if not ok:
        return f"❌ {data}", None

    if data["recognized"]:
        summary = (
            f"**Welcome back, {data['customer_id']}!**  \n"
            f"Visit #{data['visit_count']} logged.  \n"
            f"LBPH distance: {data['confidence']:.1f} (lower = more confident)"
        )
    else:
        summary = (
            "**Unknown / no confident match.**  \n"
            f"LBPH distance: {data['confidence']:.1f} — no visit logged."
        )
    return summary, data


def _files_to_multipart(filepaths):
    """Turn a gr.File(file_count='multiple') list of local paths into the
    list-of-tuples requests expects for a repeated multipart field."""
    parts = []
    for i, fp in enumerate(filepaths or []):
        with open(fp, "rb") as f:
            parts.append(("files", (f"sample_{i}.jpg", f.read(), "image/jpeg")))
    return parts


def enroll_product(label, files):
    if not label or not label.strip():
        return "Enter a name for the new product type first."
    if not files or len(files) < 3:
        return f"Upload at least 3 sample photos (got {len(files or [])})."

    ok, data = _call(
        "POST", "/enroll-product",
        data={"label": label.strip()}, files=_files_to_multipart(files),
    )
    if not ok:
        return f"❌ {data}"

    return (
        f"✅ **{data['label']}** enrolled — no retraining needed.  \n"
        f"Photos used this call: {data['images_used']}  \n"
        f"Total photos averaged into this class: {data['total_images_for_label']}  \n"
        f"Classifier now recognizes **{data['total_classes']}** classes total "
        "(try it in the Classify Product tab)."
    )


def enroll_face(person_name, files):
    if not person_name or not person_name.strip():
        return "Enter a person's name first."
    if not files or len(files) < 3:
        return f"Upload at least 3 clear, front-facing photos (got {len(files or [])})."

    ok, data = _call(
        "POST", "/enroll-face",
        data={"person_name": person_name.strip()}, files=_files_to_multipart(files),
    )
    if not ok:
        return f"❌ {data}"

    return (
        f"✅ **{data['person_name']}** enrolled — no retraining needed.  \n"
        f"Photos with a detected face: {data['images_used']}  \n"
        f"Total people now enrolled: {data['total_people']}  \n"
        "(try recognizing them in the Recognize Face tab)."
    )


# --------------------------------------------------------------------------- #
# NLP
# --------------------------------------------------------------------------- #

def analyze_sentiment(text):
    if not text or not text.strip():
        return "Enter some review or chat text first.", None

    ok, data = _call("POST", "/analyze-sentiment", json={"text": text})
    if not ok:
        return f"❌ {data}", None

    summary = f"**{data['label']}**  (confidence: {data['confidence']:.1%})"
    return summary, {"label": data["label"], "confidence": data["confidence"], "cleaned_text": data["cleaned_text"]}


# --------------------------------------------------------------------------- #
# Chatbot
# --------------------------------------------------------------------------- #

def chatbot_reply(message, history):
    history = history or []
    if not message or not message.strip():
        return history, ""

    ok, data = _call("POST", "/chatbot", json={"message": message})
    if not ok:
        history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"❌ {data}"}]
        return history, ""

    reply = f"{data['reply']}\n\n_intent: {data['matched_intent']} · method: {data['method']} · confidence: {data['confidence']:.0%}_"
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return history, ""


# --------------------------------------------------------------------------- #
# Dashboard stats
# --------------------------------------------------------------------------- #

def refresh_stats():
    ok, data = _call("GET", "/dashboard/stats")
    if not ok:
        return f"❌ {data}", None, None, None

    summary = (
        f"**Total visits:** {data['total_visits']}  \n"
        f"**Unique customers:** {data['unique_customers']}  \n"
        f"**Total chatbot queries:** {data['total_chatbot_queries']}"
    )
    sentiment_breakdown = data.get("sentiment_breakdown", {}) or {"no data yet": 0}
    top_intents = data.get("top_intents", [])
    intents_display = "\n".join(f"{i + 1}. {tag}" for i, tag in enumerate(top_intents)) or "No chatbot queries logged yet."
    return summary, sentiment_breakdown, intents_display, data


def check_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        return f"✅ API is reachable at {API_BASE_URL} — {resp.json()}"
    except requests.exceptions.RequestException as e:
        return f"❌ API unreachable at {API_BASE_URL} — {e}"


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def build_dashboard() -> gr.Blocks:
    with gr.Blocks(title="Smart Retail & Customer Intelligence") as demo:
        gr.Markdown(
            "# 🛍️ Smart Retail & Customer Intelligence Platform\n"
            f"Live dashboard for the FastAPI service at `{API_BASE_URL}`. "
            "Every action below is a real authenticated call to the API "
            "(`x-api-key` header set from the `API_KEY` env var). "
            "Raw Swagger docs are at [/docs](/docs)."
        )

        with gr.Row():
            health_btn = gr.Button("Check API health", size="sm")
            health_out = gr.Markdown()
        health_btn.click(check_health, outputs=health_out)

        with gr.Tabs():
            with gr.Tab("🖼️ Classify Product"):
                gr.Markdown("Upload a product photo → `POST /classify-product`")
                with gr.Row():
                    cp_image = gr.Image(type="pil", label="Product image")
                    with gr.Column():
                        cp_summary = gr.Markdown()
                        cp_scores = gr.Label(label="All category scores", num_top_classes=10)
                cp_btn = gr.Button("Classify", variant="primary")
                cp_btn.click(classify_product, inputs=cp_image, outputs=[cp_summary, cp_scores])

            with gr.Tab("➕ Enroll Product"):
                gr.Markdown(
                    "Teach the classifier a **new product type on the fly** → "
                    "`POST /enroll-product`. No retraining or redeploy — a few sample "
                    "photos are embedded with the same MobileNetV2 backbone and stored "
                    "as a new class centroid. Re-enrolling the same name adds more "
                    "photos to that class instead of replacing it."
                )
                ep_label = gr.Textbox(label="New product type name", placeholder="e.g. umbrellas")
                ep_files = gr.File(
                    label="Sample photos (3 or more)", file_count="multiple",
                    file_types=["image"], type="filepath",
                )
                ep_out = gr.Markdown()
                ep_btn = gr.Button("Enroll product type", variant="primary")
                ep_btn.click(enroll_product, inputs=[ep_label, ep_files], outputs=ep_out)

            with gr.Tab("🙂 Recognize Face"):
                gr.Markdown(
                    "Upload/capture a face photo → `POST /recognize-face` "
                    "(logs a visit if the customer is recognized)"
                )
                with gr.Row():
                    fr_image = gr.Image(type="pil", label="Customer photo", sources=["upload", "webcam"])
                    with gr.Column():
                        fr_summary = gr.Markdown()
                        fr_json = gr.JSON(label="Raw response")
                fr_btn = gr.Button("Recognize", variant="primary")
                fr_btn.click(recognize_face, inputs=fr_image, outputs=[fr_summary, fr_json])

            with gr.Tab("➕ Enroll Face"):
                gr.Markdown(
                    "Add a **new person on the fly** → `POST /enroll-face`. LBPH's "
                    "incremental update folds the new photos straight into the "
                    "existing face model — no full retrain. Re-enrolling an existing "
                    "name adds more photos for that same person instead of creating "
                    "a duplicate.\n\n"
                    "⚠️ Only enroll people who have given consent to have their face recognized."
                )
                ef_name = gr.Textbox(label="Person's name / ID", placeholder="e.g. alex")
                ef_files = gr.File(
                    label="Clear, front-facing photos (3 or more)", file_count="multiple",
                    file_types=["image"], type="filepath",
                )
                ef_out = gr.Markdown()
                ef_btn = gr.Button("Enroll person", variant="primary")
                ef_btn.click(enroll_face, inputs=[ef_name, ef_files], outputs=ef_out)


            with gr.Tab("💬 Analyze Sentiment"):
                gr.Markdown("Paste review/chat text → `POST /analyze-sentiment`")
                sa_text = gr.Textbox(
                    label="Text", lines=4,
                    placeholder="The product quality was amazing and delivery was fast!",
                )
                sa_summary = gr.Markdown()
                sa_json = gr.JSON(label="Raw response")
                sa_btn = gr.Button("Analyze", variant="primary")
                sa_btn.click(analyze_sentiment, inputs=sa_text, outputs=[sa_summary, sa_json])

            with gr.Tab("🤖 Chatbot"):
                gr.Markdown("Chat with the FAQ bot → `POST /chatbot`")
                cb_chat = gr.Chatbot(type="messages", height=350)
                cb_input = gr.Textbox(label="Message", placeholder="What is your return policy?")
                cb_btn = gr.Button("Send", variant="primary")
                cb_btn.click(chatbot_reply, inputs=[cb_input, cb_chat], outputs=[cb_chat, cb_input])
                cb_input.submit(chatbot_reply, inputs=[cb_input, cb_chat], outputs=[cb_chat, cb_input])

            with gr.Tab("📊 Dashboard Stats"):
                gr.Markdown("Aggregate visit / sentiment / chatbot stats → `GET /dashboard/stats`")
                ds_btn = gr.Button("Refresh stats", variant="primary")
                ds_summary = gr.Markdown()
                with gr.Row():
                    ds_sentiment = gr.Label(label="Sentiment breakdown")
                    ds_intents = gr.Markdown(label="Top intents")
                ds_raw = gr.JSON(label="Raw response")
                ds_btn.click(refresh_stats, outputs=[ds_summary, ds_sentiment, ds_intents, ds_raw])
                demo.load(refresh_stats, outputs=[ds_summary, ds_sentiment, ds_intents, ds_raw])

    return demo