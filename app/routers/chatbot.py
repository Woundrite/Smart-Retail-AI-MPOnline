import json
from collections import Counter

from fastapi import APIRouter, Depends

from app import config
from app.auth import verify_api_key
from app.schemas import ChatbotRequest, ChatbotResponse, DashboardStats
from app.services.chatbot_service import chatbot_service

router = APIRouter(tags=["Chatbot & Dashboard"])


@router.post("/chatbot", response_model=ChatbotResponse, dependencies=[Depends(verify_api_key)])
async def chatbot_reply(payload: ChatbotRequest):
    result = chatbot_service.get_reply(payload.message)
    return ChatbotResponse(**result)


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


@router.get("/dashboard/stats", response_model=DashboardStats, dependencies=[Depends(verify_api_key)])
async def dashboard_stats():
    """
    Aggregate visit + sentiment + chatbot stats for a simple frontend chart
    (Streamlit / React dashboard consuming this JSON).
    """
    visits = _read_jsonl(config.VISITS_LOG_PATH)
    chatbot_logs = _read_jsonl(config.MODELS_DIR / "chatbot_logs.jsonl")

    unique_customers = len({v["customer_id"] for v in visits if "customer_id" in v})

    # Sentiment breakdown is derived from a sentiment_logs.jsonl if present;
    # this endpoint stays useful even before any sentiment calls are logged.
    sentiment_logs = _read_jsonl(config.MODELS_DIR / "sentiment_logs.jsonl")
    sentiment_breakdown = dict(Counter(s.get("label", "unknown") for s in sentiment_logs))

    intent_counts = Counter(c.get("intent", "unknown") for c in chatbot_logs)
    top_intents = [tag for tag, _ in intent_counts.most_common(5)]

    return DashboardStats(
        total_visits=len(visits),
        unique_customers=unique_customers,
        sentiment_breakdown=sentiment_breakdown,
        total_chatbot_queries=len(chatbot_logs),
        top_intents=top_intents,
    )
