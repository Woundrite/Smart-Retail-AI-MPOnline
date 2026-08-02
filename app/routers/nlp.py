import json
import time

from fastapi import APIRouter, Depends, HTTPException

from app import config
from app.auth import verify_api_key
from app.schemas import SentimentRequest, SentimentResponse
from app.services.nlp_service import ModelNotTrainedError, sentiment_analyzer

router = APIRouter(tags=["NLP"])


def _log_sentiment(label: str, confidence: float):
    """Append to sentiment_logs.jsonl so /dashboard/stats can aggregate it."""
    entry = {"label": label, "confidence": confidence, "timestamp": time.time()}
    with open(config.MODELS_DIR / "sentiment_logs.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


@router.post("/analyze-sentiment", response_model=SentimentResponse, dependencies=[Depends(verify_api_key)])
async def analyze_sentiment(payload: SentimentRequest):
    try:
        label, confidence, cleaned = sentiment_analyzer.predict(payload.text)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    _log_sentiment(label, confidence)
    return SentimentResponse(label=label, confidence=confidence, cleaned_text=cleaned)
