"""
Minimal API-key auth dependency — simulates production security per the
project spec ("Add basic auth or API key header for the endpoints").
Swap for OAuth2 / JWT in a real deployment.
"""
from fastapi import Header, HTTPException, status

from app.config import API_KEY


async def verify_api_key(x_api_key: str = Header(..., description="Required API key")) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
