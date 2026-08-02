"""
Pydantic models used for request validation and response serialization
across all routers. Keeping them in one place means the OpenAPI/Swagger
docs at /docs stay consistent and self-documenting.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------- Vision ----------

class ProductClassificationResponse(BaseModel):
    category: str = Field(..., description="Predicted product category")
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: dict = Field(default_factory=dict, description="Per-class confidence scores")


class FaceRecognitionResponse(BaseModel):
    recognized: bool
    customer_id: Optional[str] = Field(None, description="Matched customer label, null if unknown")
    confidence: float = Field(..., description="LBPH distance score; lower = more confident")
    visit_logged: bool
    visit_count: int = Field(0, description="Total visits on record for this customer")


class FaceEnrollResponse(BaseModel):
    person_name: str
    label_id: int
    images_used: int = Field(..., description="Number of submitted photos with a detectable face")
    total_people: int = Field(..., description="Total distinct people now enrolled")


class ProductEnrollResponse(BaseModel):
    label: str
    images_used: int = Field(..., description="Number of photos used in this enrollment call")
    total_images_for_label: int = Field(..., description="Cumulative photos averaged into this class's centroid")
    total_classes: int = Field(..., description="Built-in classes + online-enrolled classes")


# ---------- NLP ----------

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["The product quality was amazing and delivery was fast!"])


class SentimentResponse(BaseModel):
    label: str = Field(..., description="Positive / Negative (binary, per the Amazon reviews training data)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    cleaned_text: str


# ---------- Chatbot ----------

class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["What is your return policy?"])


class ChatbotResponse(BaseModel):
    reply: str
    matched_intent: str
    method: str = Field(..., description="'rule' or 'ml_fallback'")
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------- Dashboard ----------

class DashboardStats(BaseModel):
    total_visits: int
    unique_customers: int
    sentiment_breakdown: dict
    total_chatbot_queries: int
    top_intents: List[str]
