from typing import Literal
from pydantic import BaseModel, Field, field_validator

class TicketRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)

    @field_validator("text")
    @classmethod
    def meaningful(cls, value):
        value = value.strip()
        if not value: raise ValueError("Enter a customer message.")
        return value

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    ticket_context: str = Field(default="", max_length=6000)
    history: list[Turn] = Field(default_factory=list, max_length=6)

    @field_validator("message")
    @classmethod
    def meaningful(cls, value):
        value = value.strip()
        if not value: raise ValueError("Enter a policy question.")
        return value

class Source(BaseModel):
    id: str
    document: str
    title: str
    page: int
    section: str
    text: str
    score: float

class AnalysisResponse(BaseModel):
    intent: str
    confidence: float
    top_intents: list[dict]
    sentiment: str
    sentiment_scores: dict[str, float]
    summary: str
    model_ids: dict[str, str]
    latency_ms: dict[str, float]

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    cited_source_ids: list[str]
    abstained: bool
    reason: str | None = None
    latency_ms: float
    evidence: list[dict[str, str]] = Field(default_factory=list)

