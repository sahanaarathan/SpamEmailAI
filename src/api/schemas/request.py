from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The email/message text to classify",
        examples=["Congratulations! You've won a free iPhone. Click here to claim."],
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message must not be blank or whitespace only.")
        return v


class BatchPredictRequest(BaseModel):
    messages: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of messages to classify (max 100 per request)",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages: list[str]) -> list[str]:
        cleaned = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, str) or not msg.strip():
                raise ValueError(f"Message at index {i} is empty or invalid.")
            if len(msg) > 5000:
                raise ValueError(f"Message at index {i} exceeds 5000 characters.")
            cleaned.append(msg)
        return cleaned


class PredictResponse(BaseModel):
    label: str
    is_spam: bool
    spam_probability: float
    ham_probability: float
    top_spam_words: list[str]
    top_ham_words: list[str]
    model_name: str


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total: int
    spam_count: int
    ham_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


class ErrorResponse(BaseModel):
    detail: str
