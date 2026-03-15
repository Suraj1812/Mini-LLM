from typing import Optional

from pydantic import BaseModel, Field

from config import DEFAULT_EPOCHS, DEFAULT_GENERATION_LENGTH


class TrainRequest(BaseModel):
    text: Optional[str] = None
    epochs: int = Field(default=DEFAULT_EPOCHS, ge=1)


class ActionResponse(BaseModel):
    message: str
    status: str


class GenerateRequest(BaseModel):
    prompt: str
    length: int = Field(default=DEFAULT_GENERATION_LENGTH, ge=1)


class GenerateResponse(BaseModel):
    prompt: str
    output: str
    length: int


class StatusResponse(BaseModel):
    status: str
    message: str
    last_error: Optional[str] = None
    last_loss: Optional[float] = None
    trained_at: Optional[str] = None
    model_ready: bool
    has_training_data: bool


class HealthResponse(BaseModel):
    ok: bool
    status: str
    model_ready: bool
