from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Prediction(BaseModel):
    disease: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    class_id: int | None = None


class PredictResponse(BaseModel):
    success: bool
    model_id: str
    predictions: list[Prediction]
    top_prediction: Prediction | None = None
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    success: bool = False
    detail: str
