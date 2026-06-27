import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model import detector
from app.schemas import ErrorResponse, HealthResponse, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Plant Disease YOLO Backend",
    description=(
        "YOLOv8 plant disease detection using the Roboflow "
        "plant-disease-ep6jy model."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_id=settings.model_id,
        model_loaded=detector.is_loaded,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        return detector.predict(image_bytes, filename=file.filename or "image.jpg")
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def root() -> dict:
    return {
        "service": "Plant Disease YOLO Backend",
        "model_id": settings.model_id,
        "inference_mode": settings.inference_mode,
        "roboflow": "https://universe.roboflow.com/yolo-v8-solrb/plant-disease-ep6jy/model/2",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict (multipart image upload)",
            "docs": "GET /docs",
        },
    }
