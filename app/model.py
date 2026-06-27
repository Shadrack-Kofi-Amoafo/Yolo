import logging
import time

import requests

from app.config import settings
from app.schemas import BoundingBox, Prediction, PredictResponse

logger = logging.getLogger(__name__)


class DiseaseDetectionModel:
    def __init__(self) -> None:
        self._model_id = settings.model_id
        self._session = requests.Session()
        logger.info(
            "Roboflow client ready (mode=%s, url=%s, model=%s)",
            settings.inference_mode,
            settings.api_url,
            self._model_id,
        )

    @property
    def is_loaded(self) -> bool:
        return True

    def predict(self, image_bytes: bytes, filename: str = "image.jpg") -> PredictResponse:
        start = time.perf_counter()

        url = f"{settings.api_url.rstrip('/')}/{self._model_id}"
        response = self._session.post(
            url,
            params={
                "api_key": settings.roboflow_api_key,
                "confidence": settings.confidence_threshold,
            },
            files={"file": (filename, image_bytes, "application/octet-stream")},
            timeout=60,
        )
        response.raise_for_status()
        results = response.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        predictions = _parse_predictions(results)
        top = max(predictions, key=lambda p: p.confidence) if predictions else None

        return PredictResponse(
            success=True,
            model_id=self._model_id,
            predictions=predictions,
            top_prediction=top,
            inference_ms=round(elapsed_ms, 2),
        )


def _parse_predictions(results: dict) -> list[Prediction]:
    raw_predictions = results.get("predictions", [])

    parsed: list[Prediction] = []
    for item in raw_predictions:
        disease = item.get("class") or item.get("class_name", "unknown")
        parsed.append(
            Prediction(
                disease=disease,
                confidence=float(item["confidence"]),
                class_id=item.get("class_id"),
                bbox=BoundingBox(
                    x=float(item["x"]),
                    y=float(item["y"]),
                    width=float(item["width"]),
                    height=float(item["height"]),
                ),
            )
        )

    parsed.sort(key=lambda p: p.confidence, reverse=True)
    return parsed


detector = DiseaseDetectionModel()
