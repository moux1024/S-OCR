from pydantic import BaseModel


class OCRResult(BaseModel):
    text: str
    raw_text: str
    score: float
    bbox: dict | None = None
    center: dict | None = None
    region: str
    source: str
    scene: str | None = None


class OCRResponse(BaseModel):
    results: list[OCRResult]


class HealthResponse(BaseModel):
    status: str
    backend: str | None
    model_loaded: bool
