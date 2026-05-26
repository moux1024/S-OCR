import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from .ocr_engine import OCRReader
from .schemas import HealthResponse, OCRResponse

logger = logging.getLogger(__name__)

_reader: OCRReader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _reader
    _reader = OCRReader()
    logger.info("OCR engine initialized (backend=%s)", _reader._backend)
    yield
    _reader = None


app = FastAPI(title="OCR Service", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        backend=_reader._backend if _reader else None,
        model_loaded=_reader.available if _reader else False,
    )


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    image: UploadFile = File(...),
    region_name: str = Form("screen"),
    offset_x: int = Form(0),
    offset_y: int = Form(0),
    source: str = Form("ocr"),
    scene: str | None = Form(None),
):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    try:
        results = _reader.read_image(
            img,
            region_name=region_name,
            offset=(offset_x, offset_y),
            source=source,
            scene=scene,
        )
    finally:
        img.close()
    return OCRResponse(results=results)
