import logging
import unicodedata

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OCRReader:
    """OCR wrapper using PaddleOCR (GPU) with RapidOCR (CPU) fallback."""

    def __init__(self, enabled: bool = True, model_profile: str = "mobile"):
        self.enabled = enabled
        self.model_profile = model_profile
        self._reader = None
        self._backend = None
        self.available = False
        if not enabled:
            return
        try:
            from paddleocr import PaddleOCR

            self._reader = self._create_paddle_reader(PaddleOCR)
            self._backend = "paddle-gpu"
            self.available = True
            logger.info("OCR backend: PaddleOCR (GPU)")
        except (ImportError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("PaddleOCR GPU unavailable: %s, falling back to RapidOCR", exc)
        if not self.available:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._reader = RapidOCR()
                self._backend = "rapid"
                self.available = True
                logger.info("OCR backend: rapidocr-onnxruntime (CPU)")
            except ImportError:
                pass

    def read_image(
        self,
        image: Image.Image,
        region_name: str,
        offset: tuple[int, int] = (0, 0),
        source: str = "ocr",
        scene: str | None = None,
    ) -> list[dict]:
        if not self.available or self._reader is None:
            return []
        try:
            rgb = image.convert("RGB")
            arr = np.array(rgb)
            rgb.close()
            raw = self._run_ocr(arr)
            del arr
        except (AssertionError, ValueError, RuntimeError) as exc:
            logger.warning("OCR failed for region %s, skipping: %s", region_name, exc)
            return []
        results: list[dict] = []
        for item in _iter_ocr_items(raw, offset):
            item["region"] = region_name
            item["source"] = source
            if scene:
                item["scene"] = scene
            results.append(item)
        return results

    def _run_ocr(self, image_array):
        if self._backend == "rapid":
            result, _ = self._reader(image_array)
            return _rapid_to_items(result)
        if self._backend == "paddle-gpu":
            if hasattr(self._reader, "predict"):
                return self._reader.predict(
                    image_array,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            return self._reader.ocr(image_array, cls=False)
        if hasattr(self._reader, "predict"):
            return self._reader.predict(
                image_array,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._reader.ocr(image_array, cls=False)

    def _create_paddle_reader(self, paddle_ocr_cls):
        import inspect

        params = inspect.signature(paddle_ocr_cls).parameters
        if "use_doc_orientation_classify" in params:
            kwargs = {
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            if "use_gpu" in params:
                kwargs["use_gpu"] = True
            if self.model_profile == "mobile":
                kwargs.update(
                    {
                        "text_detection_model_name": "PP-OCRv5_mobile_det",
                        "text_recognition_model_name": "PP-OCRv5_mobile_rec",
                    }
                )
            else:
                kwargs["lang"] = "ch"
            return paddle_ocr_cls(**kwargs)
        return paddle_ocr_cls(use_angle_cls=False, lang="ch", use_gpu=True)


def _rapid_to_items(result):
    if result is None:
        return []
    items = []
    for bbox, text, score in result:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append(
            {
                "text": str(text),
                "rec_text": str(text),
                "score": float(score),
                "poly": [[xs[i], ys[i]] for i in range(4)],
            }
        )
    return [items]


def _iter_ocr_items(raw, offset: tuple[int, int] = (0, 0)):
    for page in raw or []:
        if isinstance(page, dict):
            texts = page.get("rec_texts") or page.get("texts") or []
            scores = page.get("rec_scores") or page.get("scores") or [1.0] * len(texts)
            boxes = page.get("rec_boxes")
            polys = page.get("rec_polys") or page.get("dt_polys")
            for index, (text, score) in enumerate(zip(texts, scores, strict=False)):
                yield _ocr_item(str(text), float(score), _indexed_box(boxes, index), _indexed_poly(polys, index), offset)
            continue

        for item in page or []:
            if isinstance(item, dict):
                text = item.get("text") or item.get("rec_text")
                score = item.get("score") or item.get("rec_score") or 1.0
                poly = item.get("poly")
                if text:
                    yield _ocr_item(str(text), float(score), item.get("box") or item.get("bbox"), poly, offset)
                continue
            text = item[1][0]
            score = float(item[1][1])
            yield _ocr_item(text, score, None, item[0], offset)


def _ocr_item(text: str, score: float, box, poly, offset: tuple[int, int]) -> dict:
    bbox = _normalize_bbox(box, poly, offset)
    normalized_text = normalize_ocr_text(text)
    item = {"text": normalized_text, "raw_text": text, "score": score}
    if bbox:
        item["bbox"] = bbox
        item["center"] = {
            "x": int(round((bbox["x1"] + bbox["x2"]) / 2)),
            "y": int(round((bbox["y1"] + bbox["y2"]) / 2)),
        }
    return item


def normalize_ocr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    replacements = {
        "（": "(",
        "）": ")",
        "【": "【",
        "】": "】",
        "﹘": "(",
        "﹙": ")",
        "：": ":",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split()).strip()


def _indexed_box(boxes, index: int):
    if boxes is None:
        return None
    try:
        return boxes[index]
    except (IndexError, TypeError):
        return None


def _indexed_poly(polys, index: int):
    if polys is None:
        return None
    try:
        return polys[index]
    except (IndexError, TypeError):
        return None


def _normalize_bbox(box, poly, offset: tuple[int, int]) -> dict | None:
    if box is not None:
        values = _flat_numbers(box)
        if len(values) >= 4:
            x1, y1, x2, y2 = values[:4]
            return _offset_bbox(x1, y1, x2, y2, offset)

    if poly is not None:
        values = _flat_numbers(poly)
        if len(values) >= 4:
            xs = values[0::2]
            ys = values[1::2]
            return _offset_bbox(min(xs), min(ys), max(xs), max(ys), offset)
    return None


def _flat_numbers(value) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list | tuple):
        result = []
        for item in value:
            result.extend(_flat_numbers(item))
        return result
    return []


def _offset_bbox(x1: float, y1: float, x2: float, y2: float, offset: tuple[int, int]) -> dict:
    ox, oy = offset
    return {
        "x1": int(round(min(x1, x2) + ox)),
        "y1": int(round(min(y1, y2) + oy)),
        "x2": int(round(max(x1, x2) + ox)),
        "y2": int(round(max(y1, y2) + oy)),
    }
