"""Optional OCR backend using NVIDIA nemotron-ocr-v2.

Digital PDFs keep using pypdf because it is free and fast. This module is only
invoked when a PDF has little or no text layer (or when ``OCR_MODE=always``).
Pages are rendered with pypdfium2, encoded as JPEG tiles and sent to the
nemotron-ocr-v2 endpoint. The result is line-level text with normalized
bounding boxes and OCR confidence, which the extractor attaches to its
evidence records.

Environment variables:

    OCR_PROVIDER=none|nemotron          (default: none)
    OCR_MODE=auto|always|never          (default: auto)
    OCR_MIN_TEXT_CHARS=200              (auto triggers below this)
    OCR_FIELD_RECOVERY=false            (OCR when key fields are missing)
    OCR_DPI=150
    OCR_MAX_PAGES=10
    OCR_TIMEOUT_SECONDS=60
    OCR_MAX_RETRIES=2
    OCR_MERGE_LEVEL=sentence|word|paragraph
    NEMOTRON_OCR_URL=https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2
    NEMOTRON_OCR_API_KEY=... | NVIDIA_API_KEY=...
    NEMOTRON_OCR_MAX_BASE64_CHARS=175000

The hosted build.nvidia.com endpoint caps an inline base64 image at roughly
180,000 characters, so large pages are split into vertical tiles. Pointing
``NEMOTRON_OCR_URL`` at a self-hosted NIM removes that limit automatically.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2"
DEFAULT_MAX_BASE64_CHARS = 175_000


class OCRError(RuntimeError):
    """Raised when rendering, transport or response parsing fails."""


@dataclass
class OCRLine:
    text: str
    bbox: list[float]  # [x0, y0, x1, y1], normalized to 0..1
    confidence: float


@dataclass
class OCRResult:
    pages: list[list[OCRLine]]
    provider: str = "nemotron-ocr-v2"
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OCRConfig:
    url: str
    api_key: str | None
    timeout: float
    dpi: int
    max_base64_chars: int
    merge_level: str
    max_pages: int
    max_retries: int

    @property
    def configured(self) -> bool:
        if "ai.api.nvidia.com" in self.url:
            return bool(self.api_key)
        # Self-hosted NIM instances usually have no auth.
        return True

    @classmethod
    def from_env(cls) -> "OCRConfig":
        url = os.getenv("NEMOTRON_OCR_URL", DEFAULT_URL).strip() or DEFAULT_URL
        api_key = (
            os.getenv("NEMOTRON_OCR_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or None
        )
        max_b64_raw = os.getenv("NEMOTRON_OCR_MAX_BASE64_CHARS")
        if max_b64_raw:
            try:
                max_b64 = int(max_b64_raw)
            except ValueError:
                max_b64 = DEFAULT_MAX_BASE64_CHARS
        elif "ai.api.nvidia.com" in url:
            max_b64 = DEFAULT_MAX_BASE64_CHARS
        else:
            max_b64 = 0  # self-hosted: no inline-size splitting

        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        return cls(
            url=url,
            api_key=api_key.strip() if api_key else None,
            timeout=_float("OCR_TIMEOUT_SECONDS", 60.0),
            dpi=max(72, _int("OCR_DPI", 150)),
            max_base64_chars=max_b64,
            merge_level=os.getenv("OCR_MERGE_LEVEL", "sentence").strip().lower(),
            max_pages=max(1, _int("OCR_MAX_PAGES", 10)),
            max_retries=max(0, _int("OCR_MAX_RETRIES", 2)),
        )


def ocr_enabled() -> bool:
    provider = os.getenv("OCR_PROVIDER", "none").strip().lower()
    return provider in {"nemotron", "nemotron-ocr-v2", "nvidia"}


def ocr_mode() -> str:
    mode = os.getenv("OCR_MODE", "auto").strip().lower()
    return mode if mode in {"auto", "always", "never"} else "auto"


def min_text_chars() -> int:
    try:
        return max(0, int(os.getenv("OCR_MIN_TEXT_CHARS", "200")))
    except ValueError:
        return 200


def field_recovery_enabled() -> bool:
    return os.getenv("OCR_FIELD_RECOVERY", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@dataclass
class _Tile:
    data: bytes
    width: int
    height: int
    y_offset: float
    y_scale: float


def render_pdf_pages(path: Path, dpi: int, max_pages: int) -> list[Any]:
    """Render PDF pages to grayscale PIL images using pypdfium2."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise OCRError(
            "pypdfium2 is required for OCR rendering. Add it to requirements.txt."
        ) from exc

    try:
        document = pdfium.PdfDocument(str(path))
    except Exception as exc:  # noqa: BLE001 - surface a useful message
        raise OCRError(f"Could not open PDF for OCR: {exc}") from exc

    images: list[Any] = []
    try:
        page_count = min(len(document), max_pages)
        for index in range(page_count):
            page = document[index]
            bitmap = page.render(scale=dpi / 72.0)
            try:
                image = bitmap.to_pil().convert("L")
            finally:
                bitmap.close()
            page.close()
            images.append(image)
    finally:
        document.close()

    return images


def _jpeg_bytes(image: Any, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _b64_len(data: bytes) -> int:
    return len(base64.b64encode(data))


def _split_vertical(image: Any, parts: int) -> list[tuple[Any, int]]:
    width, height = image.size
    step = max(1, height // parts)
    crops: list[tuple[Any, int]] = []
    for index in range(parts):
        top = index * step
        bottom = height if index == parts - 1 else (index + 1) * step
        crops.append((image.crop((0, top, width, bottom)), top))
    return crops


def _downscale_to_fit(image: Any, max_chars: int) -> tuple[bytes, Any] | None:
    width, height = image.size
    for scale in (0.85, 0.75, 0.65, 0.55, 0.45):
        resized = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale)))
        )
        data = _jpeg_bytes(resized, 75)
        if _b64_len(data) <= max_chars:
            return data, resized
    return None


def _tiles_for(image: Any, max_chars: int) -> list[_Tile]:
    width, height = image.size
    if not max_chars:
        return [_Tile(_jpeg_bytes(image, 85), width, height, 0.0, 1.0)]

    for quality in (85, 75, 65):
        data = _jpeg_bytes(image, quality)
        if _b64_len(data) <= max_chars:
            return [_Tile(data, width, height, 0.0, 1.0)]

    for parts in (2, 3, 4):
        tiles: list[_Tile] = []
        fits = True
        for crop, top in _split_vertical(image, parts):
            data: bytes | None = _jpeg_bytes(crop, 80)
            crop_width, crop_height = crop.size
            if _b64_len(data) > max_chars:
                fitted = _downscale_to_fit(crop, max_chars)
                if fitted is None:
                    fits = False
                    break
                data, resized = fitted
                crop_width, crop_height = resized.size
            tiles.append(
                _Tile(
                    data=data,
                    width=crop_width,
                    height=crop_height,
                    y_offset=top / height,
                    y_scale=(crop.size[1]) / height,
                )
            )
        if fits:
            return tiles

    raise OCRError(
        "A rendered page could not fit the nemotron-ocr-v2 inline image limit. "
        "Lower OCR_DPI or point NEMOTRON_OCR_URL at a self-hosted NIM."
    )


def _payload(image_bytes: bytes, merge_level: str) -> dict[str, Any]:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload: dict[str, Any] = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:image/jpeg;base64,{encoded}",
            }
        ]
    }
    if merge_level:
        payload["merge_levels"] = [merge_level]
    return payload


def _post(config: OCRConfig, payload: dict[str, Any]) -> Any:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            response = httpx.post(
                config.url,
                json=payload,
                headers=headers,
                timeout=config.timeout,
            )
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < config.max_retries:
                time.sleep(min(2**attempt, 8))
                continue
            break

        if response.status_code in {429, 500, 502, 503, 504} and attempt < config.max_retries:
            time.sleep(min(2**attempt, 8))
            continue

        if response.status_code == 400 and "merge_levels" in payload:
            # Some hosted variants reject merge_levels; retry without it.
            payload = {key: value for key, value in payload.items() if key != "merge_levels"}
            continue

        if response.status_code >= 400:
            detail = response.text[:300].replace(config.api_key or "", "***")
            raise OCRError(
                f"nemotron-ocr-v2 returned HTTP {response.status_code}: {detail}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise OCRError("nemotron-ocr-v2 returned invalid JSON.") from exc

    raise OCRError(f"nemotron-ocr-v2 request failed: {last_error}")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_coords(
    xs: list[float], ys: list[float], width: int, height: int
) -> list[float] | None:
    if not xs or not ys:
        return None
    if max(max(xs), max(ys)) > 1.5 and width and height:
        xs = [x / width for x in xs]
        ys = [y / height for y in ys]
    return [
        round(_clamp(min(xs)), 4),
        round(_clamp(min(ys)), 4),
        round(_clamp(max(xs)), 4),
        round(_clamp(max(ys)), 4),
    ]


def _box_from_points(points: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(points, list) or not points:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if isinstance(point, dict):
            xs.append(float(point.get("x", 0.0)))
            ys.append(float(point.get("y", 0.0)))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    return _normalize_coords(xs, ys, width, height)


def _box_from_legacy(box: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or not box:
        return None
    if isinstance(box[0], (list, tuple)):
        return _box_from_points(list(box), width, height)
    if len(box) >= 4:
        return _normalize_coords(
            [float(box[0]), float(box[2])],
            [float(box[1]), float(box[3])],
            width,
            height,
        )
    return None


def _parse_response(data: Any, width: int, height: int) -> list[OCRLine]:
    """Accept the NIM ``text_detections`` shape and the legacy flat arrays."""
    lines: list[OCRLine] = []
    if not isinstance(data, dict):
        return lines

    for page in data.get("data") or []:
        if not isinstance(page, dict):
            continue
        for detection in page.get("text_detections") or []:
            if not isinstance(detection, dict):
                continue
            prediction = detection.get("text_prediction") or {}
            text = prediction.get("text") or detection.get("text") or ""
            confidence = prediction.get(
                "confidence", detection.get("confidence", 0.0)
            )
            box = detection.get("bounding_box") or {}
            points = box.get("points") if isinstance(box, dict) else box
            bbox = _box_from_points(points, width, height)
            if not bbox:
                bbox = _box_from_legacy(box, width, height) if box else None
            if not text or not bbox:
                continue
            for part in str(text).splitlines():
                part = part.strip()
                if part:
                    lines.append(
                        OCRLine(
                            text=part,
                            bbox=list(bbox),
                            confidence=_clamp(float(confidence or 0.0)),
                        )
                    )

    if lines:
        return lines

    # Legacy / build.nvidia.com prototype shape.
    texts = data.get("ocr_txts") or data.get("texts") or []
    confidences = data.get("ocr_confs") or data.get("confidences") or []
    boxes = data.get("ocr_boxes") or data.get("boxes") or []
    for index, text in enumerate(texts):
        bbox = _box_from_legacy(boxes[index] if index < len(boxes) else None, width, height)
        if not text or not bbox:
            continue
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        for part in str(text).splitlines():
            part = part.strip()
            if part:
                lines.append(
                    OCRLine(
                        text=part,
                        bbox=list(bbox),
                        confidence=_clamp(confidence),
                    )
                )
    return lines


def _group_lines(detections: list[OCRLine]) -> list[OCRLine]:
    """Group word/sentence detections into visual lines using their boxes."""
    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2.0, item.bbox[0]),
    )
    buckets: list[list[OCRLine]] = []
    for detection in ordered:
        center = (detection.bbox[1] + detection.bbox[3]) / 2.0
        placed = False
        for bucket in buckets:
            bucket_center = sum(
                (item.bbox[1] + item.bbox[3]) / 2.0 for item in bucket
            ) / len(bucket)
            bucket_height = max(
                item.bbox[3] - item.bbox[1] for item in bucket
            )
            if abs(center - bucket_center) <= max(0.006, bucket_height * 0.55):
                bucket.append(detection)
                placed = True
                break
        if not placed:
            buckets.append([detection])

    lines: list[OCRLine] = []
    for bucket in buckets:
        bucket.sort(key=lambda item: item.bbox[0])
        text = " ".join(item.text.strip() for item in bucket if item.text.strip())
        if not text:
            continue
        lines.append(
            OCRLine(
                text=text,
                bbox=[
                    round(min(item.bbox[0] for item in bucket), 4),
                    round(min(item.bbox[1] for item in bucket), 4),
                    round(max(item.bbox[2] for item in bucket), 4),
                    round(max(item.bbox[3] for item in bucket), 4),
                ],
                confidence=round(min(item.confidence for item in bucket), 4),
            )
        )

    lines.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return lines


def ocr_pdf(path: Path, config: OCRConfig | None = None) -> OCRResult:
    """Render and OCR a PDF, returning line-level text per page."""
    config = config or OCRConfig.from_env()
    if not config.configured:
        raise OCRError(
            "OCR_PROVIDER is enabled but no NVIDIA API key is configured. "
            "Set NEMOTRON_OCR_API_KEY or NVIDIA_API_KEY."
        )

    images = render_pdf_pages(path, config.dpi, config.max_pages)
    pages: list[list[OCRLine]] = []
    warnings: list[str] = []

    for page_index, image in enumerate(images, start=1):
        detections: list[OCRLine] = []
        for tile in _tiles_for(image, config.max_base64_chars):
            data = _post(config, _payload(tile.data, config.merge_level))
            for line in _parse_response(data, tile.width, tile.height):
                line.bbox = [
                    line.bbox[0],
                    round(_clamp(tile.y_offset + line.bbox[1] * tile.y_scale), 4),
                    line.bbox[2],
                    round(_clamp(tile.y_offset + line.bbox[3] * tile.y_scale), 4),
                ]
                detections.append(line)
        pages.append(_group_lines(detections))

    if not any(pages):
        warnings.append("OCR ran but did not return any text.")

    return OCRResult(pages=pages, warnings=warnings)
