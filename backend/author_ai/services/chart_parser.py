"""
Lightweight chart parser that extracts approximate structure from chart images.
Falls back to a stub when dependencies (cv2/pytesseract) are unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore

from .logger import setup_logger


logger = setup_logger(__name__)


def _decode_image(image_bytes: bytes):
    if not cv2:
        return None
    arr = np.frombuffer(image_bytes, np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None
    return img


def _ocr_region(img, region: Tuple[int, int, int, int]) -> Optional[str]:
    if not pytesseract or not cv2:
        return None
    x, y, w, h = region
    try:
        crop = img[y : y + h, x : x + w]
        text = pytesseract.image_to_string(crop)  # type: ignore[attr-defined]
        cleaned = " ".join(text.split())
        return cleaned or None
    except Exception:  # noqa: BLE001
        return None


def _detect_pie(img_gray) -> bool:
    if not cv2:
        return False
    h, w = img_gray.shape[:2]
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.1 * w * h:
        return False
    perimeter = cv2.arcLength(largest, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity > 0.7


def _detect_bars(img_gray):
    if not cv2:
        return [], False
    h, w = img_gray.shape[:2]
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 0.002 * w * h:
            continue
        if bh > bw * 1.2:  # vertical-ish rectangle
            bars.append((x, y, bw, bh))
    bars = sorted(bars, key=lambda b: b[0])
    return bars, len(bars) >= 3


def _bar_series(bars, img_height: int, x_ticks: List[str]) -> List[Dict[str, Any]]:
    points = []
    tick_labels = x_ticks or []
    for idx, (x, y, bw, bh) in enumerate(bars):
        label = tick_labels[idx] if idx < len(tick_labels) else f"{idx + 1}"
        # Normalize height to a pseudo-value between 0 and 100.
        value = max(0.0, min(100.0, (bh / float(img_height)) * 100.0))
        points.append({"x": label, "y": round(value, 2)})
    return [{"name": "series", "points": points}]


def _line_series(edges, img_height: int, x_ticks: List[str]) -> List[Dict[str, Any]]:
    # Sample columns across the image and pick the top-most edge point as the signal.
    points = []
    h, w = edges.shape[:2]
    sample_count = min(12, max(6, w // 80))
    xs = np.linspace(0, w - 1, sample_count).astype(int)
    for idx, x in enumerate(xs):
        column = edges[:, x]
        ys = np.where(column > 0)[0]
        if ys.size == 0:
            continue
        y_val = ys.min()
        label = x_ticks[idx] if idx < len(x_ticks) else f"{idx + 1}"
        value = max(0.0, min(100.0, (1.0 - (y_val / float(h))) * 100.0))
        points.append({"x": label, "y": round(value, 2)})
    if not points:
        return []
    return [{"name": "series", "points": points}]


def _extract_ticks(img_gray, axis: str) -> List[str]:
    if not cv2:
        return []
    h, w = img_gray.shape[:2]
    if axis == "x":
        region = img_gray[int(h * 0.8) : h, :]
    else:
        region = img_gray[:, 0 : int(w * 0.2)]
    text = None
    if pytesseract:
        text = _ocr_region(img_gray if axis == "x" else img_gray, (0, 0, region.shape[1], region.shape[0]))
    if text:
        tokens = re.split(r"[\s,;]+", text)
        return [tok for tok in tokens if tok]
    return []


def parse_chart_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Parse a chart image into a minimal structured representation.
    Uses OpenCV (if available) to infer chart type and approximate series values,
    plus pytesseract (if available) for labels. Falls back to a stub skeleton.
    """
    skeleton = {
        "chart_type": "unknown",
        "title": None,
        "x_axis": {"label": None, "ticks": []},
        "y_axis": {"label": None, "unit": None},
        "series": [],
    }
    img = _decode_image(image_bytes)
    if img is None or not cv2:
        return skeleton
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Chart parser failed to convert image: %s", exc)
        return skeleton

    h, w = gray.shape[:2]
    title_region = (0, 0, w, max(1, int(0.15 * h)))
    title_text = _ocr_region(img, title_region)
    x_ticks = _extract_ticks(gray, "x")
    y_ticks = _extract_ticks(gray, "y")
    y_unit = None
    if y_ticks:
        # Simple unit guess: look for trailing non-digit token.
        for tok in reversed(y_ticks):
            if re.search(r"[A-Za-z%]", tok):
                y_unit = tok
                break

    chart_type = "unknown"
    series: List[Dict[str, Any]] = []

    bars, has_bars = _detect_bars(gray)
    if has_bars:
        chart_type = "bar"
        series = _bar_series(bars, h, x_ticks)
    elif _detect_pie(gray):
        chart_type = "pie"
        series = []
    else:
        # Treat as line-like: use edges to pick a path.
        edges = cv2.Canny(gray, 50, 150)
        line_series = _line_series(edges, h, x_ticks)
        if line_series:
            chart_type = "line"
            series = line_series

    return {
        "chart_type": chart_type,
        "title": title_text,
        "x_axis": {"label": None, "ticks": x_ticks},
        "y_axis": {"label": None, "unit": y_unit},
        "series": series,
    }

