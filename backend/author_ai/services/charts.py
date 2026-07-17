"""
Helpers for detecting charts in PDFs and converting them into synthetic chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import uuid

from PyPDF2 import PdfReader  # type: ignore

from ..config import get_settings
from ..models import Chart
from .chart_parser import parse_chart_image
from .logger import setup_logger


logger = setup_logger(__name__)


def is_chart_image(image_bytes: bytes, width: int, height: int, min_area: Optional[int] = None) -> bool:
    """
    Lightweight image filter: reject tiny images and extreme aspect ratios.
    Structured so a richer classifier can replace it later.
    """
    settings = get_settings()
    min_area = min_area if min_area is not None else settings.chart_min_area
    area = max(0, width) * max(0, height)
    if area < min_area:
        return False
    if width <= 0 or height <= 0:
        return False
    aspect = width / float(height)
    return 0.3 <= aspect <= 3.5


def chart_image_to_struct(_image_bytes: bytes) -> Dict[str, Any]:
    """
    Convert a chart image into structured data. Uses a heuristic parser with
    OpenCV/pytesseract when available; falls back to a skeleton.
    """
    try:
        return parse_chart_image(_image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.debug("chart_image_to_struct fallback due to parse failure: %s", exc)
        return {
            "chart_type": "unknown",
            "title": None,
            "x_axis": {"label": None, "ticks": []},
            "y_axis": {"label": None, "unit": None},
            "series": [],
        }


def extract_charts_from_pdf(path: Path, doc_id: str) -> List[Chart]:
    """
    Scan the PDF for images that look like charts. Falls back silently on errors.
    """
    settings = get_settings()
    if not settings.chart_ingestion_enabled:
        return []
    charts: List[Chart] = []
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Chart extraction skipped; failed to read PDF %s: %s", path, exc)
        return []

    for page_number, page in enumerate(reader.pages, start=1):
        images = []
        images = []
        # PyPDF2 exposes images differently depending on version; try both access paths.
        try:
            images = list(getattr(page, "images", []) or [])
        except Exception:  # noqa: BLE001
            images = []
        try:
            for img_tuple in page.get_images(full=True):
                # Tuple layout: (xref, smask, width, height, bpc, colorspace, name, filter, ...)
                xref, _smask, width, height, *_rest = img_tuple
                data_bytes = None
                mapping = getattr(page, "images", None)
                if isinstance(mapping, dict):
                    image_obj = mapping.get(xref)
                    if image_obj is not None:
                        data_bytes = getattr(image_obj, "data", None)
                images.append(
                    {
                        "data": data_bytes,
                        "width": width,
                        "height": height,
                    }
                )
        except Exception:  # noqa: BLE001
            pass

        for image in images:
            data = getattr(image, "data", None) if not isinstance(image, dict) else image.get("data")
            width = getattr(image, "width", None) if not isinstance(image, dict) else image.get("width")
            height = getattr(image, "height", None) if not isinstance(image, dict) else image.get("height")
            if not data or not isinstance(data, (bytes, bytearray)):
                continue
            if width is None or height is None:
                continue
            if not is_chart_image(data, int(width), int(height), settings.chart_min_area):
                continue
            raw_struct = chart_image_to_struct(data)
            chart = Chart(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                page=page_number,
                figure_label=None,
                bbox=None,
                chart_type=raw_struct.get("chart_type") or "unknown",
                raw_json=raw_struct,
            )
            charts.append(chart)
    logger.info("Detected %d chart candidate(s) in %s", len(charts), path)
    return charts


def chart_to_chunks(chart: Chart, max_fact_points: Optional[int] = None) -> List[Dict[str, Any]]:
    settings = get_settings()
    fact_cap = max_fact_points if max_fact_points is not None else settings.chart_max_fact_points
    struct = chart.raw_json or {}
    chart_type = struct.get("chart_type") or chart.chart_type or "chart"
    x_axis = struct.get("x_axis") or {}
    y_axis = struct.get("y_axis") or {}
    title = struct.get("title")
    x_label = x_axis.get("label") or "x-axis"
    y_label = y_axis.get("label") or "y-axis"
    figure_label = chart.figure_label or "unknown"

    trends: List[str] = []
    for series in struct.get("series") or []:
        name = series.get("name") or "series"
        points = series.get("points") or []
        if len(points) >= 2:
            start = points[0]
            end = points[-1]
            trends.append(
                f"For series '{name}', value changes from {start.get('y')} at {start.get('x')} to {end.get('y')} at {end.get('x')}."
            )
    description_bits = []
    if title:
        description_bits.append(f"{title}.")
    description_bits.append(f"A {chart_type} showing {x_label} versus {y_label}.")
    if trends:
        description_bits.append(" ".join(trends[:2]))
    summary_text = (
        f"Chart summary (Figure {figure_label}, Page {chart.page}): " + " ".join(description_bits)
    ).strip()

    chunks: List[Dict[str, Any]] = [
        {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": chart.doc_id,
            "text": summary_text,
            "chunk_type": "chart_summary",
            "chart_id": chart.id,
            "metadata": {
                "chunk_type": "chart_summary",
                "chart_id": chart.id,
                "figure_label": chart.figure_label,
                "chart_type": chart_type,
                "parent_id": f"page-{chart.page}",
                "parent_title": f"Page {chart.page} - Chart",
                "parent_page": chart.page,
            },
        }
    ]

    # Optional fact chunks based on series points.
    def _select_points(points: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pts = list(points)
        selected: List[Dict[str, Any]] = []
        if pts:
            selected.append(pts[0])
            if len(pts) > 1:
                selected.append(pts[-1])
        numeric_points = [p for p in pts if isinstance(p.get("y"), (int, float))]
        if numeric_points:
            min_point = min(numeric_points, key=lambda p: p.get("y"))
            max_point = max(numeric_points, key=lambda p: p.get("y"))
            for candidate in (min_point, max_point):
                if candidate not in selected:
                    selected.append(candidate)
        return selected

    fact_points = 0
    y_unit = y_axis.get("unit")
    for series in struct.get("series") or []:
        series_name = series.get("name") or "series"
        for point in _select_points(series.get("points") or []):
            if fact_points >= fact_cap:
                break
            x_val = point.get("x")
            y_val = point.get("y")
            text = (
                f"Chart fact (Figure {figure_label}, Page {chart.page}): "
                f"In this chart, for series '{series_name}', the value at {x_val} is {y_val}"
            )
            if y_unit:
                text += f" {y_unit}"
            text += "."
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": chart.doc_id,
                    "text": text,
                    "metadata": {
                        "chunk_type": "chart_fact",
                        "chart_id": chart.id,
                        "figure_label": chart.figure_label,
                        "chart_type": chart_type,
                        "x_value": x_val,
                        "y_value": y_val,
                        "series_name": series_name,
                        "y_unit": y_unit,
                        "parent_id": f"page-{chart.page}",
                        "parent_title": f"Page {chart.page} - Chart",
                        "parent_page": chart.page,
                    },
                    "x_value": x_val,
                    "y_value": y_val,
                    "series_name": series_name,
                    "chunk_type": "chart_fact",
                    "chart_id": chart.id,
                }
            )
            fact_points += 1
        if fact_points >= fact_cap:
            break

    return chunks
