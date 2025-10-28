from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class RequestContextFilter(logging.Filter):
    """Attach Flask request context metadata to log records when available."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from flask import g  # Imported lazily to avoid circular imports.

            record.request_id = getattr(g, "request_id", None)
        except RuntimeError:
            record.request_id = None
        return True


class StructuredFormatter(logging.Formatter):
    """Emit structured JSON logs with safe defaults."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.request_id:
            payload["request_id"] = record.request_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logging for the application."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Prevent duplicate handlers when reloading in development.
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    handler.addFilter(RequestContextFilter())

    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(handler)

