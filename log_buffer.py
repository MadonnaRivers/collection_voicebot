"""
log_buffer.py — In-memory circular log buffer.

Captures the last MAX_LINES log records from all loggers so they can be
served via the /logs HTTP endpoint without needing file or journald access.
"""
from __future__ import annotations
import collections
import logging
import threading

MAX_LINES = 600   # keep last 600 log lines in memory

_buffer: collections.deque[dict] = collections.deque(maxlen=MAX_LINES)
_lock = threading.Lock()


class _MemHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _lock:
                _buffer.append({
                    "ts":    record.asctime if hasattr(record, "asctime") else "",
                    "level": record.levelname,
                    "msg":   msg,
                })
        except Exception:
            pass


_handler = _MemHandler()
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
)


def attach() -> None:
    """Attach the buffer handler to the root logger. Call once at startup."""
    root = logging.getLogger()
    # Avoid duplicate handlers if called more than once
    if not any(isinstance(h, _MemHandler) for h in root.handlers):
        root.addHandler(_handler)


def get_lines() -> list[dict]:
    with _lock:
        return list(_buffer)
