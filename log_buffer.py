"""
log_buffer.py — In-memory + file log buffer.

Captures log records from all loggers and:
  • Keeps the last MAX_LINES in memory (fast access, lost on restart).
  • Appends every line to logs/aditi.log (persists across restarts).

The /logs endpoint reads from the file so all logs survive server restarts.
"""
from __future__ import annotations
import collections
import logging
import os
import threading
from pathlib import Path

MAX_LINES = 2000   # in-memory cap (fallback when file unavailable)

_buffer: collections.deque[dict] = collections.deque(maxlen=MAX_LINES)
_lock   = threading.Lock()

# Resolved at attach() time so we don't import config at module level
_log_file_path: str = ""
_file_lock = threading.Lock()


class _MemHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            entry = {
                "ts":    record.asctime if hasattr(record, "asctime") else "",
                "level": record.levelname,
                "msg":   msg,
            }
            with _lock:
                _buffer.append(entry)
            # Also write to file
            if _log_file_path:
                with _file_lock:
                    try:
                        with open(_log_file_path, "a", encoding="utf-8") as fh:
                            fh.write(msg + "\n")
                    except OSError:
                        pass
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
    global _log_file_path
    from config import LOG_FILE, LOGS_DIR
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    _log_file_path = LOG_FILE

    root = logging.getLogger()
    if not any(isinstance(h, _MemHandler) for h in root.handlers):
        root.addHandler(_handler)


def get_lines() -> list[dict]:
    """Return in-memory lines (newest available since last restart)."""
    with _lock:
        return list(_buffer)


def get_lines_from_file(max_lines: int = 0) -> list[dict]:
    """
    Read all log lines from the log file.
    Returns newest-first. max_lines=0 means return everything.
    """
    if not _log_file_path:
        return get_lines()
    try:
        raw = Path(_log_file_path).read_text(encoding="utf-8", errors="replace")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if max_lines and len(lines) > max_lines:
            lines = lines[-max_lines:]
        result = []
        for ln in reversed(lines):
            # Parse level from format: "HH:MM:SS [LEVEL] name — msg"
            level = "INFO"
            if " [" in ln and "] " in ln:
                try:
                    level = ln.split("[")[1].split("]")[0].strip()
                except Exception:
                    pass
            result.append({"ts": "", "level": level, "msg": ln})
        return result
    except OSError:
        return get_lines()
