# utils/logging.py
import logging
import json
import sys

class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # agrega extras si vienen
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            base.update(record.extra_fields)
        return json.dumps(base, ensure_ascii=False)

def get_logger(name: str = "app", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    logger.addHandler(h)
    logger.propagate = False
    return logger
