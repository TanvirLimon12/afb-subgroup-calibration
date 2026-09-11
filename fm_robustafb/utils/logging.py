from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "fm_robustafb") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("fm_robustafb")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
