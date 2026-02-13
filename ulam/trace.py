from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, TextIO

from .types import ProofStep


class TraceLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self._path = path
        self._fh: Optional[TextIO] = None
        if path is None:
            return
        if str(path) == "-":
            self._fh = None
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    def log_step(self, step: ProofStep) -> None:
        payload = json.dumps(asdict(step), ensure_ascii=True)
        if self._path is None:
            return
        if self._fh is None:
            print(payload)
            return
        self._fh.write(payload + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
