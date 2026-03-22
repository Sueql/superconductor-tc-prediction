from __future__ import annotations

import time
from typing import Optional


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def stage_start(name: str) -> float:
    log(f"START | {name}")
    return time.time()


def stage_end(name: str, started_at: Optional[float] = None) -> None:
    if started_at is None:
        log(f"DONE  | {name}")
    else:
        elapsed = time.time() - started_at
        log(f"DONE  | {name} | elapsed={elapsed:.2f}s")


def progress(current: int, total: int, prefix: str, every: int = 1) -> None:
    if total <= 0:
        return
    if current == 1 or current == total or current % max(1, every) == 0:
        pct = 100.0 * current / total
        log(f"PROGRESS | {prefix} | {current}/{total} ({pct:.1f}%)")
