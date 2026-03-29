from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Tuple


@dataclass
class TimedSample:
    t: float
    x: float


class RingBuffer:
    def __init__(self, maxlen: int = 4096) -> None:
        self._buf: Deque[TimedSample] = deque(maxlen=maxlen)

    def append(self, t: float, x: float) -> None:
        self._buf.append(TimedSample(float(t), float(x)))

    def extend(self, samples: Iterable[Tuple[float, float]]) -> None:
        for t, x in samples:
            self.append(t, x)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def to_lists(self) -> Tuple[List[float], List[float]]:
        ts = [s.t for s in self._buf]
        xs = [s.x for s in self._buf]
        return ts, xs

    def last_n(self, n: int) -> Tuple[List[float], List[float]]:
        items = list(self._buf)[-int(n):]
        return [s.t for s in items], [s.x for s in items]
