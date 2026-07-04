"""Small, stable display IDs for workers.

ByteTrack's raw track IDs grow forever (W34, W115...). For a floor with a
handful of workers the owner wants W1..W4. This allocator hands out the
smallest free number and reuses numbers once a worker is gone.
"""
import threading

_lock = threading.Lock()
_used: set[int] = set()


def acquire() -> int:
    with _lock:
        n = 1
        while n in _used:
            n += 1
        _used.add(n)
        return n


def release(n: int) -> None:
    with _lock:
        _used.discard(n)
