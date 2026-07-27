"""On-disk JSON cache for EDGAR responses.

Caching previously existed only as `@st.cache_data` inside the Streamlit app,
which is per-session and invisible to anything imported as a library. That is
fine for one deal at a time and unworkable for a corpus run: scoring a few
hundred deals means two EDGAR round-trips each, and SEC's fair-access policy
caps us at 10 requests/second before it starts refusing service.

Entries are keyed on the arguments that determine the response and never
expire -- a company's FY2019 10-K is immutable, so a stale read isn't possible
for the historical data this caches. Delete `data/cache/` to force a refetch.

Negative results are cached too. At corpus scale a meaningful fraction of
targets have no pre-announcement 10-K at all, and rediscovering that costs a
full filings index load per deal per run.
"""

import json
import os
import re
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

_CACHE_DIR = Path(os.environ.get("SYNERGY_CACHE_DIR", "data/cache"))

_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9._-]")


class CachedFailure(Exception):
    """Raised when a previously-recorded failure is replayed from cache."""


def _slug(value: str) -> str:
    return _UNSAFE_KEY_CHARS.sub("_", value)


def _path(namespace: str, key: str) -> Path:
    return _CACHE_DIR / _slug(namespace) / f"{_slug(key)}.json"


def _encode(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value


def _decode(payload, cls: type[T]):
    if isinstance(payload, list):
        return [_decode(item, cls) for item in payload]
    known = {f.name for f in fields(cls)}
    # Tolerate schema drift: a field added since the entry was written comes
    # back as None rather than a TypeError on an otherwise-valid cache hit.
    missing = {name: None for name in known - payload.keys()}
    return cls(**{**{k: v for k, v in payload.items() if k in known}, **missing})


def cached(namespace: str, key: str, cls: type[T], produce: Callable[[], T | list[T]]) -> T | list[T]:
    """Return `produce()`, reading from / writing to disk on the way.

    `cls` is the dataclass to rehydrate into. `produce` raising ValueError is
    treated as a permanent negative result (this filer has no such data) and
    recorded; it replays as CachedFailure, which callers should treat exactly
    as they would the original ValueError.
    """
    path = _path(namespace, key)
    if path.exists():
        payload = json.loads(path.read_text())
        if not payload.get("ok", True):
            raise CachedFailure(payload["error"])
        return _decode(payload["data"], cls)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        value = produce()
    except ValueError as exc:
        path.write_text(json.dumps({"ok": False, "error": str(exc)}))
        raise

    path.write_text(json.dumps({"ok": True, "data": _encode(value)}, indent=2))
    return value
