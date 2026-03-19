import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional


@dataclass
class _Channel:
    queue: asyncio.Queue[dict]
    loop: asyncio.AbstractEventLoop
    created_at_s: float


_channels: Dict[str, _Channel] = {}


def ensure_channel(request_id: str, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """
    Ensure a channel exists for request_id.

    If called from an async context, pass `loop=asyncio.get_running_loop()` so
    cross-thread publishers can publish safely via loop.call_soon_threadsafe.
    """
    if request_id in _channels:
        if loop is not None:
            _channels[request_id].loop = loop
        return
    if loop is None:
        # Best-effort; publishers may not be able to wake SSE without a loop.
        loop = asyncio.get_event_loop()
    _channels[request_id] = _Channel(queue=asyncio.Queue(maxsize=500), loop=loop, created_at_s=time.time())


def close_channel(request_id: str) -> None:
    ch = _channels.pop(request_id, None)
    if not ch:
        return
    try:
        ch.loop.call_soon_threadsafe(ch.queue.put_nowait, {"type": "close", "request_id": request_id})
    except Exception:
        pass


def publish(request_id: str, event: Dict[str, Any]) -> None:
    ch = _channels.get(request_id)
    if not ch:
        return
    payload = dict(event)
    payload["request_id"] = request_id
    try:
        ch.loop.call_soon_threadsafe(_safe_put_nowait, ch.queue, payload)
    except Exception:
        # If loop is gone / not running, drop.
        pass


def _safe_put_nowait(q: asyncio.Queue, item: dict) -> None:
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        # Drop oldest by draining 1 item, then retry once.
        try:
            _ = q.get_nowait()
        except Exception:
            return
        try:
            q.put_nowait(item)
        except Exception:
            return


async def sse_subscribe(request_id: str) -> AsyncIterator[str]:
    """
    Async generator that yields SSE 'data:' lines with JSON payloads.
    """
    ensure_channel(request_id, loop=asyncio.get_running_loop())
    q = _channels[request_id].queue

    # Initial hello (lets frontend know stream is alive).
    yield f"event: hello\ndata: {json.dumps({'type': 'hello', 'request_id': request_id})}\n\n"

    while True:
        event = await q.get()
        if event.get("type") == "close":
            yield f"event: close\ndata: {json.dumps(event)}\n\n"
            return
        yield f"data: {json.dumps(event)}\n\n"
