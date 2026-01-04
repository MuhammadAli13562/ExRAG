"""
Langfuse manual tracing wrapper for ExRAG.

Design goals:
- Optional dependency: if `langfuse` SDK isn't installed or not configured, tracing becomes a no-op.
- Disciplined payloads: redact secrets and truncate large fields before sending.
- Simple spans: per query trace + stage spans + tool/LLM events.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, List
import os
import re
import time


_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|secret|token|bearer)")


def _now_ms() -> int:
    return int(time.time() * 1000)


def redact(obj: Any) -> Any:
    """Recursively redact obvious secrets."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if _SECRET_KEY_RE.search(str(k)):
                out[k] = "***REDACTED***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(redact(x) for x in obj)
    if isinstance(obj, str):
        # redact bearer tokens
        if re.search(r"(?i)bearer\\s+[a-z0-9_\\-\\.]+", obj):
            return re.sub(r"(?i)bearer\\s+[a-z0-9_\\-\\.]+", "Bearer ***REDACTED***", obj)
        return obj
    return obj


def truncate(obj: Any, max_chars: int) -> Any:
    """Recursively truncate large strings. Keeps structure."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: truncate(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [truncate(x, max_chars) for x in obj]
    if isinstance(obj, tuple):
        return tuple(truncate(x, max_chars) for x in obj)
    if isinstance(obj, str):
        if len(obj) <= max_chars:
            return obj
        return obj[: max_chars] + f"...(truncated; {len(obj)} chars total)"
    return obj


def safe_payload(obj: Any, max_chars: int) -> Any:
    return truncate(redact(obj), max_chars=max_chars)


@dataclass(frozen=True)
class LangfuseConfig:
    host: str
    public_key: str
    secret_key: str
    project: Optional[str]
    max_chars: int

    @staticmethod
    def from_env() -> Optional["LangfuseConfig"]:
        tracing = os.getenv("EXRAG_TRACING", "").strip().lower()
        if tracing != "langfuse":
            return None

        # Support both names. Prefer LANGFUSE_BASE_URL if provided.
        host = (os.getenv("LANGFUSE_BASE_URL", "") or os.getenv("LANGFUSE_HOST", "")).strip()
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        project = os.getenv("LANGFUSE_PROJECT", "").strip() or None
        max_chars = int(os.getenv("EXRAG_LANGFUSE_MAX_CHARS", "4000"))

        if not host or not public_key or not secret_key:
            return None

        return LangfuseConfig(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
            project=project,
            max_chars=max_chars,
        )


class NoopTracer:
    enabled = False

    def start_trace(self, *_: Any, **__: Any) -> None:
        return None

    def end_trace(self, *_: Any, **__: Any) -> None:
        return None

    @contextmanager
    def span(self, *_: Any, **__: Any) -> Iterator[None]:
        yield None

    def event(self, *_: Any, **__: Any) -> None:
        return None

    def flush(self) -> None:
        return None


class LangfuseTracer:
    enabled = True

    def __init__(self, cfg: LangfuseConfig):
        self.cfg = cfg
        self._client = None
        self._trace = None
        self._span_stack: List[Any] = []
        self.last_error: Optional[str] = None
        self.trace_id: Optional[str] = None

        try:
            from langfuse import Langfuse  # type: ignore

            self._client = Langfuse(
                public_key=cfg.public_key,
                secret_key=cfg.secret_key,
                host=cfg.host,
            )
        except Exception:
            # If SDK missing or init fails, fall back to noop.
            self.enabled = False

    def start_trace(self, name: str, input: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return None
        try:
            # Newer Langfuse Python SDK is OpenTelemetry-based.
            # We create a trace_id and emit a start event to ensure the trace exists server-side.
            trace_id = self._client.create_trace_id()  # type: ignore[attr-defined]
            self.trace_id = trace_id
            self._trace = {"trace_id": trace_id}

            from langfuse.types import TraceContext  # type: ignore

            trace_context = TraceContext(trace_id=trace_id)
            self._client.create_event(  # type: ignore[attr-defined]
                trace_context=trace_context,
                name=name,
                input=safe_payload(input, self.cfg.max_chars),
                metadata=safe_payload(metadata or {}, self.cfg.max_chars),
            )
            self.last_error = None
        except Exception as e:
            # Disable on failure to avoid breaking retrieval
            self.last_error = f"{type(e).__name__}: {e}"
            self.enabled = False

    def end_trace(self, output: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self._client is None or self._trace is None:
            return None
        try:
            from langfuse.types import TraceContext  # type: ignore

            trace_id = self.trace_id
            if not trace_id:
                return None
            trace_context = TraceContext(trace_id=trace_id)
            self._client.create_event(  # type: ignore[attr-defined]
                trace_context=trace_context,
                name="trace.end",
                output=safe_payload(output, self.cfg.max_chars),
                metadata=safe_payload(metadata or {}, self.cfg.max_chars),
            )
        except Exception:
            pass

    @contextmanager
    def span(self, name: str, input: Any = None, metadata: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
        if not self.enabled or self._client is None or self._trace is None:
            yield None
            return

        span_obj = None
        start = _now_ms()
        try:
            trace_id = self.trace_id
            if not trace_id:
                trace_context = None
            else:
                from langfuse.types import TraceContext  # type: ignore

                trace_context = TraceContext(trace_id=trace_id)
            span_obj = self._client.start_span(  # type: ignore[attr-defined]
                trace_context=trace_context,
                name=name,
                input=safe_payload(input, self.cfg.max_chars),
                metadata=safe_payload(metadata or {}, self.cfg.max_chars),
            )
        except Exception:
            span_obj = None

        self._span_stack.append(span_obj)
        try:
            yield span_obj
        finally:
            self._span_stack.pop()
            duration_ms = _now_ms() - start
            try:
                if span_obj is not None and hasattr(span_obj, "update"):
                    span_obj.update(metadata=safe_payload({"duration_ms": duration_ms}, self.cfg.max_chars))  # type: ignore[attr-defined]
                if span_obj is not None and hasattr(span_obj, "end"):
                    span_obj.end()  # type: ignore[attr-defined]
            except Exception:
                pass

    def event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self._client is None or self._trace is None:
            return None
        data = safe_payload(payload or {}, self.cfg.max_chars)
        try:
            trace_id = self.trace_id
            if not trace_id:
                trace_context = None
            else:
                from langfuse.types import TraceContext  # type: ignore

                trace_context = TraceContext(trace_id=trace_id)
            self._client.create_event(  # type: ignore[attr-defined]
                trace_context=trace_context,
                name=name,
                metadata=data,
            )
        except Exception:
            pass

    def flush(self) -> None:
        if not self.enabled or self._client is None:
            return None
        try:
            if hasattr(self._client, "flush"):
                self._client.flush()  # type: ignore[attr-defined]
        except Exception:
            pass


def get_tracer() -> NoopTracer | LangfuseTracer:
    """
    Return a process-wide tracer singleton.

    This is important because the agent builds stage closures at graph-construction time,
    while the trace is started later at query time. Using a singleton ensures spans/events
    emitted by stage functions attach to the same active trace.
    """
    global _TRACER
    if _TRACER is not None:
        return _TRACER

    cfg = LangfuseConfig.from_env()
    if cfg is None:
        _TRACER = NoopTracer()
        return _TRACER

    tracer = LangfuseTracer(cfg)
    if not tracer.enabled:
        _TRACER = NoopTracer()
        return _TRACER

    _TRACER = tracer
    return _TRACER


# Module-level singleton tracer (initialized on first get_tracer()).
_TRACER: NoopTracer | LangfuseTracer | None = None


def reset_tracer() -> None:
    """Reset the tracer singleton (useful for tests)."""
    global _TRACER
    _TRACER = None


