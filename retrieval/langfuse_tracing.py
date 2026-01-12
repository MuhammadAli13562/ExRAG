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
from typing import Any, Dict, Iterator, Optional, List, Mapping
from pathlib import Path
import os
import re
import time

# Load .env file from project root if it exists
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    # dotenv not installed, skip .env loading
    pass


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
        self._trace: Any = None
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

    def _get_trace_id(self, trace_obj: Any) -> Optional[str]:
        """Best-effort extraction of a trace id across SDK versions."""
        for attr in ("id", "trace_id", "traceId", "_id"):
            try:
                v = getattr(trace_obj, attr, None)
                if isinstance(v, str) and v:
                    return v
            except Exception:
                continue
        return None

    def _merge_metadata(self, base: Optional[Mapping[str, Any]], extra: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(base or {})
        if extra:
            out.update(dict(extra))
        return out

    def start_trace(self, name: str, input: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return None
        try:
            # Prefer high-level SDK APIs when available (`langfuse.trace(...)`), as those
            # correctly populate trace input/output fields in the UI.
            if hasattr(self._client, "trace"):
                self._trace = self._client.trace(  # type: ignore[attr-defined]
                    name=name,
                    input=safe_payload(input, self.cfg.max_chars),
                    metadata=safe_payload(metadata or {}, self.cfg.max_chars),
                )
                self.trace_id = self._get_trace_id(self._trace)
                self.last_error = None
                return None

            # Fallback: OpenTelemetry/low-level APIs (best-effort).
            trace_id = None
            if hasattr(self._client, "create_trace_id"):
                trace_id = self._client.create_trace_id()  # type: ignore[attr-defined]
            if isinstance(trace_id, str) and trace_id:
                self.trace_id = trace_id
                self._trace = {"trace_id": trace_id}
                try:
                    from langfuse.types import TraceContext  # type: ignore

                    trace_context = TraceContext(trace_id=trace_id)
                except Exception:
                    trace_context = None
                if hasattr(self._client, "create_event"):
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
            # High-level SDK: update trace output/metadata (these show in trace view).
            if hasattr(self._trace, "update"):
                self._trace.update(  # type: ignore[attr-defined]
                    output=safe_payload(output, self.cfg.max_chars),
                    metadata=safe_payload(metadata or {}, self.cfg.max_chars),
                )
            if hasattr(self._trace, "end"):
                self._trace.end()  # type: ignore[attr-defined]
                return None

            # Fallback: emit an end event with output.
            trace_id = self.trace_id
            if trace_id and hasattr(self._client, "create_event"):
                try:
                    from langfuse.types import TraceContext  # type: ignore

                    trace_context = TraceContext(trace_id=trace_id)
                except Exception:
                    trace_context = None
                self._client.create_event(  # type: ignore[attr-defined]
                    trace_context=trace_context,
                    name="trace.end",
                    output=safe_payload(output, self.cfg.max_chars),
                    metadata=safe_payload(metadata or {}, self.cfg.max_chars),
                )
        except Exception:
            pass

    class _SpanProxy:
        """
        A minimal adapter around Langfuse span-like objects.

        Goal: let callsites do `span.update(output=...)` and have it reliably land in
        the span's output field (not metadata), regardless of the underlying SDK version.
        """

        def __init__(self, tracer: "LangfuseTracer", span_obj: Any):
            self._tracer = tracer
            self._span = span_obj
            self._pending_output: Any = None
            self._pending_input: Any = None
            self._pending_metadata: Dict[str, Any] = {}

        def update(self, input: Any = None, output: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
            if input is not None:
                self._pending_input = input
            if output is not None:
                self._pending_output = output
            if metadata:
                self._pending_metadata.update(metadata)

            if self._span is None:
                return None
            kwargs: Dict[str, Any] = {}
            if input is not None:
                kwargs["input"] = safe_payload(input, self._tracer.cfg.max_chars)
            if output is not None:
                kwargs["output"] = safe_payload(output, self._tracer.cfg.max_chars)
            if metadata is not None:
                kwargs["metadata"] = safe_payload(metadata, self._tracer.cfg.max_chars)
            try:
                if hasattr(self._span, "update"):
                    self._span.update(**kwargs)  # type: ignore[attr-defined]
            except TypeError:
                # Older SDKs may accept only metadata updates.
                try:
                    if "metadata" in kwargs and hasattr(self._span, "update"):
                        self._span.update(metadata=kwargs["metadata"])  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                pass

        def end(self, output: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
            if output is not None:
                self._pending_output = output
            if metadata:
                self._pending_metadata.update(metadata)

            if self._span is None:
                return None

            out_payload = safe_payload(self._pending_output, self._tracer.cfg.max_chars)
            meta_payload = safe_payload(self._pending_metadata, self._tracer.cfg.max_chars)
            inp_payload = safe_payload(self._pending_input, self._tracer.cfg.max_chars)
            try:
                if hasattr(self._span, "end"):
                    # Try the richest signature first.
                    try:
                        self._span.end(output=out_payload, metadata=meta_payload)  # type: ignore[attr-defined]
                        return None
                    except TypeError:
                        # Some SDKs don't accept args on end; try update then end().
                        if hasattr(self._span, "update"):
                            try:
                                self._span.update(input=inp_payload, output=out_payload, metadata=meta_payload)  # type: ignore[attr-defined]
                            except Exception:
                                try:
                                    self._span.update(metadata=meta_payload)  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                        self._span.end()  # type: ignore[attr-defined]
            except Exception:
                pass

    @contextmanager
    def span(self, name: str, input: Any = None, metadata: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
        if not self.enabled or self._client is None or self._trace is None:
            yield None
            return

        raw_span_obj = None
        span_proxy: Optional[LangfuseTracer._SpanProxy] = None
        start = _now_ms()
        try:
            # High-level SDK: `trace.span(...)`
            if hasattr(self._trace, "span"):
                raw_span_obj = self._trace.span(  # type: ignore[attr-defined]
                    name=name,
                    input=safe_payload(input, self.cfg.max_chars),
                    metadata=safe_payload(metadata or {}, self.cfg.max_chars),
                )
            else:
                # Fallback: OpenTelemetry/low-level APIs.
                trace_id = self.trace_id
                if not trace_id:
                    trace_context = None
                else:
                    try:
                        from langfuse.types import TraceContext  # type: ignore

                        trace_context = TraceContext(trace_id=trace_id)
                    except Exception:
                        trace_context = None
                if hasattr(self._client, "start_span"):
                    raw_span_obj = self._client.start_span(  # type: ignore[attr-defined]
                        trace_context=trace_context,
                        name=name,
                        input=safe_payload(input, self.cfg.max_chars),
                        metadata=safe_payload(metadata or {}, self.cfg.max_chars),
                    )
        except Exception:
            raw_span_obj = None

        span_proxy = LangfuseTracer._SpanProxy(self, raw_span_obj)

        self._span_stack.append(span_proxy)
        try:
            yield span_proxy
        finally:
            self._span_stack.pop()
            duration_ms = _now_ms() - start
            try:
                if span_proxy is not None:
                    # Always attach duration metadata, while preserving any caller-set output.
                    span_proxy.update(metadata={"duration_ms": duration_ms})
                    span_proxy.end()
            except Exception:
                pass

    def event(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or self._client is None or self._trace is None:
            return None
        try:
            # Backwards compatible: if caller only passed `payload`, treat it as metadata.
            merged_meta = self._merge_metadata(metadata, payload)
            data_meta = safe_payload(merged_meta or {}, self.cfg.max_chars)
            data_inp = safe_payload(input, self.cfg.max_chars) if input is not None else None
            data_out = safe_payload(output, self.cfg.max_chars) if output is not None else None

            # High-level SDK: `trace.event(...)` if present.
            if hasattr(self._trace, "event"):
                self._trace.event(  # type: ignore[attr-defined]
                    name=name,
                    input=data_inp,
                    output=data_out,
                    metadata=data_meta,
                )
                return None

            # Fallback: `create_event` with trace context.
            trace_id = self.trace_id
            if not trace_id:
                trace_context = None
            else:
                try:
                    from langfuse.types import TraceContext  # type: ignore

                    trace_context = TraceContext(trace_id=trace_id)
                except Exception:
                    trace_context = None

            if hasattr(self._client, "create_event"):
                self._client.create_event(  # type: ignore[attr-defined]
                    trace_context=trace_context,
                    name=name,
                    input=data_inp,
                    output=data_out,
                    metadata=data_meta,
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


