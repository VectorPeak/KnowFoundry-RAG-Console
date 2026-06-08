"""LangSmith adapter for enterprise tracing and evaluation metadata.

The project keeps business-specific RAG logic locally and delegates generic
LLMOps platform concerns to LangSmith. Runtime code calls this module with
domain metadata; LangSmith owns storage, UI, datasets, annotation, and
experiment comparison.
"""

from __future__ import annotations

import os
from typing import Any

from langsmith.run_helpers import trace

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings


logger = get_logger(__name__)


def langsmith_enabled() -> bool:
    """Return whether LangSmith tracing is configured for this process."""
    settings = get_settings()
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)


def configure_langsmith_environment() -> None:
    """Populate LangSmith environment variables for LangChain integrations."""
    settings = get_settings()
    os.environ.setdefault("LANGSMITH_TRACING", "true" if settings.langsmith_tracing else "false")
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)


def langsmith_status() -> dict[str, Any]:
    """Return a lightweight status payload for the local admin page."""
    settings = get_settings()
    return {
        "provider": "langsmith",
        "enabled": langsmith_enabled(),
        "project": settings.langsmith_project,
        "endpoint": settings.langsmith_endpoint,
        "has_api_key": bool(settings.langsmith_api_key),
        "project_url": "https://smith.langchain.com/",
        "message": (
            "LangSmith tracing is enabled."
            if langsmith_enabled()
            else "Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable enterprise tracing."
        ),
    }


def _safe_preview(text: str, max_chars: int = 800) -> str:
    return (text or "")[:max_chars]


def record_query_trace(
    *,
    trace_id: str,
    session_id: str,
    question: str,
    answer: str,
    hit_type: str,
    scenario,
    data_scope: dict[str, Any],
    source_filter: str | None,
    kb_version: str,
    rewritten_query: str | None,
    intent: dict[str, Any] | None,
    retrieval: dict[str, Any] | None,
    sources: list[dict[str, Any]],
    elapsed_ms: float,
    error: str | None = None,
) -> None:
    """Record one QA turn to LangSmith.

    Tracing must never affect the user request. If LangSmith is disabled or the
    network call fails, this function logs and returns.
    """
    configure_langsmith_environment()
    if not langsmith_enabled():
        return

    retrieval_payload = retrieval or {}
    intent_payload = intent or {}
    metadata = {
        "trace_id": trace_id,
        "session_id": session_id,
        "scenario_id": getattr(scenario, "scenario_id", ""),
        "scenario_name": getattr(scenario, "display_name", ""),
        "source_filter": source_filter,
        "effective_source": retrieval_payload.get("source_filter") or source_filter,
        "kb_version": kb_version,
        "tenant_id": data_scope.get("tenant_id"),
        "dataset_id": data_scope.get("dataset_id"),
        "visibility": data_scope.get("visibility"),
        "user_role": data_scope.get("user_role"),
        "allowed_roles": data_scope.get("allowed_roles"),
        "intent": intent_payload.get("intent"),
        "intent_reason": intent_payload.get("reason"),
        "hit_type": hit_type,
        "prompt_profile": retrieval_payload.get("prompt_profile_name"),
        "question_category": retrieval_payload.get("question_category"),
        "rewritten_query": rewritten_query,
        "sources_count": len(sources),
        "top_source_score": sources[0].get("score") if sources else None,
        "first_token_ms": retrieval_payload.get("first_token_ms"),
        "stage_timings_ms": retrieval_payload.get("stage_timings_ms"),
        "slowest_stage": retrieval_payload.get("slowest_stage"),
        "elapsed_ms": round(elapsed_ms, 2),
        "error": error,
    }
    inputs = {
        "question": question,
        "scenario_id": metadata["scenario_id"],
        "source_filter": source_filter,
        "kb_version": kb_version,
    }
    outputs = {
        "answer_preview": _safe_preview(answer),
        "hit_type": hit_type,
        "sources": sources[:8],
        "error": error,
    }

    try:
        with trace(
            "qa_stream_query",
            run_type="chain",
            inputs=inputs,
            metadata=metadata,
            project_name=get_settings().langsmith_project,
            run_id=trace_id,
            tags=[str(metadata["scenario_id"]), hit_type],
        ) as run_tree:
            if error:
                run_tree.end(outputs=outputs, error=error)
            else:
                run_tree.end(outputs=outputs)
    except Exception as exc:  # pragma: no cover - tracing must be best effort
        logger.warning("LangSmith trace failed: %s", exc)
