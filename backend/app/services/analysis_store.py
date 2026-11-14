"""
analysis_store.py — Thread-safe, memory-bounded cache for verified analyses.

Stores complete intermediate and final results for retrieval via:
- GET /api/v1/analysis/{analysis_id}
- GET /api/v1/analysis/{analysis_id}/stage/{stage}
- GET /api/v1/analysis/{analysis_id}/explain/{feature}

Does not write sensitive image bytes or handwriting samples to disk.
"""
from __future__ import annotations

import time
import threading
from typing import Any
from collections import OrderedDict

from app.core.config import settings


class AnalysisStore:
    def __init__(self, max_size: int = settings.max_cached_analyses, ttl_minutes: int = settings.analysis_cache_ttl_minutes):
        self._max_size = max_size
        self._ttl_seconds = ttl_minutes * 60
        self._store: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def _evict_expired(self, now: float) -> None:
        """Evict records older than TTL."""
        expired = [
            k for k, (ts, _) in self._store.items()
            if now - ts > self._ttl_seconds
        ]
        for k in expired:
            self._store.pop(k, None)

    def save(self, analysis_id: str, data: dict[str, Any]) -> None:
        """Save analysis record with current timestamp."""
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)
            self._store[analysis_id] = (now, data)

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        """Retrieve analysis record if valid and not expired."""
        now = time.time()
        with self._lock:
            if analysis_id not in self._store:
                return None
            ts, data = self._store[analysis_id]
            if now - ts > self._ttl_seconds:
                self._store.pop(analysis_id, None)
                return None
            # Refresh LRU position
            self._store.move_to_end(analysis_id)
            return data

    def get_stage(self, analysis_id: str, stage_name: str) -> dict[str, Any] | None:
        """Retrieve stage details for an analysis."""
        analysis = self.get(analysis_id)
        if not analysis:
            return None
        stages = analysis.get("pipeline_stages", {})
        if stage_name in stages:
            stage_info = stages[stage_name]
            if hasattr(stage_info, "model_dump"):
                data = stage_info.model_dump()
            elif isinstance(stage_info, dict):
                data = stage_info
            else:
                data = {
                    "status": getattr(stage_info, "status", "completed"),
                    "execution_time_ms": getattr(stage_info, "execution_time_ms", None),
                    "message": getattr(stage_info, "message", None),
                    "details": getattr(stage_info, "details", {}),
                }
            return {
                "analysis_id": analysis_id,
                "stage": stage_name,
                "status": data.get("status", "completed"),
                "execution_time_ms": data.get("execution_time_ms"),
                "message": data.get("message"),
                "details": data.get("details", {}),
            }
        return None

    def get_feature_explanation(self, analysis_id: str, feature_name: str) -> dict[str, Any] | None:
        """Retrieve explanation for a specific feature from the analysis."""
        analysis = self.get(analysis_id)
        if not analysis:
            return None
        explanations = analysis.get("explanations", [])
        for exp in explanations:
            feat = getattr(exp, "feature", None) or (exp.get("feature") if isinstance(exp, dict) else None)
            name = getattr(exp, "name", None) or (exp.get("name") if isinstance(exp, dict) else None)
            if feat == feature_name or name == feature_name:
                exp_dict = exp.model_dump() if hasattr(exp, "model_dump") else exp
                return {
                    "analysis_id": analysis_id,
                    "feature": feature_name,
                    "explanation": exp_dict,
                }
        return None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# Global singleton instance
analysis_store = AnalysisStore()
