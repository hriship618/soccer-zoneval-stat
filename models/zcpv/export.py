from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def research_export(
    path: Path,
    *,
    model_version: str,
    config_hash: str,
    cutoff: str | None,
    status: str,
    reason: str | None,
    players: list[dict[str, Any]] | None = None,
    data_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0.0",
        "model_version": model_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "profile_cutoff": cutoff,
        "status": status,
        "reason": reason,
        "units": "non-penalty expected goals impact per 90",
        "reference": "exposure-weighted average player in the training competition",
        "players": players if status == "available" else None,
        "data_summary": data_summary or {},
        "limitations": [
            "v1 estimates association under regularized lineup and context controls, not causal player value",
            "goalkeepers are retained as lineup controls but excluded from public outfield rankings",
            "upstream situation-model uncertainty is not included unless explicitly reported",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
