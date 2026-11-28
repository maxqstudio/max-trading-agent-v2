"""Research Lab API endpoints: historical bar coverage, MT5 CSV/TSV import, and offline DL/ML pre-training + OOS audit."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.engine.orchestrator import TradingOrchestrator

MAX_CSV_CHARS = 15_000_000  # ~15 MB per uploaded MT5 History Center CSV/TSV


class ResearchRunRequest(BaseModel):
    window_bars: int | None = Field(default=None, ge=300, le=100_000)
    dry_run: bool = False
    force: bool = False


class ResearchCsvImportRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)


def register_research_routes(api: APIRouter, orchestrator: TradingOrchestrator) -> None:
    @api.get("/research")
    async def get_research() -> dict[str, Any]:
        return await orchestrator.get_research_status()

    @api.post("/research/run")
    async def run_research(req: ResearchRunRequest) -> dict[str, Any]:
        return await orchestrator.run_research_pretrain(
            window_bars=req.window_bars,
            dry_run=req.dry_run,
            force=req.force,
        )

    @api.post("/research/import-csv")
    async def import_research_csv(req: ResearchCsvImportRequest) -> dict[str, Any]:
        if len(req.content) > MAX_CSV_CHARS:
            raise HTTPException(status_code=413, detail="CSV file too large (max 15 MB per upload)")
        try:
            return await orchestrator.import_research_csv(req.filename, req.content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
