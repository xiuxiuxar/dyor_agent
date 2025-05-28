# ------------------------------------------------------------------------------
#
#   Copyright 2025 xiuxiuxar
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""This module contains the Pydantic models for the DYOR API."""

from typing import Any
from datetime import datetime

from pydantic import Field, BaseModel, root_validator


class AssetBase(BaseModel):
    """Base asset model."""

    symbol: str = Field(..., min_length=1, max_length=32)
    name: str = Field(..., min_length=1, max_length=128)
    coingecko_id: str | None = Field(None, max_length=128)
    category: str | None = Field(None, max_length=64)


class AssetCreate(AssetBase):
    """Asset creation model."""


class AssetResponse(AssetBase):
    """Asset response model."""

    asset_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        """Config for the asset response model."""

        from_attributes = True


class ReportBase(BaseModel):
    """Base report model."""

    asset_id: int
    trigger_id: int
    report_content_markdown: str
    report_data_json: dict[str, Any] | None = None
    llm_model_used: str | None = Field(None, max_length=128)
    generation_time_ms: int | None = None
    token_usage: dict[str, Any] | None = None


class ReportCreate(ReportBase):
    """Report creation model."""


class ReportResponse(ReportBase):
    """Report response model."""

    report_id: int
    created_at: datetime

    class Config:
        """Config for the report response model."""

        from_attributes = True


class TriggerBase(BaseModel):
    """Base trigger model."""

    asset_id: int | None = None
    asset_symbol: str | None = None
    trigger_type: str = Field(..., max_length=64)
    trigger_details: dict[str, Any] | None = None
    status: str = Field(default="pending", max_length=32)
    report_id: int | None = None

    @root_validator(pre=True)
    @classmethod
    def check_asset_id_or_symbol(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check that either asset_id or asset_symbol is provided."""
        if isinstance(values, dict):
            asset_id = values.get("asset_id")
            asset_symbol = values.get("asset_symbol")
            if asset_id is None and not asset_symbol:
                msg = "Either asset_id or asset_symbol must be provided."
                raise ValueError(msg)
        return values


class TriggerCreate(TriggerBase):
    """Trigger creation model."""


class TriggerResponse(TriggerBase):
    """Trigger response model."""

    trigger_id: int
    created_at: datetime
    processing_started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    report_id: int | None = None

    class Config:
        """Config for the trigger response model."""

        from_attributes = True
