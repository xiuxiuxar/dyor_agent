"""Data models for the simple FSM."""

from datetime import datetime

from pydantic import Field, BaseModel


class AssetInfo(BaseModel):
    """Asset information."""

    name: str
    symbol: str
    category: str | None = None
    coin_id: str | None
    market_cap: float | None
    market_cap_rank: int | None
    contract_address: str | None


class TriggerInfo(BaseModel):
    """Trigger information."""

    type: str
    timestamp: datetime


class KeyMetrics(BaseModel):
    """Key metrics."""

    mindshare_1h: float | None = Field(
        None,
        description="Relative share of social mentions last 1h",
    )
    volume_change_24h: float | None = Field(
        None,
        description="24h trading volume change in %",
    )
    price_change_24h: float | None = Field(
        None,
        description="24h price change in %",
    )


class SocialSummary(BaseModel):
    """Social summary."""

    sentiment_score: float | None = Field(None, description="Aggregated sentiment score (e.g. -1 to +1)")
    recent_mention_count: int | None = Field(None, description="Number of social mentions in the last report")


class NewsItem(BaseModel):
    """News item."""

    timestamp: datetime
    source: str
    headline: str
    snippet: str


class OnchainHighlight(BaseModel):
    """Onchain highlight."""

    timestamp: datetime
    source: str
    event: str
    details: str


class OfficialUpdate(BaseModel):
    """Official update."""

    timestamp: datetime
    source: str
    title: str
    snippet: str


class StructuredPayload(BaseModel):
    """Structured payload."""

    asset_info: AssetInfo
    trigger_info: TriggerInfo
    key_metrics: KeyMetrics
    social_summary: SocialSummary
    recent_news: list[NewsItem]
    onchain_highlights: list[OnchainHighlight]
    official_updates: list[OfficialUpdate]
