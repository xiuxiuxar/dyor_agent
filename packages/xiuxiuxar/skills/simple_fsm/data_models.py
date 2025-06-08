"""Data models for the simple FSM."""

from datetime import UTC, datetime

from pydantic import Field, BaseModel


class ProjectSummary(BaseModel):
    """Project summary information."""

    coin_id: str | None = None
    name: str | None = None
    symbol: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    summary: str | None = None
    created_at: datetime | None = None


class RecentSentiment(BaseModel):
    """Recent sentiment information."""

    sentiment: str | None = None
    bullish: str | None = None
    bearish: str | None = None


class TopicSummary(BaseModel):
    """Topic summary information."""

    overview: str | None = None
    recent_sentiment: RecentSentiment | None = None
    developments_and_catalysts: str | None = None
    full_report: str | None = None
    topic: str | None = None
    generated_at: datetime | None = None


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

    mindshare: float | None = Field(
        None,
        description="Latest mindshare",
    )

    mindshare_24h: float | None = Field(
        None,
        description="24h mindshare change in %",
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
    mention_change_24h: float | None = Field(None, description="24h change in social mentions in %")


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


class UnlockEvent(BaseModel):
    """Unlock event."""

    category: str
    timestamp: int
    noOfTokens: list[float]  # noqa: N815
    unlockType: str  # noqa: N815
    description: str

    @property
    def formatted_date(self) -> str:
        """Return formatted date string from timestamp."""
        return datetime.fromtimestamp(self.timestamp, UTC).strftime("%Y-%m-%d")

    @property
    def formatted_amount(self) -> str:
        """Return formatted token amount."""
        if not self.noOfTokens:
            return "0"
        amount = self.noOfTokens[0] if isinstance(self.noOfTokens, list) else self.noOfTokens
        return f"{amount:,.2f}"

    def format_description(self) -> str:
        """Return formatted description with actual values."""
        try:
            return self.description.format(timestamp=self.formatted_date, tokens=self.noOfTokens)
        except (KeyError, ValueError, TypeError):
            return self.description


class StructuredPayload(BaseModel):
    """Structured payload."""

    asset_info: AssetInfo
    trigger_info: TriggerInfo
    key_metrics: KeyMetrics
    social_summary: SocialSummary
    recent_news: list[NewsItem]
    onchain_highlights: list[OnchainHighlight]
    official_updates: list[OfficialUpdate]
    project_summary: ProjectSummary | None = None
    topic_summary: TopicSummary | None = None
    unlocks_data: dict | None = None  # Raw unlocks data for reporting/LLM
    unlocks_recent: list[UnlockEvent] = []  # Recent unlocks data for reporting/LLM
    unlocks_upcoming: list[UnlockEvent] = []  # Upcoming unlocks data for reporting/LLM
