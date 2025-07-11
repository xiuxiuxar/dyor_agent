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

"""Payload builder for DYOR App skill."""

from datetime import UTC, datetime

from pydantic import ValidationError

from packages.xiuxiuxar.skills.dyor_app.utils import (
    safe_get_nested,
    get_asset_identifiers,
    parse_timestamp_safely,
)
from packages.xiuxiuxar.skills.dyor_app.data_models import (
    NewsItem,
    AssetInfo,
    TriggerInfo,
    OfficialUpdate,
    OnchainHighlight,
    StructuredPayload,
)


class PayloadBuilder:
    """Handles structured payload assembly from processed data."""

    def __init__(self, context, logger):
        """Initialize with context and logger."""
        self.context = context
        self.logger = logger

    def build_structured_payload(self, metrics_calculator, unlock_processor) -> StructuredPayload:
        """Build and validate the StructuredPayload using Pydantic models."""
        raw_sources = self.extract_raw_data_sources()

        social_data, coin_details, project_summary, topic_summary = self.extract_trendmoon_social_and_coin_details(
            raw_sources["trendmoon"]
        )
        trend_market_data = self.extract_trend_market_data(social_data)
        unlocks_data, unlocks_recent, unlocks_upcoming = unlock_processor.build_unlock_events(raw_sources["unlocks"])

        prepared_unlocks_data = unlock_processor.prepare_unlocks_data(unlocks_data, unlocks_recent, unlocks_upcoming)

        return StructuredPayload(
            asset_info=self.build_asset_info(coin_details, social_data),
            trigger_info=TriggerInfo(
                type="manual_test",
                timestamp=datetime.now(tz=UTC).isoformat(),
            ),
            key_metrics=metrics_calculator.build_key_metrics(trend_market_data),
            social_summary=metrics_calculator.build_social_summary(social_data),
            recent_news=self.build_news_items(raw_sources["tree"]),
            onchain_highlights=self.build_onchain_highlights(raw_sources["research"]),
            official_updates=self.build_official_updates(raw_sources["look"]),
            project_summary=project_summary,
            topic_summary=topic_summary,
            unlocks_data=prepared_unlocks_data,
            unlocks_recent=unlocks_recent,
            unlocks_upcoming=unlocks_upcoming,
        )

    def extract_trendmoon_social_and_coin_details(self, trendmoon_raw):
        """Extract social data and coin details from Trendmoon raw data."""
        if not isinstance(trendmoon_raw, dict):
            return {}, {}, {}, {}

        social_data = trendmoon_raw.get("social") or {}
        coin_details = trendmoon_raw.get("coin_details") or {}
        project_summary = trendmoon_raw.get("project_summary") or {}
        topic_summary = trendmoon_raw.get("topic_summary") or {}

        # Handle case where social_data is a list
        if isinstance(social_data, list):
            social_data = social_data[0] if social_data else {}

        return social_data, coin_details, project_summary, topic_summary

    def extract_trend_market_data(self, social_data):
        """Extract trend market data from social data."""
        if social_data is None:
            return []
        return social_data.get("trend_market_data", [])

    def extract_raw_data_sources(self) -> dict:
        """Extract raw data from different sources."""
        return {
            "trendmoon": self.context.raw_data.get("trendmoon", {}),
            "tree": self.context.raw_data.get("treeofalpha", []),
            "look": self.context.raw_data.get("lookonchain"),
            "research": self.context.raw_data.get("researchagent"),
            "unlocks": self.context.raw_data.get("unlocks"),
        }

    def build_asset_info(self, coin_details, social_data) -> AssetInfo:
        """Build asset info from coin details and social data."""
        asset_source = coin_details or social_data or {}
        identifiers = get_asset_identifiers(self.context)

        return AssetInfo(
            name=asset_source.get("name", identifiers["asset_name"] or ""),
            symbol=asset_source.get("symbol", identifiers["asset_symbol"] or ""),
            category=safe_get_nested(asset_source, "categories", 0),
            coin_id=asset_source.get("id") or asset_source.get("coin_id"),
            market_cap=asset_source.get("market_cap"),
            market_cap_rank=asset_source.get("market_cap_rank"),
            contract_address=asset_source.get("contract_address"),
        )

    def build_news_items(self, tree_raw) -> list[NewsItem]:
        """Build news items from TreeOfAlpha raw data."""
        if not tree_raw:
            return []
        return [
            NewsItem(
                headline=r.get("title", ""),
                snippet=r.get("content", r.get("title", "")),
                url=r.get("url", ""),
                timestamp=datetime.fromtimestamp(r["time"] / 1000, UTC).isoformat() if r.get("time") else None,
                source=r.get("source", ""),
            )
            for r in tree_raw
        ]

    def build_official_updates(self, look_raw) -> list[OfficialUpdate]:
        """Build official updates from LookOnChain raw data."""
        return [
            OfficialUpdate(
                timestamp=r.timestamp if hasattr(r, "timestamp") else r.get("timestamp"),
                source=r.source if hasattr(r, "source") else r.get("source"),
                title=r.title if hasattr(r, "title") else r.get("title"),
                snippet=r.summary if hasattr(r, "summary") else r.get("summary"),
            )
            for r in look_raw or []
        ]

    def build_onchain_highlights(self, research_raw) -> list[OnchainHighlight]:
        """Build onchain highlights from research agent raw data."""
        highlights = []
        tweets = []
        for entry in research_raw or []:
            # Each entry is expected to have a 'data' dict with a 'tweets' list
            if isinstance(entry, dict) and "data" in entry and "tweets" in entry["data"]:
                tweets.extend(entry["data"]["tweets"])
        for tweet in tweets:
            ts = tweet.get("timestamp")
            parsed_ts = parse_timestamp_safely(ts, self.logger)
            if not isinstance(parsed_ts, datetime):
                self.logger.warning(f"Skipping OnchainHighlight due to missing or invalid timestamp: {ts!r}")
                continue
            highlights.append(
                OnchainHighlight(
                    timestamp=parsed_ts,
                    source=tweet.get("account", "researchagent"),
                    headline=tweet.get("html", ""),
                    snippet="",
                    details=tweet.get("html", ""),
                    event="tweet",
                )
            )
        return highlights

    def try_build_structured_payload(self, metrics_calculator, unlock_processor):
        """Build the payload, catch and log all validation errors in detail."""
        try:
            return self.build_structured_payload(metrics_calculator, unlock_processor)
        except ValidationError as e:
            validation_errors = []
            for err in e.errors():
                loc = ".".join(str(path_item) for path_item in err.get("loc", []))
                msg = err.get("msg", "Unknown error")
                typ = err.get("type", "Unknown type")
                validation_errors.append({"location": loc, "message": msg, "error_type": typ})
                self.logger.exception(f"Validation error at {loc}: {msg} (type: {typ})")

            self.context.error_context = {
                "error_type": "validation_error",
                "error_source": "payload_schema",
                "validation_errors": validation_errors,
                "raw_data_sample": self.get_safe_data_sample(),
            }
            msg = "Payload validation failed"
            raise ValidationError(msg) from e

    def get_safe_data_sample(self):
        """Create a safe sample of raw data for error context."""
        try:
            # Create a sample that won't blow up logs but provides context
            sample = {}
            for source, data in self.context.raw_data.items():
                if isinstance(data, dict):
                    sample[source] = dict.fromkeys(data.keys(), "present")
                elif isinstance(data, list):
                    sample[source] = f"list with {len(data)} items"
                else:
                    sample[source] = str(type(data))
            return sample
        except (TypeError, AttributeError, KeyError):
            return "Could not sample raw data"
