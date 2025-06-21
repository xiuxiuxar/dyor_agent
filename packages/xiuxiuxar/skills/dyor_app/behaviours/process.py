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

"""This module contains the implementation of the behaviours of DYOR App skill."""

from typing import Any
from datetime import UTC, tzinfo, datetime, timedelta

import requests
import sqlalchemy
from pydantic import ValidationError
from sqlalchemy import text

from packages.xiuxiuxar.skills.dyor_app.data_models import (
    NewsItem,
    AssetInfo,
    KeyMetrics,
    TriggerInfo,
    SocialSummary,
    OfficialUpdate,
    OnchainHighlight,
    StructuredPayload,
)
from packages.xiuxiuxar.skills.dyor_app.data_sources import DATA_SOURCES, unlocks_project_filter
from packages.xiuxiuxar.skills.dyor_app.behaviours.base import BaseState, DyorabciappEvents, DyorabciappStates


class ProcessDataRound(BaseState):
    """This class implements the behaviour of the state ProcessDataRound."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.PROCESSDATAROUND

    def _get_source_config(self, source: str) -> dict | None:
        """Get configuration for a data source."""
        return DATA_SOURCES.get(source)

    def _process_unlocks_data(
        self,
        raw_data: Any,
        processor_func: callable,
        trigger_id: int,
        asset_id: int,
    ) -> dict[str, str]:
        """Process unlocks data with special filtering logic."""
        all_projects = raw_data
        asset_name = self.context.trigger_context.get("asset_name")
        asset_symbol = self.context.trigger_context.get("asset_symbol")
        coingecko_id = self.context.trigger_context.get("coingecko_id")

        # Log input to unlocks_project_filter
        self.context.logger.info(
            f"Filtering unlocks: asset_name={asset_name}, asset_symbol={asset_symbol}, coingecko_id={coingecko_id} "
            f"all_projects_count={len(all_projects) if isinstance(all_projects, list) else 'N/A'}"
        )

        project, filtered_events = unlocks_project_filter(
            all_projects,
            coingecko_id=coingecko_id,
            asset_name=asset_name,
            symbol=asset_symbol,
        )

        # Log output of unlocks_project_filter
        self.context.logger.info(
            f"unlocks_project_filter result: project={project}, "
            f"filtered_events_count={len(filtered_events) if isinstance(filtered_events, list) else 'N/A'}"
        )

        # Always wrap in a dict for unlocks_data
        unlocks_filtered = {
            "project": project,
            "filtered_events": filtered_events,
        }

        # Defensive: if unlocks_filtered is not a dict, fix it
        if not isinstance(unlocks_filtered, dict):
            unlocks_filtered = {"filtered_events": filtered_events}

        serialized_data = processor_func(unlocks_filtered)
        self.store_processed_data(
            source="unlocks",
            data_type="default",
            data=serialized_data,
            trigger_id=trigger_id,
            asset_id=asset_id,
        )
        return {}

    def _process_standard_data(
        self, source: str, raw_data: Any, processor_func: callable, trigger_id: int, asset_id: int, is_multi: bool
    ) -> dict[str, str]:
        """Process standard (non-unlocks) data sources."""
        return self.process_data_type(
            source=source,
            data=raw_data,
            processor_func=processor_func,
            trigger_id=trigger_id,
            asset_id=asset_id,
            is_multi=is_multi,
        )

    def process_source_data(self, source: str, raw_data: Any, trigger_id: int, asset_id: int) -> dict[str, str]:
        """Process data for a source, handling both single and multi data types."""
        config = self._get_source_config(source)
        if not config:
            return {source: "No processor configured for this source"}

        is_multi = config["data_type_handler"] == "multi"
        processor_func = getattr(self, config["processor"])

        # Special handling for unlocks: filter for project and event type
        if source == "unlocks":
            return self._process_unlocks_data(raw_data, processor_func, trigger_id, asset_id)

        # Default: original logic
        return self._process_standard_data(source, raw_data, processor_func, trigger_id, asset_id, is_multi)

    def _serialize_for_storage(self, data):
        """Recursively serialize data for storage."""
        result = data

        if isinstance(data, dict):
            result = {key: self._serialize_for_storage(value) for key, value in data.items()}
        elif isinstance(data, list):
            result = [self._serialize_for_storage(item) for item in data]
        elif isinstance(data, datetime):
            result = data.isoformat()
        elif isinstance(data, tzinfo):
            result = str(data)
        elif hasattr(data, "model_dump"):
            result = data.model_dump(mode="json")
        elif hasattr(data, "__dict__"):
            result = {k: self._serialize_for_storage(v) for k, v in data.__dict__.items() if not k.startswith("_")}

        return result

    def serialize_unlocks_data(self, data):
        """Serialize unlocks data (ScrapedDataItem) to dict for storage."""
        if hasattr(data, "to_dict"):
            return data.to_dict()
        return data

    def _store_unlocks_raw_data(self, raw: Any, trigger_id: int, asset_id: int) -> None:
        """Store filtered unlocks data as raw data."""
        asset_name = self.context.trigger_context.get("asset_name")
        asset_symbol = self.context.trigger_context.get("asset_symbol")
        coingecko_id = self.context.trigger_context.get("coingecko_id")

        project, filtered_events = unlocks_project_filter(
            raw,
            coingecko_id=coingecko_id,
            asset_name=asset_name,
            symbol=asset_symbol,
        )

        unlocks_filtered = {
            "project": project,
            "filtered_events": filtered_events,
        }

        self.context.logger.debug(
            "Storing filtered unlocks raw data for asset=%s: project=%s, filtered_events_count=%s",
            asset_symbol,
            project,
            len(filtered_events) if isinstance(filtered_events, list) else "N/A",
        )

        self.context.db_model.store_raw_data(
            source="unlocks",
            data_type="raw",
            data=unlocks_filtered,
            trigger_id=trigger_id,
            timestamp=datetime.now(tz=UTC),
            asset_id=asset_id,
        )

    def _store_multi_raw_data(self, source: str, serialized_raw: dict, trigger_id: int, asset_id: int) -> None:
        """Store multi-type raw data."""
        if not isinstance(serialized_raw, dict):
            self.context.logger.warning(
                f"Expected dict for multi data type in source={source}, " f"got {type(serialized_raw)}. Skipping."
            )
            return

        for subkey, subval in serialized_raw.items():
            self.context.logger.info(
                "Storing multi raw data for source=%s, subkey=%s, type(subval)=%s, subval=%s",
                source,
                subkey,
                type(subval),
                repr(subval)[:500],
            )
            self.context.db_model.store_raw_data(
                source=source,
                data_type=subkey,
                data=subval,
                trigger_id=trigger_id,
                timestamp=datetime.now(tz=UTC),
                asset_id=asset_id,
            )

    def _store_single_raw_data(self, source: str, serialized_raw: Any, trigger_id: int, asset_id: int) -> None:
        """Store single-type raw data."""
        self.context.db_model.store_raw_data(
            source=source,
            data_type="raw",
            data=serialized_raw,
            trigger_id=trigger_id,
            timestamp=datetime.now(tz=UTC),
            asset_id=asset_id,
        )

    def _store_all_raw_data(self, trigger_id: int, asset_id: int) -> None:
        """Store all raw data from different sources."""
        try:
            for source, config in DATA_SOURCES.items():
                raw = self.context.raw_data.get(source)

                # Special handling for unlocks: only store filtered data as raw
                if source == "unlocks":
                    self._store_unlocks_raw_data(raw, trigger_id, asset_id)
                    continue

                serialized_raw = self._serialize_for_storage(raw)
                self.context.logger.info(
                    "Storing raw data for source=%s, type=%s: "
                    "type(raw)=%s, type(serialized_raw)=%s, raw=%s, serialized_raw=%s",
                    source,
                    "multi" if config.get("data_type_handler") == "multi" else "raw",
                    type(raw),
                    type(serialized_raw),
                    repr(raw)[:500],
                    repr(serialized_raw)[:500],
                )

                if config.get("data_type_handler") == "multi":
                    self._store_multi_raw_data(source, serialized_raw, trigger_id, asset_id)
                else:
                    self._store_single_raw_data(source, serialized_raw, trigger_id, asset_id)

        except (sqlalchemy.exc.SQLAlchemyError, TypeError, ValueError, AttributeError) as e:
            self.context.logger.warning(f"Failed to store raw data for debugging: {e}")

    def process_data_type(
        self, source: str, data: Any, processor_func: callable, trigger_id: int, asset_id: int, is_multi: bool
    ) -> dict[str, str]:
        """Process data for a source, handling both single and multi data types."""
        errors: dict[str, str] = {}

        if is_multi:
            if not isinstance(data, dict):
                errors[source] = "Expected dict for multi data type"
                return errors
            for data_type, subdata in data.items():
                if not subdata:
                    continue
                try:
                    serialized_data = processor_func(subdata)
                    self.store_processed_data(
                        source=source,
                        data_type=data_type,
                        data=serialized_data,
                        trigger_id=trigger_id,
                        asset_id=asset_id,
                    )
                except (TypeError, ValueError) as e:
                    errors[f"{source}_{data_type}"] = f"Data serialization error: {e}"
                except sqlalchemy.exc.SQLAlchemyError as e:
                    errors[f"{source}_{data_type}"] = f"Database error: {e}"
                except AttributeError as e:
                    errors[f"{source}_{data_type}"] = f"Invalid data structure: {e}"
        else:
            if not data:
                return errors
            try:
                serialized_data = processor_func(data)
                self.store_processed_data(
                    source=source,
                    data_type="default",
                    data=serialized_data,
                    trigger_id=trigger_id,
                    asset_id=asset_id,
                )
            except (TypeError, ValueError) as e:
                errors[source] = f"Data serialization error: {e}"
            except sqlalchemy.exc.SQLAlchemyError as e:
                errors[source] = f"Database error: {e}"
            except AttributeError as e:
                errors[source] = f"Invalid data structure: {e}"

        return errors

    def store_processed_data(self, source: str, data_type: str, data: Any, trigger_id: int, asset_id: int) -> None:
        """Store processed data in the database."""
        try:
            serialized_data = {"source": source, "data": data, "error": None}
            self.context.logger.info(
                "Storing processed data for source=%s, data_type=%s, type(data)=%s, data=%s",
                source,
                data_type,
                type(data),
                repr(data)[:500],
            )
            self.context.db_model.store_raw_data(
                source=source,
                data_type=data_type,
                data=serialized_data,
                trigger_id=trigger_id,
                timestamp=datetime.now(tz=UTC),
                asset_id=asset_id,
            )
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.context.logger.warning(f"Database error storing {source} data: {e}")
            raise
        except (TypeError, ValueError) as e:
            self.context.logger.warning(f"Data serialization error for {source}: {e}")
            raise
        except AttributeError as e:
            self.context.logger.warning(f"Invalid data structure for {source}: {e}")
            raise

    def _handle_processing_error(self, errors: dict, key: str, exc: Exception, error_type: str) -> None:
        """Generic error handler for processing."""
        errors[key] = f"{error_type}: {exc!s}"
        self.context.logger.warning(f"{error_type} for {key}: {exc}")

    def _build_unlock_events(self, unlocks_raw, context):
        unlocks_data = self._normalize_unlocks_data(unlocks_raw)
        events = self._extract_project_events(unlocks_raw, unlocks_data, context)
        unlocks_recent, unlocks_upcoming = self._categorize_unlock_events(events, context)
        if isinstance(unlocks_data, list):
            unlocks_data = {"projects": unlocks_data}
        return unlocks_data, unlocks_recent, unlocks_upcoming

    def _normalize_unlocks_data(self, unlocks_raw):
        if hasattr(unlocks_raw, "to_dict"):
            return unlocks_raw.to_dict()
        if isinstance(unlocks_raw, dict):
            return unlocks_raw
        return str(unlocks_raw)

    def _extract_project_events(self, unlocks_raw, unlocks_data, context):
        # Handle case where unlocks_raw is the all_projects list
        if isinstance(unlocks_raw, list):
            asset_name = context.trigger_context.get("asset_name")
            asset_symbol = context.trigger_context.get("asset_symbol")
            coingecko_id = context.trigger_context.get("coingecko_id")
            project = None
            if coingecko_id:
                project = next((p for p in unlocks_raw if p.get("gecko_id") == coingecko_id), None)
            if not project and asset_name:
                project = next((p for p in unlocks_raw if p.get("name", "").lower() == asset_name.lower()), None)
            if not project and asset_symbol:
                project = next(
                    (p for p in unlocks_raw if p.get("token", "").split(":")[-1].lower() == asset_symbol.lower()),
                    None,
                )
            if project:
                context.logger.info(
                    f"Found project {project.get('name', '')} with {len(project.get('events', []))} events"
                )
                return project.get("events", [])
            context.logger.warning(
                f"Could not find project for asset_name={asset_name}, asset_symbol={asset_symbol}, "
                f"coingecko_id={coingecko_id}"
            )
            return []
        if isinstance(unlocks_data, dict):
            return (
                unlocks_data.get("events")
                or unlocks_data.get("filtered_events", [])
                or unlocks_data.get("metadata", {}).get("raw_emissions", {}).get("meta", {}).get("events", [])
            )
        return []

    def _categorize_unlock_events(self, events, context):
        unlocks_recent = []
        unlocks_upcoming = []
        now = datetime.now(UTC)
        context.logger.info(f"Processing {len(events)} events for recent/upcoming unlock analysis. Current time: {now}")
        for event in events:
            # Filter for cliff unlocks AND specific categories
            if event.get("unlockType") != "cliff" or event.get("category") not in {"insiders", "privateSale"}:
                continue
            try:
                event_date = datetime.fromtimestamp(event["timestamp"], UTC)
                context.logger.debug(
                    f"Processing significant unlock event: date={event_date}, "
                    f"type={event.get('unlockType')}, "
                    f"category={event.get('category')}, "
                    f"tokens={event.get('noOfTokens')}, "
                    f"description={event.get('description')}"
                )
                if now - timedelta(days=14) <= event_date <= now:
                    unlocks_recent.append(event)
                    context.logger.info(f"Added to recent unlocks: {event_date}")
                elif now < event_date <= now + timedelta(days=30):
                    unlocks_upcoming.append(event)
                    context.logger.info(
                        f"Found upcoming significant unlock in {(event_date - now).days} days: "
                        f"{event.get('description')}"
                    )
            except (KeyError, ValueError, TypeError) as e:
                context.logger.warning(f"Failed to parse unlock event timestamp: {e}")
                continue
        context.logger.info(f"Final results: {len(unlocks_recent)} recent, {len(unlocks_upcoming)} upcoming unlocks")
        return unlocks_recent, unlocks_upcoming

    def build_structured_payload(self, context) -> StructuredPayload:
        """Build and validate the StructuredPayload using Pydantic models."""
        trendmoon_raw = context.raw_data.get("trendmoon", {})
        tree_raw = context.raw_data.get("treeofalpha", [])
        look_raw = context.raw_data.get("lookonchain")
        research_raw = context.raw_data.get("researchagent")
        unlocks_raw = context.raw_data.get("unlocks")

        social_data, coin_details, project_summary, topic_summary = self._extract_trendmoon_social_and_coin_details(
            trendmoon_raw
        )
        trend_market_data = self._extract_trend_market_data(social_data)
        unlocks_data, unlocks_recent, unlocks_upcoming = self._build_unlock_events(unlocks_raw, context)

        # --- FIX: Ensure unlocks_data is always a dict ---
        if isinstance(unlocks_data, list):
            unlocks_data = {"projects": unlocks_data}
        elif not isinstance(unlocks_data, dict):
            unlocks_data = {"data": unlocks_data}
        # --- END FIX ---

        # --- NEW: Always pass a dict for unlocks_data ---
        if not (unlocks_recent or unlocks_upcoming):
            unlocks_data = {"summary": "No unlock data available."}
        elif "summary" not in unlocks_data:
            unlocks_data["summary"] = ""
        # --- END NEW ---

        return StructuredPayload(
            asset_info=self._build_asset_info(coin_details, social_data, context),
            trigger_info=TriggerInfo(
                type="manual_test",
                timestamp=datetime.now(tz=UTC).isoformat(),
            ),
            key_metrics=self._build_key_metrics(trend_market_data),
            social_summary=self._build_social_summary(social_data),
            recent_news=self._build_news_items(tree_raw),
            onchain_highlights=self._build_onchain_highlights(research_raw, context),
            official_updates=self._build_official_updates(look_raw),
            project_summary=project_summary,
            topic_summary=topic_summary,
            unlocks_data=None,
            unlocks_recent=unlocks_recent,
            unlocks_upcoming=unlocks_upcoming,
        )

    def _extract_trendmoon_social_and_coin_details(self, trendmoon_raw):
        if isinstance(trendmoon_raw, dict):
            social_data = trendmoon_raw.get("social", {})
            coin_details = trendmoon_raw.get("coin_details", {})
            project_summary = trendmoon_raw.get("project_summary", {})
            topic_summary = trendmoon_raw.get("topic_summary", {})

            if social_data is None:
                social_data = {}
            if coin_details is None:
                coin_details = {}
            if project_summary is None:
                project_summary = {}
            if topic_summary is None:
                topic_summary = {}
        else:
            social_data = {}
            coin_details = {}
            project_summary = {}
            topic_summary = {}
        if isinstance(social_data, list):
            social_data = social_data[0] if social_data else {}
        return social_data, coin_details, project_summary, topic_summary

    def _extract_trend_market_data(self, social_data):
        if social_data is None:
            return []
        return social_data.get("trend_market_data", [])

    def _build_key_metrics(self, trend_market_data):
        def get_latest_two(entries, key):
            filtered = [e for e in entries if key in e and isinstance(e[key], int | float) and e[key] is not None]
            filtered.sort(key=lambda x: datetime.fromisoformat(x["date"]), reverse=True)
            return filtered[:2]

        # Price change 24h
        price_points = get_latest_two(trend_market_data, "price")
        if len(price_points) == 2:
            latest_price = float(price_points[0]["price"])
            previous_price = float(price_points[1]["price"])
            price_change_24h = (
                round(((latest_price - previous_price) / previous_price) * 100, 1) if previous_price else 0.0
            )
        else:
            price_change_24h = 0.0

        # Volume change 24h
        volume_points = get_latest_two(trend_market_data, "total_volume")
        if len(volume_points) == 2:
            latest_volume = float(volume_points[0]["total_volume"])
            previous_volume = float(volume_points[1]["total_volume"])
            volume_change_24h = (
                round(((latest_volume - previous_volume) / previous_volume) * 100, 1) if previous_volume else 0.0
            )
        else:
            volume_change_24h = 0.0

        # Mindshare 24h (latest non-null lc_social_dominance)
        mindshare_24h = 0.0
        mindshare_points = [
            e
            for e in sorted(trend_market_data, key=lambda x: datetime.fromisoformat(x["date"]), reverse=True)
            if "lc_social_dominance" in e
            and isinstance(e["lc_social_dominance"], int | float)
            and e["lc_social_dominance"] is not None
        ]
        if mindshare_points:
            mindshare_24h = round(float(mindshare_points[0]["lc_social_dominance"]), 1)

        # Use the latest (most recent by date) lc_social_dominance for mindshare
        latest_mindshare = 0.0
        if mindshare_points:
            latest_mindshare = mindshare_points[0]["lc_social_dominance"]

        return KeyMetrics(
            mindshare=latest_mindshare,
            mindshare_24h=mindshare_24h,
            volume_change_24h=volume_change_24h,
            price_change_24h=price_change_24h,
        )

    def _build_social_summary(self, social_data):
        if social_data is None:
            social_data = {}
        trend_market_data = social_data.get("trend_market_data", [])
        latest_sentiment = 0.0
        if trend_market_data:
            # Sort by date descending, pick the first with lc_sentiment
            sorted_data = sorted(trend_market_data, key=lambda x: x.get("date", ""), reverse=True)
            for entry in sorted_data:
                if "lc_sentiment" in entry and entry["lc_sentiment"] is not None:
                    latest_sentiment = entry["lc_sentiment"]
                    break

        def get_mention_points(entries):
            # Sort by date descending
            sorted_entries = sorted(entries, key=lambda x: datetime.fromisoformat(x["date"]), reverse=True)
            if not sorted_entries:
                return []
            # If most recent day has 0 mentions, skip it
            if "social_mentions" in sorted_entries[0] and sorted_entries[0]["social_mentions"] == 0:
                sorted_entries = sorted_entries[1:]
            # Take the next two most recent days (even if zero)
            return sorted_entries[:2]

        mention_points = get_mention_points(trend_market_data)
        if len(mention_points) == 2:
            try:
                latest_mentions = float(mention_points[0].get("social_mentions", 0))
                previous_mentions = float(mention_points[1].get("social_mentions", 0))
                if "social_mentions" not in mention_points[0] or "social_mentions" not in mention_points[1]:
                    self.context.logger.warning(f"Missing social mentions in mention_points: {mention_points}")
                if previous_mentions:
                    mention_change_24h = round(((latest_mentions - previous_mentions) / previous_mentions) * 100, 1)
                else:
                    mention_change_24h = 0.0
            except (ValueError, TypeError) as e:
                self.context.logger.warning(f"Failed to parse social mentions: {e}")
                mention_change_24h = 0.0
        else:
            mention_change_24h = 0.0

        return SocialSummary(
            sentiment_score=latest_sentiment,
            top_keywords=social_data.get("symbols", []),
            mention_change_24h=mention_change_24h,
        )

    def _build_asset_info(self, coin_details, social_data, context):
        asset_source = coin_details or social_data or {}
        return AssetInfo(
            name=asset_source.get("name", context.trigger_context.get("asset_name", "")),
            symbol=asset_source.get("symbol", context.trigger_context.get("asset_symbol", "")),
            category=asset_source.get("categories", [None])[0] if asset_source.get("categories") else None,
            coin_id=asset_source.get("id") or asset_source.get("coin_id"),
            market_cap=asset_source.get("market_cap"),
            market_cap_rank=asset_source.get("market_cap_rank"),
            contract_address=asset_source.get("contract_address"),
        )

    def _build_news_items(self, tree_raw):
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

    def _build_official_updates(self, look_raw):
        return [
            OfficialUpdate(
                timestamp=r.timestamp if hasattr(r, "timestamp") else r.get("timestamp"),
                source=r.source if hasattr(r, "source") else r.get("source"),
                title=r.title if hasattr(r, "title") else r.get("title"),
                snippet=r.summary if hasattr(r, "summary") else r.get("summary"),
            )
            for r in look_raw or []
        ]

    def _build_onchain_highlights(self, research_raw, context):
        highlights = []
        tweets = []
        for entry in research_raw or []:
            # Each entry is expected to have a 'data' dict with a 'tweets' list
            if isinstance(entry, dict) and "data" in entry and "tweets" in entry["data"]:
                tweets.extend(entry["data"]["tweets"])
        for tweet in tweets:
            ts = tweet.get("timestamp")
            parsed_ts = self._parse_timestamp(ts, context)
            if not isinstance(parsed_ts, datetime):
                context.logger.warning(f"Skipping OnchainHighlight due to missing or invalid timestamp: {ts!r}")
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

    def _parse_timestamp(self, ts, context):
        if isinstance(ts, str) and ts.isdigit():
            try:
                return datetime.fromtimestamp(int(ts), UTC)
            except (ValueError, TypeError) as e:
                context.logger.warning(f"Failed to parse unix timestamp '{ts}': {e}")
        elif isinstance(ts, int | float):
            try:
                return datetime.fromtimestamp(ts, UTC)
            except (ValueError, TypeError, OSError) as e:
                context.logger.warning(f"Failed to parse numeric timestamp '{ts}': {e}")
        elif isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError) as e:
                context.logger.warning(f"Failed to parse ISO timestamp '{ts}': {e}")
        return None

    def _update_asset_metadata_from_trendmoon(self, asset_id: int) -> None:
        trendmoon_raw = self.context.raw_data.get("trendmoon", {})
        coin_details = trendmoon_raw.get("coin_details", {}) if isinstance(trendmoon_raw, dict) else {}

        if coin_details is None:
            coin_details = {}
        coingecko_id = coin_details.get("id")
        categories = coin_details.get("categories")
        category = categories[0] if categories and isinstance(categories, list) and categories else None
        if coingecko_id or category:
            try:
                with self.context.db_model.engine.connect() as conn:
                    result = conn.execute(
                        text("SELECT coingecko_id, category FROM assets WHERE asset_id = :asset_id"),
                        {"asset_id": asset_id},
                    ).fetchone()
                    current_coingecko_id, current_category = result or (None, None)
                # Only update fields that are currently null
                patch_coingecko_id = coingecko_id if current_coingecko_id is None else current_coingecko_id
                patch_category = category if current_category is None else current_category
                if (current_coingecko_id is None and coingecko_id is not None) or (
                    current_category is None and category is not None
                ):
                    self.context.db_model.update_asset_metadata(asset_id, patch_coingecko_id, patch_category)
            except (sqlalchemy.exc.SQLAlchemyError, TypeError, ValueError) as e:
                self.context.logger.warning(f"Failed to patch asset metadata: {e}")

    def _validate_data_sources(self) -> None:
        """Validate and log missing or errored data sources."""
        expected_sources = set(DATA_SOURCES.keys())
        received_sources = set(self.context.raw_data.keys())
        expected_sources - received_sources

        for source in expected_sources:
            data = self.context.raw_data.get(source)
            if not data:
                error_info = getattr(self.context, "raw_errors", {}).get(source)
                self.context.logger.warning(
                    f"Missing or empty data for source '{source}'."
                    f"{' Error: ' + str(error_info) if error_info else ''}"
                )
                if error_info and error_info.get("http_response"):
                    self.context.logger.warning(f"HTTP response for '{source}': {error_info['http_response']}")

    def _process_and_store_data(self, trigger_id: int, asset_id: int) -> None:
        """Process and store all data for the trigger."""
        self._store_all_raw_data(trigger_id, asset_id)
        self._update_asset_metadata_from_trendmoon(asset_id)

        payload = self._try_build_structured_payload()
        if payload is None:
            msg = "Failed to build structured payload"
            raise ValidationError(msg)
        if not self._store_structured_payload(payload, trigger_id, asset_id):
            msg = "Failed to store structured payload"
            raise RuntimeError(msg)

    def _handle_validation_error(self, e: ValidationError, trigger_id: int, asset_id: int) -> None:
        """Handle validation errors and set error context."""
        self.context.logger.error(f"Validation error: {e}")
        self.context.error_context = {
            "error_type": "validation_error",
            "error_message": str(e),
            "error_source": "payload_validation",
            "trigger_id": trigger_id,
            "asset_id": asset_id,
            "recoverable": False,
        }
        self._event = DyorabciappEvents.ERROR

    def _handle_storage_error(self, e: Exception, trigger_id: int, asset_id: int) -> None:
        """Handle storage errors and set error context."""
        self.context.logger.error(f"Storage error: {e}")
        self.context.error_context = {
            "error_type": "storage_error",
            "error_message": str(e),
            "error_source": "database_operation",
            "trigger_id": trigger_id,
            "asset_id": asset_id,
            "recoverable": True,
        }
        self._event = DyorabciappEvents.ERROR

    def _handle_http_error(self, e: Exception, trigger_id: int, asset_id: int) -> None:
        """Handle HTTP errors and set error context."""
        self.context.logger.error(f"Unexpected error during HTTP request: {e}")
        self.context.error_context = {
            "error_type": "http_error",
            "error_message": str(e),
            "error_source": "http_request",
            "trigger_id": trigger_id,
            "asset_id": asset_id,
        }
        self._event = DyorabciappEvents.ERROR

    def act(self) -> None:
        """Process raw data → validate/structure → store."""
        self.context.logger.info(f"Entering state: {self._state}")
        trigger_id = self.context.trigger_context.get("trigger_id")
        asset_id = self.context.trigger_context.get("asset_id")

        self._validate_data_sources()

        try:
            self._process_and_store_data(trigger_id, asset_id)
            self._event = DyorabciappEvents.DONE

        except ValidationError as e:
            self._handle_validation_error(e, trigger_id, asset_id)

        except (sqlalchemy.exc.SQLAlchemyError, RuntimeError) as e:
            self._handle_storage_error(e, trigger_id, asset_id)

        except (requests.RequestException, requests.Timeout, requests.ConnectionError) as e:
            self._handle_http_error(e, trigger_id, asset_id)

        finally:
            self._is_done = True

    def _try_build_structured_payload(self):
        """Build the payload, catch and log all validation errors in detail."""
        try:
            return self.build_structured_payload(self.context)
        except ValidationError as e:
            validation_errors = []
            for err in e.errors():
                loc = ".".join(str(path_item) for path_item in err.get("loc", []))
                msg = err.get("msg", "Unknown error")
                typ = err.get("type", "Unknown type")
                validation_errors.append({"location": loc, "message": msg, "error_type": typ})
                self.context.logger.exception(f"Validation error at {loc}: {msg} (type: {typ})")

            self.context.error_context = {
                "error_type": "validation_error",
                "error_source": "payload_schema",
                "validation_errors": validation_errors,
                "raw_data_sample": self._get_safe_data_sample(),
            }
            msg = "Payload validation failed"
            raise ValidationError(msg) from e

    def _get_safe_data_sample(self):
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

    def _store_structured_payload(self, payload, trigger_id, asset_id):
        """Store the structured payload in the database."""
        try:
            self.context.logger.info(
                f"Storing structured payload with unlocks:\n"
                f"Recent: {len(payload.unlocks_recent)} events\n"
                f"Upcoming: {len(payload.unlocks_upcoming)} events"
            )
            payload_dict = payload.model_dump(mode="json")
            self.context.db_model.store_raw_data(
                source="structured",
                data_type="default",
                data=payload_dict,
                trigger_id=trigger_id,
                timestamp=datetime.now(tz=UTC),
                asset_id=asset_id,
            )
            return True
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.context.logger.exception(f"Database error: {e}")
            self.context.error_context = {
                "error_type": "database_error",
                "error_source": "store_payload",
                "error_message": str(e),
                "query_info": "store_raw_data",
            }
            msg = f"Database error while storing structured payload: {e}"
            raise RuntimeError(msg) from e
        except TypeError as e:
            self.context.logger.exception(f"Serialization error: {e}")
            self.context.error_context = {
                "error_type": "serialization_error",
                "error_source": "payload_serialization",
                "error_message": str(e),
            }
            msg = f"Failed to serialize payload: {e}"
            raise RuntimeError(msg) from e
