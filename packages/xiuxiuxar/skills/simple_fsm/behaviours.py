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

"""This package contains a simple FSM behaviour."""

import os
import re
import sys
import json
import time
import random
import importlib
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Any, cast
from pathlib import Path
from datetime import UTC, tzinfo, datetime, timedelta
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

import yaml
import markdown
import requests
import sqlalchemy
from pydantic import ValidationError
from sqlalchemy import text
from aea.skills.behaviours import State, FSMBehaviour

from packages.xiuxiuxar.skills.simple_fsm.models import LLMServiceError
from packages.xiuxiuxar.skills.simple_fsm.prompt import build_report_prompt
from packages.xiuxiuxar.skills.simple_fsm.data_models import (
    NewsItem,
    AssetInfo,
    KeyMetrics,
    TriggerInfo,
    SocialSummary,
    OfficialUpdate,
    OnchainHighlight,
    StructuredPayload,
)
from packages.xiuxiuxar.skills.simple_fsm.data_sources import DATA_SOURCES, unlocks_fetcher, unlocks_project_filter
from packages.xiuxiuxar.skills.simple_fsm.trendmoon_client import TrendmoonAPIError
from packages.xiuxiuxar.skills.simple_fsm.lookonchain_client import LookOnChainAPIError
from packages.xiuxiuxar.skills.simple_fsm.treeofalpha_client import TreeOfAlphaAPIError
from packages.xiuxiuxar.skills.simple_fsm.researchagent_client import ResearchAgentAPIError


if TYPE_CHECKING:
    from packages.xiuxiuxar.skills.simple_fsm.models import APIClientStrategy


PROTOCOL_HTTP = "eightballer/http:0.1.0"
PROTOCOL_WEBSOCKETS = "eightballer/websockets:0.1.0"
PROTOCOL_HANDLER_MAP = {
    PROTOCOL_HTTP: "http_handlers",
    PROTOCOL_WEBSOCKETS: "ws_handlers",
}


def dynamic_import(component_name, module_name):
    """Dynamically import a module."""

    module = importlib.import_module(component_name)
    return getattr(module, module_name)


class DyorabciappEvents(Enum):
    """Events for the fsm."""

    NO_TRIGGER = "NO_TRIGGER"
    DONE = "DONE"
    RETRY = "RETRY"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    TRIGGER = "TRIGGER"


class DyorabciappStates(Enum):
    """States for the fsm."""

    WATCHINGROUND = "watchinground"
    PROCESSDATAROUND = "processdataround"
    SETUPDYORROUND = "setupdyorround"
    DELIVERREPORTROUND = "deliverreportround"
    TRIGGERROUND = "triggerround"
    INGESTDATAROUND = "ingestdataround"
    GENERATEREPORTROUND = "generatereportround"
    HANDLEERRORROUND = "handleerrorround"


class BaseState(State, ABC):
    """Base class for states."""

    _state: DyorabciappStates = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._event = None
        self._is_done = False

    def act(self) -> None:
        """Perform the act."""
        self._is_done = True
        self._event = DyorabciappEvents.DONE

    def is_done(self) -> bool:
        """Is done."""
        return self._is_done

    @property
    def event(self) -> str | None:
        """Current event."""
        return self._event


class WatchingRound(BaseState):
    """This class implements the behaviour of the state WatchingRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.WATCHINGROUND

    def act(self) -> None:
        """Act:
        1. Check if there are any triggers in the database.
        2. If there are, set the event to TRIGGER.
        3. If there are no triggers, set the event to NO_TRIGGER.
        """
        self.context.logger.debug(f"Entering state: {self._state}")

        try:
            # Query for unprocessed triggers
            with self.context.db_model.engine.begin() as conn:
                row = conn.execute(
                    text("""
                    WITH next AS (
                        SELECT trigger_id
                        FROM   triggers
                        WHERE  status = 'pending'
                        ORDER  BY created_at
                        LIMIT  1
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE triggers AS t
                    SET    status = 'processing',
                        processing_started_at = NOW()
                    FROM   next
                    WHERE  t.trigger_id = next.trigger_id
                    RETURNING t.trigger_id,
                            t.asset_id,
                            (SELECT symbol FROM assets WHERE asset_id = t.asset_id)   AS symbol,
                            (SELECT name   FROM assets WHERE asset_id = t.asset_id)   AS name,
                            t.trigger_details
                """)
                ).fetchone()

                if row:
                    # Found a trigger, set up context and transition
                    self.context.trigger_context = {
                        "trigger_id": row.trigger_id,
                        "asset_id": row.asset_id,
                        "asset_symbol": row.symbol,
                        "asset_name": row.name,
                        "trigger_details": row.trigger_details,
                    }
                    self.context.logger.info(
                        f"Found trigger {row.trigger_id} for asset {row.symbol} - set to processing"
                    )
                    self._event = DyorabciappEvents.TRIGGER
                else:
                    self.context.logger.debug("No pending triggers found")
                    self._event = DyorabciappEvents.NO_TRIGGER

        except sqlalchemy.exc.SQLAlchemyError as e:
            self.context.logger.exception(f"Database error while checking triggers: {e}")
            self._event = DyorabciappEvents.ERROR

        self._is_done = True


class ProcessDataRound(BaseState):
    """This class implements the behaviour of the state ProcessDataRound."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.PROCESSDATAROUND

    def process_source_data(self, source: str, raw_data: Any, trigger_id: int, asset_id: int) -> dict[str, str]:
        """Process data for a source, handling both single and multi data types."""
        config = DATA_SOURCES.get(source)
        if not config:
            return {source: "No processor configured for this source"}
        is_multi = config["data_type_handler"] == "multi"
        processor_func = getattr(self, config["processor"])
        # Special handling for unlocks: filter for project and event type
        if source == "unlocks":
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
                source=source,
                data_type="default",
                data=serialized_data,
                trigger_id=trigger_id,
                asset_id=asset_id,
            )
            return {}
        # Default: original logic
        return self.process_data_type(
            source=source,
            data=raw_data,
            processor_func=processor_func,
            trigger_id=trigger_id,
            asset_id=asset_id,
            is_multi=is_multi,
        )

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

    def _store_all_raw_data(self, trigger_id, asset_id):
        try:
            for source, config in DATA_SOURCES.items():
                raw = self.context.raw_data.get(source)
                # Special handling for unlocks: only store filtered data as raw
                if source == "unlocks":
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
                        source=source,
                        data_type="raw",
                        data=unlocks_filtered,
                        trigger_id=trigger_id,
                        timestamp=datetime.now(tz=UTC),
                        asset_id=asset_id,
                    )
                    continue  # skip default logic for unlocks
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
                    if not isinstance(serialized_raw, dict):
                        self.context.logger.warning(
                            f"Expected dict for multi data type in source={source}, "
                            f"got {type(serialized_raw)}. Skipping."
                        )
                        continue
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
                else:
                    self.context.db_model.store_raw_data(
                        source=source,
                        data_type="raw",
                        data=serialized_raw,
                        trigger_id=trigger_id,
                        timestamp=datetime.now(tz=UTC),
                        asset_id=asset_id,
                    )
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

    def act(self) -> None:
        """Process raw data → validate/structure → store."""
        self.context.logger.info(f"Entering state: {self._state}")
        trigger_id = self.context.trigger_context.get("trigger_id")
        asset_id = self.context.trigger_context.get("asset_id")

        # Log missing or errored sources
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

        try:
            self._store_all_raw_data(trigger_id, asset_id)

            self._update_asset_metadata_from_trendmoon(asset_id)

            payload = self._try_build_structured_payload()
            if payload is None:
                msg = "Failed to build structured payload"
                raise ValidationError(msg)
            if not self._store_structured_payload(payload, trigger_id, asset_id):
                msg = "Failed to store structured payload"
                raise RuntimeError(msg)
            self._event = DyorabciappEvents.DONE

        except ValidationError as e:
            self.context.logger.exception(f"Validation error: {e}")
            self.context.error_context = {
                "error_type": "validation_error",
                "error_message": str(e),
                "error_source": "payload_validation",
                "trigger_id": trigger_id,
                "asset_id": asset_id,
                "recoverable": False,
            }
            self._event = DyorabciappEvents.ERROR

        except (sqlalchemy.exc.SQLAlchemyError, RuntimeError) as e:
            self.context.logger.exception(f"Storage error: {e}")
            self.context.error_context = {
                "error_type": "storage_error",
                "error_message": str(e),
                "error_source": "database_operation",
                "trigger_id": trigger_id,
                "asset_id": asset_id,
                "recoverable": True,
            }
            self._event = DyorabciappEvents.ERROR

        except (requests.RequestException, requests.Timeout, requests.ConnectionError) as e:
            self.context.logger.exception(f"Unexpected error during HTTP request: {e}")
            self.context.error_context = {
                "error_type": "http_error",
                "error_message": str(e),
                "error_source": "http_request",
                "trigger_id": trigger_id,
                "asset_id": asset_id,
            }
            self._event = DyorabciappEvents.ERROR

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


class SetupDYORRound(BaseState):
    """This class implements the behaviour of the state SetupDYORRound."""

    def __init__(self, **kwargs: Any) -> None:
        self.api_name = kwargs.pop("api_name", None)
        super().__init__(**kwargs)
        self.context.logger.info(f"API name: {self.api_name}")
        self._state = DyorabciappStates.SETUPDYORROUND

    @property
    def strategy(self) -> str | None:
        """Get the strategy."""
        return cast("APIClientStrategy", self.context.api_client_strategy)

    @property
    def custom_api_component_info(self) -> tuple[str, str, Path, dict[str, Any]]:
        """Check load of custom API component."""
        try:
            author, component_name = self.api_name.split("/")
            directory = Path("vendor") / author / "customs" / component_name
            config_path = directory / "component.yaml"

            if not config_path.exists():
                msg = f"Component config file not found: {config_path}"
                raise FileNotFoundError(msg)

            config = yaml.safe_load(config_path.read_text())
            return author, component_name, directory, config
        except (ValueError, FileNotFoundError, yaml.YAMLError) as e:
            self.context.logger.exception(f"Error getting custom API component info: {e}")
            raise

    API_CLIENT_CONFIGS = {
        "trendmoon": {
            "client_attr": "trendmoon_client",
            "config_fields": ["base_url", "insights_url", "max_retries", "backoff_factor", "timeout"],
            "special_fields": {
                "api_key": lambda client: (getattr(client, "session", None) and client.session.headers.get("Api-key"))
                or None,
            },
        },
        "lookonchain": {
            "client_attr": "lookonchain_client",
            "config_fields": ["base_url", "search_endpoint", "max_retries", "backoff_factor", "timeout"],
        },
        "treeofalpha": {
            "client_attr": "treeofalpha_client",
            "config_fields": ["base_url", "news_endpoint", "cache_ttl", "max_retries", "backoff_factor", "timeout"],
        },
        "researchagent": {
            "client_attr": "researchagent_client",
            "config_fields": ["base_url", "api_key"],
            "special_fields": {
                "api_key": lambda client: (getattr(client, "session", None) and client.session.headers.get("Api-key"))
                or None,
            },
        },
        "unlocks": {
            "client_attr": "unlocks_client",
            "config_fields": ["base_url", "max_retries", "backoff_factor", "timeout"],
        },
    }

    def load_handlers(self, author, component_name, directory, config) -> Generator[Any, Any, None]:
        """Load in the handlers."""
        self.context.logger.info(f"Loading handlers for Author: {author}, Component: {component_name}")

        # Add the root directory to Python path
        parent_dir = str(directory.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            self.context.logger.info(f"Added {parent_dir} to Python path")

        handlers_config = config.get("handlers", [])
        if not handlers_config:
            self.context.logger.info("No handlers found in config")
            return

        module = dynamic_import(component_name, "handlers")

        for handler_config in handlers_config:
            self._load_single_handler(module, handler_config)

    def _load_single_handler(self, module: Any, handler_config: dict[str, Any]) -> None:
        """Load a single handler."""
        class_name = handler_config["class_name"]
        handler_kwargs = handler_config.get("kwargs", {})

        try:
            handler_class = getattr(module, class_name)
            handler = handler_class(name=class_name, skill_context=self.context, **handler_kwargs)

            protocol = getattr(handler_class, "SUPPORTED_PROTOCOL", None)
            if str(protocol) in PROTOCOL_HANDLER_MAP:
                handler_list = getattr(self.context.api_client_strategy, PROTOCOL_HANDLER_MAP[str(protocol)])
                handler_list.append(handler)
                self.context.logger.info(f"Handler {class_name} added to {PROTOCOL_HANDLER_MAP[str(protocol)]}. ")
            else:
                self.context.logger.warning(
                    f"Handler {class_name} has no supported protocol. "
                    f"Available protocols: {list(PROTOCOL_HANDLER_MAP.keys())}"
                )
        except (AttributeError, TypeError) as e:
            self.context.logger.exception(f"Error loading handler {class_name}: {e}")
            raise

    def _initialize_single_client(self, client_name: str, config: dict[str, Any]) -> dict[str, Any] | None:
        """Initialize a single API client and return its configuration."""
        try:
            client = getattr(self.context, config["client_attr"])
            self.context.api_clients[client_name] = client

            client_config = {}

            for field in config.get("config_fields", []):
                client_config[field] = getattr(client, field, None)

            for field_name, extractor in config.get("special_fields", {}).items():
                client_config[field_name] = extractor(client)

            self.context.api_client_configs[client_name] = client_config
            return None

        except (AttributeError, ValueError) as e:
            return {f"{client_name}_init": str(e)}

    def _initialize_api_clients(self) -> dict[str, str]:
        """Initialize API clients and collect errors. Also build per-client config for per-thread instantiation."""
        self.context.api_clients = {}
        self.context.api_client_configs = {}
        all_errors = {}

        for client_name, config in self.API_CLIENT_CONFIGS.items():
            error = self._initialize_single_client(client_name, config)
            if error:
                all_errors.update(error)

        return all_errors

    def _create_error_context(self, error_type: str, error_message: str) -> dict[str, Any]:
        """Create the error context."""
        trigger_context = getattr(self.context, "trigger_context", {})
        return {
            "error_type": error_type,
            "error_message": error_message,
            "originating_round": str(self._state),
            "trigger_id": trigger_context.get("trigger_id"),
            "asset_id": trigger_context.get("asset_id"),
        }

    def _setup_database(self) -> None:
        """Setup the database connection."""
        self.context.db_model.setup()
        is_valid, error_msg = self.context.strategy.validate_database_schema()
        if not is_valid:
            raise ValueError(error_msg)

    def act(self) -> None:
        """Setup the database connection and load the handlers."""
        self.context.logger.info(f"In state: {self._state}")

        try:
            # Setup database
            self._setup_database()

            # Load component configuration and handlers
            author, component_name, directory, config = self.custom_api_component_info

            if config.get("handlers"):
                self.load_handlers(author, component_name, directory, config)
                self.context.logger.info("Handlers loaded successfully")

            # Initialize API clients
            errors = self._initialize_api_clients()

            if errors:
                self.context.logger.warning(f"Failed to initialize API clients: {errors}")
                self.context.error_context = self._create_error_context("configuration_error", str(errors))
                self._event = DyorabciappEvents.ERROR
            else:
                self.context.logger.info("Successfully initialized API clients")
                self._event = DyorabciappEvents.DONE

        except ValueError as e:
            self.context.logger.exception(f"Configuration error during DB setup: {e}")
            self.context.error_context = self._create_error_context("configuration_error", str(e))
            self._event = DyorabciappEvents.ERROR

        except (sqlalchemy.exc.SQLAlchemyError, requests.exceptions.RequestException) as e:
            self.context.logger.exception(f"Unexpected error during DB setup: {e}")
            self.context.error_context = self._create_error_context("database_error", str(e))
            self._event = DyorabciappEvents.ERROR

        finally:
            self._is_done = True


class DeliverReportRound(BaseState):
    """This class implements the behaviour of the state DeliverReportRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.DELIVERREPORTROUND

    def _update_trigger_status(self, conn) -> bool:
        """Update trigger status to processed."""
        try:
            conn.execute(
                text("""
                    UPDATE triggers
                    SET status = 'processed', completed_at = NOW()
                    WHERE trigger_id = :trigger_id
                """),
                {"trigger_id": self.context.trigger_context["trigger_id"]},
            )
            conn.commit()
            return True
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.context.logger.exception(f"Failed to update trigger status: {e}")
            return False

    def _publish_report_event(self) -> bool:
        """Publish report_generated event via WebSocket."""
        try:
            for handler in self.context.api_client_strategy.ws_handlers:
                if hasattr(handler, "broadcast_event"):
                    handler.broadcast_event(
                        "report_generated",
                        {
                            "trigger_id": self.context.trigger_context["trigger_id"],
                            "asset_id": self.context.trigger_context["asset_id"],
                            "asset_symbol": self.context.trigger_context["asset_symbol"],
                            "report_id": self.context.report_context["report_id"],
                            "status": "success",
                        },
                    )
            return True
        except (ConnectionError, TimeoutError, ValueError) as e:
            self.context.logger.warning(f"Error publishing report event: {e}")
            return False

    def act(self) -> None:
        """Update trigger status and notify about report completion."""
        self.context.logger.info(f"Entering state: {self._state}")

        try:
            # Update trigger status
            with self.context.db_model.engine.connect() as conn:
                if not self._update_trigger_status(conn):
                    msg = "Failed to update trigger status"
                    raise RuntimeError(msg)

            self._publish_report_event()

            self.context.logger.info("Report published successfully")

            self._event = DyorabciappEvents.DONE

        except Exception as e:
            self.context.logger.exception(f"Error in DeliverReportRound: {e}")
            self.context.error_context = {
                "error_type": "delivery_error",
                "error_message": str(e),
                "trigger_id": self.context.trigger_context.get("trigger_id"),
                "asset_id": self.context.trigger_context.get("asset_id"),
                "critical": True,  # DB failure is critical
            }
            self._event = DyorabciappEvents.ERROR

        self._is_done = True


class TriggerRound(BaseState):
    """This class implements the behaviour of the state TriggerRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.TRIGGERROUND

    def _validate_asset(self, conn) -> tuple[bool, str]:
        """Validate asset exists and is active."""
        try:
            result = conn.execute(
                text("""
                    SELECT a.asset_id, a.symbol, a.name
                    FROM assets a
                    WHERE a.asset_id = :asset_id
                """),
                {"asset_id": self.context.trigger_context["asset_id"]},
            )
            asset = result.fetchone()

            if not asset:
                return False, "Asset not found"

            return True, ""
        except sqlalchemy.exc.SQLAlchemyError as e:
            return False, f"Database error: {e!s}"

    def _check_recent_report(self, conn, asset_id: int) -> tuple[bool, str]:
        """Check if a recent report exists (within last 24h unless force_refresh)."""
        trigger_details = self.context.trigger_context.get("trigger_details", {})
        if trigger_details.get("force_refresh"):
            return True, ""

        try:
            result = conn.execute(
                text("""
                    SELECT r.report_id, r.created_at
                    FROM reports r
                    WHERE r.asset_id = :asset_id
                    AND r.created_at > NOW() - INTERVAL '24 hours'
                    ORDER BY r.created_at DESC
                    LIMIT 1
                """),
                {"asset_id": asset_id},
            )
            report = result.fetchone()

            if report:
                return False, f"Recent report exists from {report[1]}"
            return True, ""
        except sqlalchemy.exc.SQLAlchemyError as e:
            return False, f"Database error checking recent report: {e!s}"

    def act(self) -> None:
        """Process trigger request and prepare for data ingestion."""
        self.context.logger.info(f"Entering state: {self._state}")

        try:
            with self.context.db_model.engine.connect() as conn:
                # Validate asset and check recent report
                is_valid, error_msg = self._validate_asset(conn)
                if not is_valid:
                    raise ValueError(error_msg)

                # Update trigger context with validated asset info
                self.context.trigger_context.update(
                    {
                        "asset_id": self.context.trigger_context["asset_id"],
                        "asset_symbol": self.context.trigger_context["asset_symbol"],
                        "asset_name": self.context.trigger_context["asset_name"],
                    }
                )

                can_proceed, error_msg = self._check_recent_report(conn, self.context.trigger_context["asset_id"])
                if not can_proceed:
                    raise ValueError(error_msg)

                # Update metrics
                self.context.strategy.increment_active_triggers()
                self._event = DyorabciappEvents.DONE

        except ValueError as e:
            self.context.logger.warning(f"Validation error: {e!s}")
            self.context.error_context = {
                "error_type": "validation_error",
                "error_message": str(e),
                "trigger_id": self.context.trigger_context.get("trigger_id"),
                "asset_id": self.context.trigger_context.get("asset_id"),
            }
            self._event = DyorabciappEvents.ERROR

        except Exception as e:
            self.context.logger.exception(f"Unexpected error in TriggerRound: {e}")
            self.context.error_context = {
                "error_type": "unexpected_error",
                "error_message": str(e),
                "trigger_id": self.context.trigger_context.get("trigger_id"),
                "asset_id": self.context.trigger_context.get("asset_id"),
            }
            self._event = DyorabciappEvents.ERROR

        self._is_done = True


class IngestDataRound(BaseState):
    """This class implements the behaviour of the state IngestDataRound."""

    def __init__(self, **kwargs: Any) -> None:
        self._max_workers = kwargs.pop("max_workers", None)
        super().__init__(**kwargs)
        self.context.logger.info(f"IngestDataRound max_workers: {self._max_workers}")
        self._state = DyorabciappStates.INGESTDATAROUND

        if self._max_workers is None:
            self._max_workers = os.cpu_count() or 4
            self.context.logger.warning(f"max_workers not provided. Falling back to {self._max_workers}")

        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._request_queue: list | None = None
        self._futures: dict | None = None
        self._phase: str | None = None

    def _validate_trigger_context(self) -> tuple[str, int]:
        asset_symbol = self.context.trigger_context.get("asset_symbol")
        trigger_id = self.context.trigger_context.get("trigger_id")
        if not asset_symbol or not trigger_id:
            msg = "Missing asset symbol or trigger ID in trigger context"
            raise ValueError(msg)
        return asset_symbol, trigger_id

    def _initialize_raw_data(self) -> None:
        self.context.raw_data = {
            source: {} if config.get("data_type_handler") == "multi" else None
            for source, config in DATA_SOURCES.items()
        }

    def _get_existing_full_unlocks_data(self) -> dict | None:
        """Check if we have recent full unlocks data in the database."""
        try:
            with self.context.db_model.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT raw_data
                        FROM scraped_data
                        WHERE source = 'unlocks'
                        AND data_type = 'all_projects'
                        AND ingested_at > NOW() - INTERVAL '30 days'
                        ORDER BY ingested_at DESC
                        LIMIT 1
                    """),
                )
                row = result.fetchone()
                if row and row[0]:
                    self.context.logger.info("Found existing full unlocks data in DB.")
                    data = row[0]
                    if isinstance(data, str):
                        data = json.loads(data)
                    return data
                return None
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.context.logger.warning(f"Error checking for existing full unlocks data: {e}")
            return None

    def _update_asset_name_if_needed(self, asset_symbol, asset_name):
        """Update asset name in DB and trigger_context if needed, return the resolved asset_name."""
        trendmoon_raw = self.context.raw_data.get("trendmoon", {})
        real_name = None
        if isinstance(trendmoon_raw, dict):
            coin_details = trendmoon_raw.get("coin_details", {})
            project_summary = trendmoon_raw.get("project_summary", {})

            if coin_details is None:
                coin_details = {}
            if project_summary is None:
                project_summary = {}
            real_name = coin_details.get("name") or project_summary.get("name")
        if real_name and real_name.lower() != asset_symbol.lower():
            try:
                query = text("UPDATE assets SET name = :name, updated_at = NOW() WHERE symbol = :symbol")
                with self.context.db_model.engine.connect() as conn:
                    conn.execute(query, {"name": real_name, "symbol": asset_symbol})
                    conn.commit()
                self.context.logger.info(f"Updated asset name in DB for symbol {asset_symbol} to '{real_name}'")
                asset_name = real_name
                self.context.trigger_context["asset_name"] = asset_name
            except (sqlalchemy.exc.SQLAlchemyError, TypeError, ValueError) as e:
                self.context.logger.warning(f"Failed to update asset name in DB: {e}")
        else:
            resolved_name = self.context.db_model.get_asset_name_by_symbol(asset_symbol)
            if resolved_name:
                asset_name = resolved_name
                self.context.trigger_context["asset_name"] = asset_name
        return asset_name

    def _fetch_unlocks_data_task(self):
        """Fetch or reuse full unlocks data and store in raw_data['unlocks']."""
        try:
            full_unlocks_item = self._get_existing_full_unlocks_data()
            if not full_unlocks_item:
                full_unlocks_item = unlocks_fetcher(self.context)
                # Store the full unlocks dataset in DB
                self.context.db_model.store_raw_data(
                    source="unlocks",
                    data_type="all_projects",
                    data=full_unlocks_item.to_dict() if hasattr(full_unlocks_item, "to_dict") else full_unlocks_item,
                    trigger_id=self.context.trigger_context.get("trigger_id"),
                    timestamp=datetime.now(tz=UTC),
                    asset_id=self.context.trigger_context.get("asset_id"),
                )
                self.context.logger.info("Fetched and stored fresh full unlocks data.")
            else:
                self.context.logger.info("Using cached full unlocks data from DB.")
            # Always store the all_projects list in raw_data['unlocks']
            if hasattr(full_unlocks_item, "metadata"):
                self.context.raw_data["unlocks"] = full_unlocks_item.metadata.get("all_projects", [])
            elif isinstance(full_unlocks_item, dict):
                self.context.raw_data["unlocks"] = full_unlocks_item.get("metadata", {}).get("all_projects", [])
            else:
                self.context.raw_data["unlocks"] = []
        except (sqlalchemy.exc.SQLAlchemyError, TypeError, ValueError) as e:
            self.context.logger.warning(f"Error fetching or storing unlocks data: {e}")
            self.context.raw_data["unlocks"] = []
            raise

    def _process_future_result(self, name, result, error=None):
        if error:
            if not hasattr(self.context, "raw_errors"):
                self.context.raw_errors = {}
            self.context.raw_errors[name] = error
        if name.startswith("trendmoon"):
            source, data_type = name.split("_", 1)
            self.context.raw_data[source][data_type] = result
        elif name.startswith("researchagent_"):
            if self.context.raw_data["researchagent"] is None:
                self.context.raw_data["researchagent"] = []
            if isinstance(result, list):
                self.context.raw_data["researchagent"].extend(result)
            elif result is not None:
                self.context.raw_data["researchagent"].append(result)
        else:
            self.context.raw_data[name] = result

    def _setup_run(self) -> None:
        """Set up a new run."""
        self.context.logger.info(
            f"Setting up data ingestion run for trigger {self.context.trigger_context.get('trigger_id')}"
        )
        self._request_queue = []
        self._futures = {}
        self._is_done = False  # Reset done state for new trigger
        self._initialize_raw_data()
        asset_symbol, _ = self._validate_trigger_context()
        asset_name = self.context.trigger_context.get("asset_name")

        phase1_sources = [s for s in DATA_SOURCES if s != "unlocks"]
        for source in phase1_sources:
            config = DATA_SOURCES[source]
            if config.get("data_type_handler") == "multi":
                for endpoint, fetcher in config["fetchers"].items():
                    self._request_queue.append(
                        {
                            "name": f"{source}_{endpoint}",
                            "fetcher": fetcher,
                            "symbol": asset_symbol,
                            "asset_name": asset_name,
                        }
                    )
            else:
                fetcher = config["fetcher"]
                self._request_queue.append(
                    {"name": source, "fetcher": fetcher, "symbol": asset_symbol, "asset_name": asset_name}
                )

    def _submit_new_requests(self) -> None:
        """Submit new requests from the queue up to max_workers."""
        while self._request_queue and len(self._futures) < self._max_workers:
            request_info = self._request_queue.pop(0)
            name = request_info["name"]
            fetcher = request_info["fetcher"]
            symbol = request_info["symbol"]
            asset_name = request_info["asset_name"]

            future = self._executor.submit(fetcher, self.context, symbol, asset_name=asset_name)
            self._futures[name] = future

    def _process_completed_futures(self) -> None:
        """Process any futures that have completed."""
        if not self._futures:
            return
        done_futures = {name: future for name, future in self._futures.items() if future.done()}

        for name, future in done_futures.items():
            del self._futures[name]
            try:
                result = future.result()
                self._process_future_result(name, result)
            except (TrendmoonAPIError, TreeOfAlphaAPIError, LookOnChainAPIError, ResearchAgentAPIError) as e:
                http_response = getattr(e, "response", None)
                status_code = getattr(http_response, "status_code", None) if http_response else None
                response_text = getattr(http_response, "text", None) if http_response else None

                error_dump = {
                    "error": str(e),
                    "http_response": response_text,
                    "status_code": status_code,
                }
                self._process_future_result(name, None, error=error_dump)
                self.context.logger.warning(f"Error fetching {name}: {e} | HTTP: {error_dump}")

                # Check for critical asset validation errors (404 "Coin not found")
                error_str = str(e).lower()
                # TrendMoon coin_details 404 means invalid asset symbol
                is_coin_not_found = name == "trendmoon_coin_details" and "status 404" in error_str
                if is_coin_not_found:
                    self.context.logger.exception(f"Invalid asset symbol detected for {name}: {e}")
                    asset_symbol = self.context.trigger_context.get("asset_symbol")
                    self.context.error_context = {
                        "error_type": "asset_validation_error",
                        "error_message": f"Asset symbol '{asset_symbol}' not found in external APIs",
                        "error_source": "asset_lookup",
                        "trigger_id": self.context.trigger_context.get("trigger_id"),
                        "asset_id": self.context.trigger_context.get("asset_id"),
                        "originating_round": str(self._state),
                        "critical": True,
                        "recoverable": False,
                    }

                    self._event = DyorabciappEvents.ERROR
                    self._is_done = True
                    return
            except Exception as e:
                self.context.logger.exception(f"Unexpected error processing future for {name}: {e}")
                self._process_future_result(name, None, error={"error": str(e)})

    def _finish_run(self) -> None:
        """Finalize the run and transition."""
        if self._futures and any(not f.done() for f in self._futures.values()):
            self.context.logger.info("Data ingestion run not complete. Waiting for futures to complete.")
            return
        self.context.logger.info(
            f"Data ingestion run complete for trigger {self.context.trigger_context.get('trigger_id')}"
        )
        self._event = DyorabciappEvents.DONE
        self._is_done = True
        self._phase = None
        self._futures = None
        self._request_queue = None

    def act(self) -> None:
        """Ingest data from all sources."""
        try:
            if self._phase is None:
                self.context.logger.info(f"Entering state: {self._state}")
                self._setup_run()
                self._phase = "phase1"

            self._process_completed_futures()

            if self._phase == "phase1":
                self._submit_new_requests()
                if not self._request_queue and not self._futures:
                    self.context.logger.info("Phase 1 data ingestion complete.")
                    asset_symbol, _ = self._validate_trigger_context()
                    asset_name = self.context.trigger_context.get("asset_name")
                    self._update_asset_name_if_needed(asset_symbol, asset_name)
                    self._phase = "unlocks"
                    future = self._executor.submit(self._fetch_unlocks_data_task)
                    self._futures["unlocks"] = future

            elif self._phase == "unlocks":
                if not self._futures:
                    self.context.logger.info("Unlocks data ingestion complete.")
                    self._finish_run()

        except Exception as e:
            self.context.logger.exception(f"Error during data ingestion: {e}")
            self.context.error_context = {
                "error_type": "ingestion_error",
                "error_message": str(e),
                "originating_round": str(self._state),
                "trigger_id": getattr(self.context, "trigger_context", {}).get("trigger_id"),
                "asset_id": getattr(self.context, "trigger_context", {}).get("asset_id"),
            }
            self._event = DyorabciappEvents.ERROR
            self._is_done = True


class GenerateReportRound(BaseState):
    """This class implements the behaviour of the state GenerateReportRound."""

    REQUIRED_SECTIONS = [
        "Overview",
        "Key Recent Changes",
        "Recent News/Events",
        "Analysis",
        "Conclusion",
    ]

    MODEL_CONFIG = {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.GENERATEREPORTROUND
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._prompt = None
        self._payload = None
        self._attempt = 0
        self._max_retries = 3
        self._last_error = None

    def _fetch_structured_payload(self):
        """Fetch the structured payload from the database."""
        trigger_id = self.context.trigger_context.get("trigger_id")
        asset_id = self.context.trigger_context.get("asset_id")

        payload = self.context.db_model.get_structured_payload(trigger_id, asset_id)
        if not payload:
            msg = "No structured payload found for this trigger/asset."
            raise ValueError(msg)
        return payload

    def _validate_markdown(self, text: str) -> bool:
        """Check if the text is valid Markdown (basic check: can be parsed)."""
        try:
            html = markdown.markdown(text)
            return bool(html.strip())
        except (ValueError, TypeError):
            return False

    def _check_required_sections(self, text: str) -> list[str]:
        """Return a list of missing required section headers."""
        missing = []
        for section in self.REQUIRED_SECTIONS:
            # Look for a Markdown header (### Section)
            if not re.search(rf"^###\s*{re.escape(section)}", text, re.MULTILINE | re.IGNORECASE):
                missing.append(section)
        return missing

    def _setup_run(self) -> bool:
        """Set up a new report generation run. Returns False if report already exists."""
        self._future = None
        self._prompt = None
        self._payload = None
        self._attempt = 1
        self._last_error = None
        self._is_done = False  # Reset done state for new trigger

        trigger_id = self.context.trigger_context.get("trigger_id")
        if self.context.db_model.report_exists(trigger_id):
            self.context.logger.warning(f"Report already exists for trigger_id={trigger_id}")
            return False

        self._payload = self._fetch_structured_payload()
        self._prompt = build_report_prompt(self._payload)
        self.context.logger.info("Built report prompt.")
        self.context.logger.debug(f"Prompt: {self._prompt}")
        return True

    def _submit_llm_request(self) -> None:
        """Submit a request to the LLM service."""
        self.context.logger.info(f"Submitting LLM request (attempt {self._attempt}/{self._max_retries})")
        self._future = self._executor.submit(self.context.llm_service.generate_summary, self._prompt, self.MODEL_CONFIG)

    def _process_llm_result_and_store(self, llm_result: dict) -> None:
        """Process, validate, and store the LLM result."""
        llm_output = llm_result["content"]
        self.context.logger.info(f"LLM output (attempt {self._attempt}): {llm_output[:500]}...")

        if not self._validate_markdown(llm_output):
            msg = f"LLM output is not valid Markdown (attempt {self._attempt})."
            raise ValueError(msg)

        missing_sections = self._check_required_sections(llm_output)
        if missing_sections:
            msg = f"LLM output missing required sections: {', '.join(missing_sections)} (attempt {self._attempt})"
            raise ValueError(msg)

        report_id = self.context.db_model.store_report(
            trigger_id=self.context.trigger_context.get("trigger_id"),
            asset_id=self.context.trigger_context.get("asset_id"),
            report_content_markdown=llm_output,
            report_data_json=self._payload,
            llm_model_used=llm_result["llm_model_used"],
            generation_time_ms=llm_result["generation_time_ms"],
            token_usage=llm_result["token_usage"],
        )

        if not hasattr(self.context, "report_context"):
            self.context.report_context = {}
        self.context.report_context["report_id"] = report_id
        self.context.strategy.record_report_generated()
        self.context.logger.info(f"Stored report with ID {report_id}.")

    def _create_error_context(self, error_type: str, error_message: str) -> dict:
        """Create a standardized error context dictionary."""
        return {
            "error_type": error_type,
            "error_message": error_message,
            "trigger_id": self.context.trigger_context.get("trigger_id"),
            "asset_id": self.context.trigger_context.get("asset_id"),
            "originating_round": str(self._state),
        }

    def _handle_max_retries_exceeded(self) -> None:
        """Log and set context when max retries are exceeded."""
        error_message = f"Report generation failed after {self._max_retries} attempts: {self._last_error}"
        critical_info = self._create_error_context("report_generation_error", error_message)
        critical_info["level"] = "CRITICAL"
        self.context.logger.critical(error_message, extra=critical_info)
        self.context.error_context = critical_info

    def _finish_run(self, success: bool = True) -> None:
        """Finalize the run and set the event."""
        self._event = DyorabciappEvents.DONE if success else DyorabciappEvents.ERROR
        self._is_done = True
        self._future = None
        self._prompt = None
        self._payload = None
        self._attempt = 0
        self._last_error = None

    def act(self) -> None:
        """Generate a report using a non-blocking, asynchronous pattern."""
        if self._attempt == 0:  # First call for this trigger
            self.context.logger.info(f"Entering state: {self._state}")
            if not self._setup_run():
                self._finish_run(success=True)  # Report exists, consider it done
                return
            self._submit_llm_request()
            return  # Defer processing to the next tick

        if self._future is None:
            self.context.logger.error("GenerateReportRound: _future is None unexpectedly.")
            self.context.error_context = self._create_error_context("internal_error", "Future was not set.")
            self._finish_run(success=False)
            return

        if not self._future.done():
            self.context.logger.debug("Waiting for LLM response...")
            return

        # Future is done, process it
        try:
            llm_result = self._future.result()
            self._process_llm_result_and_store(llm_result)
            self._finish_run(success=True)
        except (LLMServiceError, ValueError) as e:
            self._last_error = e
            self.context.logger.warning(
                f"LLM report generation failed (attempt {self._attempt}/{self._max_retries}): {e}"
            )
            self._attempt += 1
            if self._attempt > self._max_retries:
                self._handle_max_retries_exceeded()
                self._finish_run(success=True)  # Finished with critical error, but "done" per original logic
            else:
                self._submit_llm_request()  # Retry
        except Exception as e:
            self.context.logger.exception(f"Unexpected error in GenerateReportRound: {e}")
            self.context.error_context = self._create_error_context("report_generation_error", str(e))
            self._finish_run(success=False)


class HandleErrorRound(BaseState):
    """This class implements the behaviour of the state HandleErrorRound."""

    # Error classification rules
    RETRYABLE_ERRORS = {
        "database_error": True,
        "storage_error": True,
        "api_error": True,
        "timeout_error": True,
        "llm_api_error": True,
        "llm_rate_limit": True,
        "llm_generation_error": True,  # Sometimes retryable
        "scraping_error": True,
    }
    NON_RETRYABLE_ERRORS = {
        "configuration_error": False,
        "validation_error": False,
        "data_validation_error": False,
        "asset_validation_error": False,
        "internal_logic_error": False,
        "resource_exhaustion": False,
        "llm_content_filter": False,
    }
    # Default retry/backoff config
    BASE_DELAY = 5  # seconds
    MAX_DELAY = 300  # seconds
    MAX_ATTEMPTS = 5
    JITTER = 0.2  # 20% jitter

    def __init__(self, **kwargs: Any) -> None:
        self.ntfy_topic = kwargs.pop("ntfy_topic", "alerts")
        super().__init__(**kwargs)
        self._state = DyorabciappStates.HANDLEERRORROUND
        self._retry_states = {}  # Store retry states in the class instance

    def _classify_error(self, error_context: dict) -> tuple[bool, str]:
        """Classify error and determine if retryable."""
        error_type = (error_context.get("error_type") or "").lower()
        if error_type in self.RETRYABLE_ERRORS:
            return True, error_type
        if error_type in self.NON_RETRYABLE_ERRORS:
            return False, error_type
        # Fallback: retry on API/DB/timeout, not on validation/config
        if "api" in error_type or "timeout" in error_type or "storage" in error_type or "database" in error_type:
            return True, error_type
        if "validation" in error_type or "config" in error_type or "resource" in error_type:
            return False, error_type
        return False, error_type or "unknown"

    def _get_retry_state(self, trigger_id: int) -> dict:
        """Get or initialize retry state for this trigger."""
        return self._retry_states.setdefault(trigger_id, {"attempt": 0, "last_ts": None})

    def _increment_error_metrics(self, error_type: str) -> None:
        """Increment error metrics in strategy if available."""
        if hasattr(self.context, "strategy") and hasattr(self.context.strategy, "_metrics"):
            metrics = self.context.strategy._metrics  # noqa: SLF001
            key = f"errors_{error_type}"
            metrics[key] = metrics.get(key, 0) + 1

    def _log_json_error(self, error_context: dict, retryable: bool, attempt: int, max_attempts: int) -> None:
        log_record = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "level": "ERROR" if retryable else "CRITICAL",
            "round": error_context.get("originating_round", "unknown"),
            "asset_id": error_context.get("asset_id"),
            "trigger_id": error_context.get("trigger_id"),
            "error_type": error_context.get("error_type"),
            "message": error_context.get("error_message"),
            "stack": error_context.get("stack_trace"),
            "retryable": retryable,
            "attempt": attempt,
            "max_attempts": max_attempts,
        }
        self.context.logger.error(json.dumps(log_record, default=str))

    def _send_alert(self, error_context: dict, critical: bool = False) -> None:
        # Integrate with ntfy.sh alerting system, else log CRITICAL
        alert_msg = (
            f"ALERT: {error_context.get('error_type')} | {error_context.get('error_message')} | "
            f"Trigger: {error_context.get('trigger_id')} | Asset: {error_context.get('asset_id')}"
        )
        try:
            requests.post(
                f"https://ntfy.sh/{self.ntfy_topic}",
                data=alert_msg.encode("utf-8"),
                headers={
                    "Title": f"{error_context.get('error_type', 'Error').replace('_', ' ').title()} detected",
                    "Priority": "urgent" if critical else "high",
                    "Tags": "warning,skull" if critical else "warning",
                },
                timeout=5,
            )
        except (requests.RequestException, requests.Timeout, requests.ConnectionError) as e:
            self.context.logger.critical(f"Failed to send alert to ntfy.sh: {e!s} | {alert_msg}")
            self.context.logger.critical(alert_msg)

    def _update_trigger_status_error(self, trigger_id: int, error_message: str | None = None) -> None:
        # Mark the trigger as errored in the DB
        try:
            with self.context.db_model.engine.connect() as conn:
                conn.execute(
                    text("""
                        UPDATE triggers
                        SET status = 'error', completed_at = NOW(), error_message = :error_message
                        WHERE trigger_id = :trigger_id
                    """),
                    {"trigger_id": trigger_id, "error_message": error_message},
                )
                conn.commit()
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.context.logger.critical(f"Failed to update trigger status to error for trigger_id={trigger_id}: {e!s}")

    def _calculate_backoff(self, attempt: int) -> float:
        base = self.BASE_DELAY * (2**attempt)
        jitter = base * self.JITTER * (random.random() * 2 - 1)  # +/- jitter  # noqa: S311
        return min(self.MAX_DELAY, max(1, base + jitter))

    def act(self) -> None:
        """Handle error: log, increment metrics, retry or mark as failed, alert if critical."""
        self.context.logger.info(f"Entering state: {self._state}")
        error_context = getattr(self.context, "error_context", {}) or {}
        trigger_id = error_context.get("trigger_id")

        originating_round = error_context.get("originating_round", "unknown")

        retryable, error_type = self._classify_error(error_context)
        retry_state = self._get_retry_state(trigger_id) if trigger_id is not None else {"attempt": 0}
        attempt = retry_state["attempt"]

        self._log_json_error(error_context, retryable, attempt, self.MAX_ATTEMPTS)

        self._increment_error_metrics(error_type)

        if not retryable or attempt >= self.MAX_ATTEMPTS:
            self._send_alert(error_context, critical=True)

        if retryable and attempt < self.MAX_ATTEMPTS:
            delay = self._calculate_backoff(attempt)
            retry_state["attempt"] += 1
            retry_state["last_ts"] = datetime.now(tz=UTC).isoformat()
            self.context.logger.info(
                f"Retrying {originating_round} for trigger_id={trigger_id} in {delay:.1f}s"
                f"(attempt {attempt + 1}/{self.MAX_ATTEMPTS})"
            )
            time.sleep(delay)
            self._event = DyorabciappEvents.RETRY
        else:
            if trigger_id is not None:
                error_message = error_context.get("error_message", "Unknown error")
                self._update_trigger_status_error(trigger_id, error_message)
            self._event = DyorabciappEvents.DONE
        self._is_done = True


class DyorabciappFsmBehaviour(FSMBehaviour):
    """This class implements a simple Finite State Machine behaviour."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.register_state(DyorabciappStates.SETUPDYORROUND.value, SetupDYORRound(**kwargs), True)

        self.register_state(DyorabciappStates.WATCHINGROUND.value, WatchingRound(**kwargs))
        self.register_state(DyorabciappStates.PROCESSDATAROUND.value, ProcessDataRound(**kwargs))
        self.register_state(DyorabciappStates.DELIVERREPORTROUND.value, DeliverReportRound(**kwargs))
        self.register_state(DyorabciappStates.TRIGGERROUND.value, TriggerRound(**kwargs))
        self.register_state(DyorabciappStates.INGESTDATAROUND.value, IngestDataRound(**kwargs))
        self.register_state(DyorabciappStates.GENERATEREPORTROUND.value, GenerateReportRound(**kwargs))
        self.register_state(DyorabciappStates.HANDLEERRORROUND.value, HandleErrorRound(**kwargs))

        self.register_transition(
            source=DyorabciappStates.DELIVERREPORTROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.DELIVERREPORTROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.DELIVERREPORTROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.DELIVERREPORTROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.GENERATEREPORTROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.DELIVERREPORTROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.GENERATEREPORTROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.GENERATEREPORTROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.GENERATEREPORTROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.HANDLEERRORROUND.value,
            event=DyorabciappEvents.RETRY,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.HANDLEERRORROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.INGESTDATAROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.PROCESSDATAROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.INGESTDATAROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.INGESTDATAROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.INGESTDATAROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.PROCESSDATAROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.GENERATEREPORTROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.PROCESSDATAROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.PROCESSDATAROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.PROCESSDATAROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.SETUPDYORROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.SETUPDYORROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.SETUPDYORROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.SETUPDYORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.TRIGGERROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.INGESTDATAROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.TRIGGERROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.TRIGGERROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.TRIGGERROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.WATCHINGROUND.value,
            event=DyorabciappEvents.NO_TRIGGER,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.WATCHINGROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.WATCHINGROUND.value,
            event=DyorabciappEvents.TRIGGER,
            destination=DyorabciappStates.TRIGGERROUND.value,
        )

    def setup(self) -> None:
        """Implement the setup."""
        self.context.logger.info("Setting up Dyorabciapp FSM behaviour.")

    def teardown(self) -> None:
        """Implement the teardown."""
        self.context.logger.info("Tearing down Dyorabciapp FSM behaviour.")

    def act(self) -> None:
        """Implement the act."""
        super().act()
        if self.current is None:
            self.context.logger.info("No state to act on.")
            self.terminate()

    def terminate(self) -> None:
        """Implement the termination."""
        os._exit(0)
