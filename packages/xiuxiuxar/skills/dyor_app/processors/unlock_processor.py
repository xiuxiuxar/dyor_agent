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

"""Unlock data processor for DYOR App skill."""

from typing import Any
from datetime import UTC, datetime, timedelta

from packages.xiuxiuxar.skills.dyor_app.utils import (
    safe_get_nested,
    get_asset_identifiers,
)
from packages.xiuxiuxar.skills.dyor_app.constants import (
    RAW_DATA_TYPE,
    UNLOCKS_SOURCE,
    CLIFF_UNLOCK_TYPE,
    DEFAULT_DATA_TYPE,
    RECENT_UNLOCK_DAYS,
    UPCOMING_UNLOCK_DAYS,
    SIGNIFICANT_UNLOCK_CATEGORIES,
)
from packages.xiuxiuxar.skills.dyor_app.data_sources import unlocks_project_filter


class UnlockProcessor:
    """Handles all unlock-related data processing and filtering."""

    def __init__(self, context, logger):
        """Initialize with context and logger."""
        self.context = context
        self.logger = logger

    def filter_unlocks_data(self, raw_data: Any) -> dict:
        """Filter unlocks data for the current asset."""
        identifiers = get_asset_identifiers(self.context)

        self.logger.info(
            f"Filtering unlocks: asset_name={identifiers['asset_name']}, "
            f"asset_symbol={identifiers['asset_symbol']}, coingecko_id={identifiers['coingecko_id']} "
            f"all_projects_count={len(raw_data) if isinstance(raw_data, list) else 'N/A'}"
        )

        project, filtered_events = unlocks_project_filter(
            raw_data,
            coingecko_id=identifiers["coingecko_id"],
            asset_name=identifiers["asset_name"],
            symbol=identifiers["asset_symbol"],
        )

        self.logger.info(
            f"unlocks_project_filter result: project={project}, "
            f"filtered_events_count={len(filtered_events) if isinstance(filtered_events, list) else 'N/A'}"
        )

        return {"project": project, "filtered_events": filtered_events}

    def process_unlocks_data(
        self,
        raw_data: Any,
        processor_func: callable,
        trigger_id: int,
        asset_id: int,
        store_processed_data_func: callable,
    ) -> dict[str, str]:
        """Process unlocks data with special filtering logic."""
        unlocks_filtered = self.filter_unlocks_data(raw_data)
        serialized_data = processor_func(unlocks_filtered)
        store_processed_data_func(
            source=UNLOCKS_SOURCE,
            data_type=DEFAULT_DATA_TYPE,
            data=serialized_data,
            trigger_id=trigger_id,
            asset_id=asset_id,
        )
        return {}

    def serialize_unlocks_data(self, data):
        """Serialize unlocks data (ScrapedDataItem) to dict for storage."""
        if hasattr(data, "to_dict"):
            return data.to_dict()
        return data

    def store_unlocks_raw_data(self, raw: Any, trigger_id: int, asset_id: int) -> None:
        """Store filtered unlocks data as raw data."""
        unlocks_filtered = self.filter_unlocks_data(raw)

        self.logger.debug(
            "Storing filtered unlocks raw data for asset=%s: project=%s, filtered_events_count=%s",
            get_asset_identifiers(self.context)["asset_symbol"],
            unlocks_filtered.get("project"),
            len(unlocks_filtered.get("filtered_events", [])),
        )

        self.context.db_model.store_raw_data(
            source=UNLOCKS_SOURCE,
            data_type=RAW_DATA_TYPE,
            data=unlocks_filtered,
            trigger_id=trigger_id,
            timestamp=datetime.now(tz=UTC),
            asset_id=asset_id,
        )

    def build_unlock_events(self, unlocks_raw):
        """Build unlock events with categorization."""
        unlocks_data = self.normalize_unlocks_data(unlocks_raw)
        events = self.extract_project_events(unlocks_raw, unlocks_data)
        unlocks_recent, unlocks_upcoming = self.categorize_unlock_events(events)
        if isinstance(unlocks_data, list):
            unlocks_data = {"projects": unlocks_data}
        return unlocks_data, unlocks_recent, unlocks_upcoming

    def normalize_unlocks_data(self, unlocks_raw):
        """Normalize unlocks data to consistent format."""
        if hasattr(unlocks_raw, "to_dict"):
            return unlocks_raw.to_dict()
        if isinstance(unlocks_raw, dict):
            return unlocks_raw
        return str(unlocks_raw)

    def extract_project_events(self, unlocks_raw, unlocks_data):
        """Extract events for the current project from unlocks data."""
        # Handle case where unlocks_raw is the all_projects list
        if isinstance(unlocks_raw, list):
            identifiers = get_asset_identifiers(self.context)
            project = None

            # Try to find project by coingecko_id first, then name, then symbol
            search_methods = [
                (identifiers["coingecko_id"], lambda p: p.get("gecko_id")),
                (identifiers["asset_name"], lambda p: p.get("name", "").lower()),
                (identifiers["asset_symbol"], lambda p: p.get("token", "").split(":")[-1].lower()),
            ]

            for search_value, get_value in search_methods:
                if search_value and not project:
                    compare_value = search_value.lower() if isinstance(search_value, str) else search_value
                    project = next((p for p in unlocks_raw if get_value(p) == compare_value), None)

            if project:
                events = project.get("events", [])
                self.logger.info(f"Found project {project.get('name', '')} with {len(events)} events")
                return events

            self.logger.warning(f"Could not find project for identifiers: {identifiers}")
            return []

        # Handle dict case - try multiple possible event locations
        if isinstance(unlocks_data, dict):
            return (
                unlocks_data.get("events")
                or unlocks_data.get("filtered_events", [])
                or safe_get_nested(unlocks_data, "metadata", "raw_emissions", "meta", "events", default=[])
            )
        return []

    def categorize_unlock_events(self, events):
        """Categorize unlock events into recent and upcoming."""
        unlocks_recent = []
        unlocks_upcoming = []
        now = datetime.now(UTC)
        self.logger.info(f"Processing {len(events)} events for recent/upcoming unlock analysis. Current time: {now}")

        for event in events:
            # Filter for cliff unlocks AND specific categories
            unlock_type = event.get("unlockType")
            category = event.get("category")
            if unlock_type != CLIFF_UNLOCK_TYPE or category not in SIGNIFICANT_UNLOCK_CATEGORIES:
                continue
            try:
                event_date = datetime.fromtimestamp(event["timestamp"], UTC)
                self.logger.debug(
                    f"Processing significant unlock event: date={event_date}, "
                    f"type={event.get('unlockType')}, "
                    f"category={event.get('category')}, "
                    f"tokens={event.get('noOfTokens')}, "
                    f"description={event.get('description')}"
                )
                if now - timedelta(days=RECENT_UNLOCK_DAYS) <= event_date <= now:
                    unlocks_recent.append(event)
                    self.logger.info(f"Added to recent unlocks: {event_date}")
                elif now < event_date <= now + timedelta(days=UPCOMING_UNLOCK_DAYS):
                    unlocks_upcoming.append(event)
                    self.logger.info(
                        f"Found upcoming significant unlock in {(event_date - now).days} days: "
                        f"{event.get('description')}"
                    )
            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse unlock event timestamp: {e}")
                continue

        self.logger.info(f"Final results: {len(unlocks_recent)} recent, {len(unlocks_upcoming)} upcoming unlocks")
        return unlocks_recent, unlocks_upcoming

    def prepare_unlocks_data(self, unlocks_data, unlocks_recent, unlocks_upcoming) -> dict | None:
        """Prepare unlocks data with proper structure."""
        # Ensure unlocks_data is always a dict
        if isinstance(unlocks_data, list):
            unlocks_data = {"projects": unlocks_data}
        elif not isinstance(unlocks_data, dict):
            unlocks_data = {"data": unlocks_data}

        # Add summary if no unlock events
        if not (unlocks_recent or unlocks_upcoming):
            unlocks_data = {"summary": "No unlock data available."}
        elif "summary" not in unlocks_data:
            unlocks_data["summary"] = ""

        return None  # Always return None as per original logic
