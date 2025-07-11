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
# ------------------------------------------------------------------------------
"""Utility functions for DYOR App skill behaviours."""

from datetime import UTC, tzinfo, datetime


def serialize_for_storage(data) -> any:
    """Recursively serialize data for storage."""
    # Handle basic collection types
    if isinstance(data, dict):
        return {key: serialize_for_storage(value) for key, value in data.items()}
    if isinstance(data, list):
        return [serialize_for_storage(item) for item in data]

    # Handle datetime and timezone types
    if isinstance(data, datetime | tzinfo):
        return data.isoformat() if isinstance(data, datetime) else str(data)

    # Handle objects with serialization methods
    if hasattr(data, "model_dump"):
        return data.model_dump(mode="json")
    if hasattr(data, "__dict__"):
        return {k: serialize_for_storage(v) for k, v in data.__dict__.items() if not k.startswith("_")}

    # Default case
    return data


def calculate_percentage_change(latest_value: float, previous_value: float) -> float:
    """Calculate percentage change between two values."""
    if not previous_value:
        return 0.0
    return round(((latest_value - previous_value) / previous_value) * 100, 1)


def safe_get_nested(data: dict, *keys, default=None):
    """Safely get nested dictionary values."""
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def get_asset_identifiers(context) -> dict[str, str]:
    """Extract asset identifiers from trigger context."""
    return {
        "asset_name": context.trigger_context.get("asset_name"),
        "asset_symbol": context.trigger_context.get("asset_symbol"),
        "coingecko_id": context.trigger_context.get("coingecko_id"),
    }


def parse_timestamp_safely(ts, logger) -> datetime | None:
    """Parse various timestamp formats safely."""
    if isinstance(ts, str) and ts.isdigit():
        try:
            return datetime.fromtimestamp(int(ts), UTC)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse unix timestamp '{ts}': {e}")
    elif isinstance(ts, int | float):
        try:
            return datetime.fromtimestamp(ts, UTC)
        except (ValueError, TypeError, OSError) as e:
            logger.warning(f"Failed to parse numeric timestamp '{ts}': {e}")
    elif isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse ISO timestamp '{ts}': {e}")
    return None
