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

"""This package contains a scaffold of a model."""

from typing import Any
from datetime import UTC, datetime

from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from aea.skills.base import Model


class DYORStrategy(Model):
    """DYOR strategy."""

    # Configuration
    REQUIRED_TABLES: list[str] = [
        "assets",
        "triggers",
    ]

    # Initialization
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the strategy."""
        super().__init__(**kwargs)

        # Initialize metrics state
        self._metrics = {
            "active_triggers": 0,
            "reports_generated_today": 0,
            "latest_report_timestamp": None,
            "_last_reset_date": datetime.now(tz=UTC).date(),
        }

    # Database Operations
    def validate_database_schema(self) -> tuple[bool, str]:
        """Validate database schema and required tables."""
        try:
            engine = self.context.db_model.engine
            if not engine:
                return False, "Database engine not initialized"

            with engine.connect() as connection:
                # Basic connectivity
                connection.execute(text("SELECT 1"))

                # Schema validation
                inspector = inspect(engine)
                if "public" not in inspector.get_schema_names():
                    return False, "Public schema does not exist"

                # Table validation
                existing_tables = inspector.get_table_names(schema="public")
                missing_tables = set(self.REQUIRED_TABLES) - set(existing_tables)

                if missing_tables:
                    return False, f"Missing required tables: {missing_tables}"

                return True, ""

        except SQLAlchemyError as e:
            return False, f"Schema validation failed: {e!s}"

    # Metrics Operations
    def _check_daily_reset(self) -> None:
        """Reset daily counters if needed."""
        today = datetime.now(tz=UTC).date()
        if today > self._metrics["_last_reset_date"]:
            self._metrics.update({"reports_generated_today": 0, "_last_reset_date": today})

    @property
    def active_triggers(self) -> int:
        """Get number of active triggers."""
        return self._metrics["active_triggers"]

    def increment_active_triggers(self) -> None:
        """Increment active triggers count."""
        self._metrics["active_triggers"] += 1

    def decrement_active_triggers(self) -> None:
        """Decrement active triggers count."""
        self._metrics["active_triggers"] = max(0, self._metrics["active_triggers"] - 1)

    def record_report_generated(self) -> None:
        """Record a new report generation."""
        self._check_daily_reset()
        self._metrics["reports_generated_today"] += 1
        self._metrics["latest_report_timestamp"] = datetime.now(tz=UTC)

    # Public API
    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics state."""
        self._check_daily_reset()
        return {
            "active_triggers": self._metrics["active_triggers"],
            "reports_generated_today": self._metrics["reports_generated_today"],
            "latest_report_timestamp": self._metrics["latest_report_timestamp"].isoformat() + "Z"
            if self._metrics["latest_report_timestamp"]
            else None,
        }
