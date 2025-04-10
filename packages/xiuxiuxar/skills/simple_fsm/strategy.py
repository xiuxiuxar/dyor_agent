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

from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from aea.skills.base import Model


class DYORStrategy(Model):
    """DYOR strategy."""

    REQUIRED_TABLES: list[str] = [
        "assets",
        "triggers",
    ]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the strategy."""
        super().__init__(**kwargs)
        self.context.logger.info("DYOR strategy initialized.")

    def validate_database_schema(self) -> tuple[bool, str]:
        """Validate database schema and required tables.

        Returns
        -------
            Tuple[bool, str]: (success, error_message)

        """
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
