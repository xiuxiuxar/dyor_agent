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

from sqlalchemy import create_engine
from aea.skills.base import Model
from sqlalchemy.engine import Engine


class DatabaseModel(Model):
    """Database connection and configuration."""

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model."""
        self.POSTGRES_USER = kwargs.pop("POSTGRES_USER", "default_user")
        self.POSTGRES_PASSWORD = kwargs.pop("POSTGRES_PASSWORD", "default_password")
        self.POSTGRES_HOST = kwargs.pop("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = kwargs.pop("POSTGRES_PORT", 5432)
        self.POSTGRES_DB = kwargs.pop("POSTGRES_DB", "default_db")
        super().__init__(**kwargs)
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine | None:
        """Get the engine."""
        return self._engine

    def setup(self) -> None:
        """Setup the model."""
        self.context.logger.info("Setting up the database model...")
        self.context.logger.info("Database model setup complete.")

        try:
            # Get DB connection parameters from context/config.
            db_user = self.POSTGRES_USER
            db_password = self.POSTGRES_PASSWORD
            db_host = self.POSTGRES_HOST
            db_port = self.POSTGRES_PORT
            db_name = self.POSTGRES_DB

            if not all([db_user, db_password, db_host, db_port, db_name]):
                msg = "Missing one or more required database configuration parameters."
                raise ValueError(msg)

            # Create a connection string.
            db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            self.context.logger.debug(f"Database URL: {db_url}")

            self._engine = create_engine(db_url, pool_size=5, max_overflow=10, pool_timeout=30, pool_recycle=1800)
            self.context.logger.info("Database engine created successfully.")

        except Exception as e:
            self.context.logger.exception(f"Failed to setup database engine: {e}")
            raise

    def teardown(self) -> None:
        """Teardown the database connection."""
        self.context.logger.info("Tearing down the database model...")
        self.context.logger.info("Database model teardown complete.")

        if self._engine:
            self._engine.dispose()
            self.context.logger.info("Database engine disposed successfully.")


class ScrapedDataItem:
    """Data structure for scraped items."""

    def __init__(
        self,
        source: str,
        title: str,
        url: str,
        summary: str = "",
        timestamp: str | None = None,
        target: str | None = None,
        scraped_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.source = source
        self.title = title
        self.url = url
        self.summary = summary
        self.timestamp = timestamp
        self.target = target
        self.scraped_at = scraped_at or datetime.now(UTC).isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert the item to a dictionary."""
        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "target": self.target,
            "scraped_at": self.scraped_at,
            "metadata": self.metadata,
        }
