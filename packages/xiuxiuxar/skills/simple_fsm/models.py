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

import time
from typing import Any
from datetime import UTC, datetime

from openai import OpenAI, OpenAIError, APIStatusError, RateLimitError, APITimeoutError, APIConnectionError
from sqlalchemy import text, create_engine
from aea.skills.base import Model
from psycopg2.extras import Json
from sqlalchemy.engine import Engine


class APIClientStrategy(Model):
    """This class represents a api client strategy."""

    http_handlers: list = []
    ws_handlers: list = []
    routes: dict = {}
    clients: dict = {}


class DatabaseModel(Model):
    """Database connection and configuration."""

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model."""
        self.POSTGRES_USER = kwargs.pop("POSTGRES_USER", None)
        self.POSTGRES_PASSWORD = kwargs.pop("POSTGRES_PASSWORD", None)
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

    def store_raw_data(
        self,
        source: str,
        data: Any,
        trigger_id: str,
        timestamp: datetime,
        data_type: str,
        asset_id: int,
    ) -> None:
        """Store raw data in the scraped_data table.

        Args:
        ----
            source: The data source (e.g., 'trendmoon', 'lookonchain')
            data: The raw data to store
            trigger_id: ID of the trigger that initiated the data collection
            timestamp: When the data was collected
            data_type: Type of the data (e.g., 'social', 'coin_details')
            asset_id: ID of the asset this data relates to

        """
        if not self._engine:
            msg = "Database engine not initialized. Call setup() first."
            raise RuntimeError(msg)

        try:
            # Convert data to JSON-serializable format
            if hasattr(data, "to_dict"):
                data = data.to_dict()

            # Construct the insert query
            query = text("""
                INSERT INTO scraped_data (
                    trigger_id,
                    asset_id,
                    source,
                    data_type,
                    raw_data,
                    ingested_at
                ) VALUES (
                    :trigger_id,
                    :asset_id,
                    :source,
                    :data_type,
                    :raw_data,
                    :ingested_at
                )
            """)

            # Execute the insert
            with self._engine.connect() as conn:
                conn.execute(
                    query,
                    {
                        "trigger_id": trigger_id,
                        "asset_id": asset_id,
                        "source": source,
                        "data_type": data_type,
                        "raw_data": Json(data),  # SQLAlchemy/psycopg2 will handle JSON conversion
                        "ingested_at": timestamp,
                    },
                )
                conn.commit()

            self.context.logger.debug(
                f"Stored scraped data: source={source}, type={data_type}, " f"trigger={trigger_id}, asset={asset_id}"
            )

        except Exception as e:
            self.context.logger.exception(
                f"Failed to store scraped data: source={source}, type={data_type}, "
                f"trigger={trigger_id}, asset={asset_id}, error={e!s}"
            )
            raise

    def create_trigger(
        self,
        asset_id: int,
        trigger_type: str,
        trigger_details: dict[str, Any] | None = None,
    ) -> int:
        """Create a new trigger in the database.

        Args:
        ----
            asset_id: ID of the asset this trigger is for
            trigger_type: Type of trigger (e.g., 'manual', 'scheduled')
            trigger_details: Optional JSON details for the trigger

        Returns:
        -------
            The ID of the created trigger

        """
        if not self._engine:
            msg = "Database engine not initialized. Call setup() first."
            raise RuntimeError(msg)

        try:
            query = text("""
                INSERT INTO triggers (
                    asset_id,
                    trigger_type,
                    trigger_details,
                    status,
                    processing_started_at
                ) VALUES (
                    :asset_id,
                    :trigger_type,
                    :trigger_details,
                    'processing',
                    NOW()
                )
                RETURNING trigger_id
            """)

            with self._engine.connect() as conn:
                result = conn.execute(
                    query,
                    {
                        "asset_id": asset_id,
                        "trigger_type": trigger_type,
                        "trigger_details": Json(trigger_details) if trigger_details else None,
                    },
                )
                trigger_id = result.scalar_one()
                conn.commit()
                return trigger_id

        except Exception as e:
            self.context.logger.exception(f"Failed to create trigger: {e!s}")
            raise

    def get_structured_payload(self, trigger_id: int, asset_id: int) -> dict | None:
        """Fetch the structured payload (report_data_json) for a given trigger and asset."""
        if not self._engine:
            msg = "Database engine not initialized. Call setup() first."
            raise RuntimeError(msg)

        try:
            query = text("""
                SELECT raw_data
                FROM scraped_data
                WHERE source = 'structured' AND trigger_id = :trigger_id AND asset_id = :asset_id
                ORDER BY ingested_at DESC
                LIMIT 1
            """)
            with self._engine.connect() as conn:
                result = conn.execute(query, {"trigger_id": trigger_id, "asset_id": asset_id}).fetchone()
                if result:
                    return result[0]
                return None
        except Exception as e:
            self.context.logger.exception(
                f"Failed to fetch structured payload for trigger_id={trigger_id}, asset_id={asset_id}: {e!s}"
            )
            raise

    def store_report(
        self,
        trigger_id: int,
        asset_id: int,
        report_content_markdown: str,
        report_data_json: dict,
        llm_model_used: str,
        generation_time_ms: int,
        token_usage: dict,
    ) -> int:
        """Store a generated report in the reports table.

        Args:
        ----
            trigger_id: ID of the trigger for this report
            asset_id: ID of the asset
            report_content_markdown: The report content in Markdown
            report_data_json: The structured payload (as dict)
            llm_model_used: The LLM model used for generation
            generation_time_ms: Time taken to generate the report (ms)
            token_usage: Token usage statistics (as dict)

        Returns:
        -------
            The ID of the created report

        """
        if not self._engine:
            msg = "Database engine not initialized. Call setup() first."
            raise RuntimeError(msg)

        try:
            query = text("""
                INSERT INTO reports (
                    trigger_id,
                    asset_id,
                    report_content_markdown,
                    report_data_json,
                    llm_model_used,
                    generation_time_ms,
                    token_usage,
                    created_at
                ) VALUES (
                    :trigger_id,
                    :asset_id,
                    :report_content_markdown,
                    :report_data_json,
                    :llm_model_used,
                    :generation_time_ms,
                    :token_usage,
                    NOW()
                )
                RETURNING report_id
            """)
            with self._engine.connect() as conn:
                result = conn.execute(
                    query,
                    {
                        "trigger_id": trigger_id,
                        "asset_id": asset_id,
                        "report_content_markdown": report_content_markdown,
                        "report_data_json": Json(report_data_json),
                        "llm_model_used": llm_model_used,
                        "generation_time_ms": generation_time_ms,
                        "token_usage": Json(token_usage),
                    },
                )
                report_id = result.scalar_one()
                conn.commit()
                return report_id
        except Exception as e:
            self.context.logger.exception(
                f"Failed to store report: trigger_id={trigger_id}, asset_id={asset_id}, error={e!s}"
            )
            raise

    def report_exists(self, trigger_id: int) -> bool:
        """Check if a report already exists for this trigger ID."""
        if not self._engine:
            msg = "Database engine not initialized. Call setup() first."
            raise RuntimeError(msg)

        try:
            query = text("""
                SELECT EXISTS (
                    SELECT 1 FROM reports WHERE trigger_id = :trigger_id
                )
            """)
            with self._engine.connect() as conn:
                result = conn.execute(query, {"trigger_id": trigger_id}).scalar()
                return bool(result)
        except Exception as e:
            self.context.logger.exception(f"Failed to check if report exists for trigger_id={trigger_id}: {e!s}")
            raise


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""


class LLMRateLimitError(LLMServiceError):
    """Raised on rate limit errors."""


class LLMAPIError(LLMServiceError):
    """Raised on API errors."""


class LLMContentFilterError(LLMServiceError):
    """Raised on content filtering errors."""


class LLMInvalidResponseError(LLMServiceError):
    """Raised on invalid or unexpected responses."""


class LLMService(Model):
    """LLM Service for OpenAI-compatible APIs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.primary_model = kwargs.pop("LLM_PRIMARY_MODEL", "primary_model")
        self.fallback_model = kwargs.pop("LLM_FALLBACK_MODEL", "fallback_model")
        self.api_key = kwargs.pop("LLM_API_KEY", "api_key")
        self.base_url = kwargs.pop("LLM_BASE_URL", "https://api.openai.com/v1")
        self.max_retries = kwargs.pop("LLM_MAX_RETRIES", 4)
        self.backoff_factor = kwargs.pop("LLM_BACKOFF_FACTOR", 1.5)
        self.timeout = kwargs.pop("LLM_TIMEOUT", 60)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_summary(self, prompt: str, model_config: dict) -> dict:
        """Generate a summary using the configured LLM backend.
        Returns dict: {
            "content": str,
            "llm_model_used": str,
            "generation_time_ms": int,
            "token_usage": dict
        }.
        """
        models_to_try = [self.primary_model]
        if self.fallback_model and self.fallback_model != self.primary_model:
            models_to_try.append(self.fallback_model)

        last_exception = None
        for model in models_to_try:
            try:
                return self._call_llm(prompt, model, model_config)
            except (LLMRateLimitError, LLMAPIError, LLMContentFilterError, LLMInvalidResponseError) as e:
                self.context.logger.warning(f"LLM call failed for model '{model}': {e}")
                last_exception = e
                continue
        raise last_exception or LLMServiceError("LLM call failed for all models.")

    def _call_llm(self, prompt: str, model: str, model_config: dict) -> dict:
        retries = 0
        while retries <= self.max_retries:
            try:
                start_time = time.monotonic()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=model_config.get("temperature", 0.7),
                    max_tokens=model_config.get("max_tokens", 512),
                    top_p=model_config.get("top_p", 1.0),
                    frequency_penalty=model_config.get("frequency_penalty", 0.0),
                    presence_penalty=model_config.get("presence_penalty", 0.0),
                    timeout=self.timeout,
                )
                latency = time.monotonic() - start_time
                generation_time_ms = int(latency * 1000)

                content = response.choices[0].message.content.strip()
                usage = getattr(response, "usage", None)
                token_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

                self.context.logger.info(
                    f"LLM call success | model={model} | latency={latency:.2f}s | "
                    f"prompt_tokens={token_usage['prompt_tokens']} | "
                    f"completion_tokens={token_usage['completion_tokens']} | "
                    f"total_tokens={token_usage['total_tokens']}"
                )
                return {
                    "content": content,
                    "llm_model_used": model,
                    "generation_time_ms": generation_time_ms,
                    "token_usage": token_usage,
                }

            except RateLimitError as e:
                self.context.logger.warning(f"Rate limit error (429): {e}")
                if retries == self.max_retries:
                    msg = "Rate limit exceeded"
                    raise LLMRateLimitError(msg) from e
                self._backoff(retries)
                retries += 1
            except (APIConnectionError, APITimeoutError, APIStatusError) as e:
                self.context.logger.warning(f"Transient API error: {e}")
                if retries == self.max_retries:
                    msg = "API connection or timeout error"
                    raise LLMAPIError(msg) from e
                self._backoff(retries)
                retries += 1
            except OpenAIError as e:
                # Content filtering or other OpenAI errors
                if "content_filter" in str(e).lower():
                    msg = "Content filtered by LLM provider"
                    raise LLMContentFilterError(msg) from e
                msg = f"OpenAI API error: {e}"
                raise LLMAPIError(msg) from e
            except Exception as e:
                self.context.logger.exception(f"Unexpected error from LLM: {e}")
                msg = f"Unexpected LLM error: {e}"
                raise LLMInvalidResponseError(msg) from e

        msg = "Max retries exceeded for LLM call."
        raise LLMServiceError(msg)

    def _backoff(self, retries: int):
        delay = self.backoff_factor * (2**retries)
        self.context.logger.info(f"Retrying after {delay:.2f}s (retry {retries + 1})")
        time.sleep(delay)

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Static pricing table (USD per 1K tokens). Extend as needed.
        pricing = {
            "gpt-4-turbo": 0.01,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.001,
            "Meta-Llama-3-1-8B-Instruct-FP8": 0.0005,
            "Meta-Llama-3-70B-Instruct": 0.002,
            "DeepSeek-R1": 0.001,
            "DeepSeek-R1-Distill-Llama-70B": 0.001,
            "DeepSeek-R1-Distill-Qwen-14B": 0.001,
            "DeepSeek-R1-Distill-Qwen-32B": 0.001,
            "Meta-Llama-3-2-3B-Instruct": 0.0005,
            "Meta-Llama-4-Maverick-17B-128E-Instruct-FP8": 0.001,
        }
        price_per_1k = pricing.get(model, 0.001)
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        return (total_tokens / 1000.0) * price_per_1k


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
