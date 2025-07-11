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
from datetime import UTC, datetime

import requests
import sqlalchemy
from pydantic import ValidationError
from sqlalchemy import text

from packages.xiuxiuxar.skills.dyor_app.utils import (
    safe_get_nested,
    serialize_for_storage,
)
from packages.xiuxiuxar.skills.dyor_app.constants import (
    DEFAULT_DATA_TYPE,
    STRUCTURED_SOURCE,
)
from packages.xiuxiuxar.skills.dyor_app.data_sources import DATA_SOURCES
from packages.xiuxiuxar.skills.dyor_app.behaviours.base import BaseState, DyorabciappEvents, DyorabciappStates
from packages.xiuxiuxar.skills.dyor_app.processors.data_processor import DataProcessor
from packages.xiuxiuxar.skills.dyor_app.processors.payload_builder import PayloadBuilder
from packages.xiuxiuxar.skills.dyor_app.processors.unlock_processor import UnlockProcessor
from packages.xiuxiuxar.skills.dyor_app.processors.metrics_calculator import MetricsCalculator


class ProcessDataRound(BaseState):
    """This class implements the behaviour of the state ProcessDataRound."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.PROCESSDATAROUND

        # Initialize processors with dependency injection
        self.unlock_processor = UnlockProcessor(self.context, self.context.logger)
        self.metrics_calculator = MetricsCalculator(self.context, self.context.logger)
        self.data_processor = DataProcessor(self.context, self.context.logger)
        self.payload_builder = PayloadBuilder(self.context, self.context.logger)

    # Serialization methods (delegated to processors)
    def serialize_unlocks_data(self, data):
        """Serialize unlocks data using unlock processor."""
        return self.unlock_processor.serialize_unlocks_data(data)

    def serialize_trendmoon_data(self, data):
        """Serialize trendmoon data."""
        return serialize_for_storage(data)

    def serialize_lookonchain_data(self, data):
        """Serialize lookonchain data."""
        return serialize_for_storage(data)

    def serialize_treeofalpha_data(self, data):
        """Serialize treeofalpha data."""
        return serialize_for_storage(data)

    def serialize_researchagent_data(self, data):
        """Serialize researchagent data."""
        return serialize_for_storage(data)

    # Data processing methods (delegated to processors)
    def process_source_data(self, source: str, raw_data: Any, trigger_id: int, asset_id: int) -> dict[str, str]:
        """Process data for a source using data processor."""
        return self.data_processor.process_source_data(source, raw_data, trigger_id, asset_id, self.unlock_processor)

    def _validate_data_sources(self) -> None:
        """Validate and log missing or errored data sources."""
        expected_sources = set(DATA_SOURCES.keys())

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

    def _update_asset_metadata_from_trendmoon(self, asset_id: int) -> None:
        """Update asset metadata from trendmoon data."""
        trendmoon_raw = self.context.raw_data.get("trendmoon", {})
        coin_details = safe_get_nested(trendmoon_raw, "coin_details", default={})

        coingecko_id = coin_details.get("id")
        category = safe_get_nested(coin_details, "categories", 0)
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

    def _process_and_store_data(self, trigger_id: int, asset_id: int) -> None:
        """Process and store all data for the trigger."""
        self.data_processor.store_all_raw_data(trigger_id, asset_id, self.unlock_processor)
        self._update_asset_metadata_from_trendmoon(asset_id)

        payload = self.payload_builder.try_build_structured_payload(self.metrics_calculator, self.unlock_processor)
        if payload is None:
            msg = "Failed to build structured payload"
            raise ValidationError(msg)
        if not self._store_structured_payload(payload, trigger_id, asset_id):
            msg = "Failed to store structured payload"
            raise RuntimeError(msg)

    def _store_structured_payload(self, payload, trigger_id, asset_id):
        """Store the structured payload in the database."""
        try:
            serialized_payload = serialize_for_storage(payload)
            self.context.db_model.store_raw_data(
                source=STRUCTURED_SOURCE,
                data_type=DEFAULT_DATA_TYPE,
                data=serialized_payload,
                trigger_id=trigger_id,
                timestamp=datetime.now(tz=UTC),
                asset_id=asset_id,
            )
            self.context.logger.info(f"Stored structured payload for trigger {trigger_id}")
            return True
        except (sqlalchemy.exc.SQLAlchemyError, TypeError, ValueError) as e:
            self.context.logger.exception(f"Failed to store structured payload: {e}")
            return False

    # Error handling methods
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
