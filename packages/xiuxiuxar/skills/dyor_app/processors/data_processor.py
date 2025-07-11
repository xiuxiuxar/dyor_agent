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

"""Data processor for DYOR App skill."""

from typing import Any
from datetime import UTC, datetime

import sqlalchemy

from packages.xiuxiuxar.skills.dyor_app.utils import (
    serialize_for_storage,
)
from packages.xiuxiuxar.skills.dyor_app.constants import (
    REPR_LIMIT,
    RAW_DATA_TYPE,
    UNLOCKS_SOURCE,
    DEFAULT_DATA_TYPE,
)
from packages.xiuxiuxar.skills.dyor_app.data_sources import DATA_SOURCES


class DataProcessor:
    """Handles data processing pipeline for all data sources."""

    def __init__(self, context, logger):
        """Initialize with context and logger."""
        self.context = context
        self.logger = logger

    def get_source_config(self, source: str) -> dict | None:
        """Get configuration for a data source."""
        return DATA_SOURCES.get(source)

    def process_source_data(
        self,
        source: str,
        raw_data: Any,
        trigger_id: int,
        asset_id: int,
        unlock_processor,
    ) -> dict[str, str]:
        """Process data for a source, handling both single and multi data types."""
        config = self.get_source_config(source)
        if not config:
            return {source: "No processor configured for this source"}

        is_multi = config["data_type_handler"] == "multi"
        processor_func = getattr(self, config["processor"], None)

        # If processor doesn't exist on this class, try to get it from the context (backward compatibility)
        if processor_func is None:
            processor_func = getattr(self.context, config["processor"], None)

        if processor_func is None:
            return {source: f"Processor function {config['processor']} not found"}

        # Special handling for unlocks: filter for project and event type
        if source == UNLOCKS_SOURCE:
            return unlock_processor.process_unlocks_data(
                raw_data, processor_func, trigger_id, asset_id, self.store_processed_data
            )

        # Default: original logic
        return self.process_standard_data(source, raw_data, processor_func, trigger_id, asset_id, is_multi)

    def process_standard_data(
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

    def store_raw_data_by_type(
        self, source: str, serialized_raw: Any, trigger_id: int, asset_id: int, is_multi: bool
    ) -> None:
        """Store raw data, handling both single and multi data types."""
        if is_multi:
            if not isinstance(serialized_raw, dict):
                self.logger.warning(
                    f"Expected dict for multi data type in source={source}, got {type(serialized_raw)}. Skipping."
                )
                return

            for subkey, subval in serialized_raw.items():
                self.logger.info(
                    "Storing multi raw data for source=%s, subkey=%s, type(subval)=%s, subval=%s",
                    source,
                    subkey,
                    type(subval),
                    repr(subval)[:REPR_LIMIT],
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
                data_type=RAW_DATA_TYPE,
                data=serialized_raw,
                trigger_id=trigger_id,
                timestamp=datetime.now(tz=UTC),
                asset_id=asset_id,
            )

    def store_all_raw_data(self, trigger_id: int, asset_id: int, unlock_processor) -> None:
        """Store all raw data from different sources."""
        try:
            for source, config in DATA_SOURCES.items():
                raw = self.context.raw_data.get(source)

                # Special handling for unlocks: only store filtered data as raw
                if source == UNLOCKS_SOURCE:
                    unlock_processor.store_unlocks_raw_data(raw, trigger_id, asset_id)
                    continue

                serialized_raw = serialize_for_storage(raw)
                is_multi = config.get("data_type_handler") == "multi"

                self.logger.info(
                    "Storing raw data for source=%s, type=%s: "
                    "type(raw)=%s, type(serialized_raw)=%s, raw=%s, serialized_raw=%s",
                    source,
                    "multi" if is_multi else RAW_DATA_TYPE,
                    type(raw),
                    type(serialized_raw),
                    repr(raw)[:REPR_LIMIT],
                    repr(serialized_raw)[:REPR_LIMIT],
                )

                self.store_raw_data_by_type(source, serialized_raw, trigger_id, asset_id, is_multi)

        except (sqlalchemy.exc.SQLAlchemyError, TypeError, ValueError, AttributeError) as e:
            self.logger.warning(f"Failed to store raw data for debugging: {e}")

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
                    data_type=DEFAULT_DATA_TYPE,
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
            self.logger.info(
                "Storing processed data for source=%s, data_type=%s, type(data)=%s, data=%s",
                source,
                data_type,
                type(data),
                repr(data)[:REPR_LIMIT],
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
            self.logger.warning(f"Database error storing {source} data: {e}")
            raise
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Data serialization error for {source}: {e}")
            raise
        except AttributeError as e:
            self.logger.warning(f"Invalid data structure for {source}: {e}")
            raise

    def handle_processing_error(self, errors: dict, key: str, exc: Exception, error_type: str) -> None:
        """Handle processing errors and add to errors dict."""
        errors[key] = f"{error_type}: {exc!s}"
        self.logger.warning(f"{error_type} for {key}: {exc}")
