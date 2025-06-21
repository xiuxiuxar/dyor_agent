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

import os
import json
from typing import Any
from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor

import sqlalchemy
from sqlalchemy import text

from packages.xiuxiuxar.skills.dyor_app.data_sources import DATA_SOURCES, unlocks_fetcher
from packages.xiuxiuxar.skills.dyor_app.behaviours.base import BaseState, DyorabciappEvents, DyorabciappStates
from packages.xiuxiuxar.skills.dyor_app.trendmoon_client import TrendmoonAPIError
from packages.xiuxiuxar.skills.dyor_app.lookonchain_client import LookOnChainAPIError
from packages.xiuxiuxar.skills.dyor_app.treeofalpha_client import TreeOfAlphaAPIError
from packages.xiuxiuxar.skills.dyor_app.researchagent_client import ResearchAgentAPIError


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
                    # Clean up ongoing requests before setting error context
                    self._cleanup_futures()

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

    def _cleanup_futures(self) -> None:
        """Cancel ongoing futures and clean up executor state."""
        if self._futures:
            for name, future in self._futures.items():
                if not future.done():
                    future.cancel()
                    self.context.logger.info(f"Cancelled future: {name}")
            self._futures.clear()

        if self._request_queue:
            self._request_queue.clear()

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
