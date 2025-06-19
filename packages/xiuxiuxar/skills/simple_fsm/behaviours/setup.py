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

"""This module contains the implementation of the behaviours of 'simple_fsm' skill."""

import sys
from typing import TYPE_CHECKING, Any, cast
from pathlib import Path
from collections.abc import Generator

import yaml
import requests
import sqlalchemy

from packages.xiuxiuxar.skills.simple_fsm.behaviours.base import (
    PROTOCOL_HANDLER_MAP,
    BaseState,
    DyorabciappEvents,
    DyorabciappStates,
    dynamic_import,
)


if TYPE_CHECKING:
    from packages.xiuxiuxar.skills.simple_fsm.models import APIClientStrategy


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
