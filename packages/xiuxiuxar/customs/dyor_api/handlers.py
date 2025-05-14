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

"""This package contains a scaffold of a handler."""

import re
import json
from datetime import UTC, datetime
from contextlib import contextmanager
from urllib.parse import unquote, urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session
from aea.skills.base import Handler

from packages.eightballer.protocols.http.message import HttpMessage as ApiHttpMessage

from . import db, api


Asset = db.Asset
Report = db.Report
Trigger = db.Trigger
AssetResponse = api.AssetResponse
TriggerCreate = api.TriggerCreate
ReportResponse = api.ReportResponse
TriggerResponse = api.TriggerResponse


def handle_exception(handler_func):
    """Handle exception in the handler."""

    def wrapper(self, message, *args, **kwargs):
        try:
            return handler_func(self, message, *args, **kwargs)
        except json.JSONDecodeError:
            self.context.logger.warning("Invalid JSON in request body")
            return self.error_response(message, "Invalid JSON in request body", 400)
        except ValueError as e:
            self.context.logger.warning(f"Validation error: {e!s}")
            return self.error_response(message, str(e), 400)
        except Exception as e:
            self.context.logger.exception(f"Unhandled exception {e!s}")
            return self.error_response(message, "Internal server error", 500)

    return wrapper


class DyorApiHandler(Handler):
    """Implements the API HTTP handler."""

    SUPPORTED_PROTOCOL = ApiHttpMessage.protocol_id

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _db_model(self):
        return self.context.db_model

    def setup(self) -> None:
        """Set up the handler."""
        self.context.logger.info("Setting up DyorApiHandler...")
        self.context.logger.info("DyorApiHandler setup complete")

    def teardown(self) -> None:
        """Tear down the handler."""

    @handle_exception
    def handle(self, message: ApiHttpMessage) -> None:
        """Handle incoming API HTTP messages."""
        method = message.method.lower()
        parsed_url = urlparse(unquote(message.url))
        path = parsed_url.path
        body = message.body

        self.context.logger.info(f"Received {method.upper()} request for {path}")

        normalized_path = self.normalize_path(path)

        handler_name, kwargs = self.get_handler_name_and_kwargs(method, normalized_path, path, body)

        handler_method = getattr(self, handler_name, None)

        if handler_method:
            self.context.logger.debug(f"Found handler method: {handler_name}")
            return handler_method(message, **kwargs)
        self.context.logger.warning(f"No handler found for {method.upper()} request to {path}")
        return self.handle_unexpected_message(message)

    def normalize_path(self, path: str) -> str:
        """Normalize the path using regex substitution."""
        normalized_path = path.rstrip("/")
        self.context.logger.debug(f"Normalized path: {normalized_path}")

        substitutions = {
            r"^/api/reports/(?P<report_id>[^/]+)$": "/api/reports/report_id",
            r"^/api/reports/asset/(?P<asset_id>[^/]+)$": "/api/reports/asset/asset_id",
            r"^/api/trigger/(?P<trigger_id>[^/]+)$": "/api/trigger/trigger_id",
            r"^/api/assets/(?P<asset_id>[^/]+)$": "/api/assets/asset_id",
        }

        for pattern, replacement in substitutions.items():
            normalized_path = re.sub(pattern, replacement, normalized_path)

        self.context.logger.debug(f"After regex substitutions: {normalized_path}")
        return normalized_path

    def get_handler_name_and_kwargs(
        self, method: str, normalized_path: str, original_path: str, body: bytes
    ) -> tuple[str, dict]:
        """Get the handler name and kwargs for the given method and path."""
        handler_name = f"handle_{method}_{normalized_path.lstrip('/').replace('/', '_')}"

        self.context.logger.debug(f"Initial handler name: {handler_name}")
        handler_name = handler_name.replace("report_id", "by_report_id")
        handler_name = handler_name.replace("asset_id", "by_asset_id")
        handler_name = handler_name.replace("trigger_id", "by_trigger_id")
        self.context.logger.debug(f"Final handler name: {handler_name}")

        kwargs = {"body": body} if method in {"post", "put", "patch"} else {}
        patterns = [
            (r"^/api/reports/(?P<report_id>[^/]+)$", ["report_id"]),
            (r"^/api/reports/asset/(?P<asset_id>[^/]+)$", ["asset_id"]),
            (r"^/api/trigger/(?P<trigger_id>[^/]+)$", ["trigger_id"]),
            (r"^/api/assets/(?P<asset_id>[^/]+)$", ["asset_id"]),
        ]

        for pattern, param_names in patterns:
            match = re.search(pattern, original_path)
            if match:
                for param_name in param_names:
                    kwargs[param_name] = match.group(param_name)
                break
        self.context.logger.debug(f"Final kwargs: {kwargs}")
        return handler_name, kwargs

    def create_response(self, message, status_code, status_text, body):
        """Create an ApiHttpMessage response."""

        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            msg = f"Object of type {type(obj)} is not JSON serializable"
            raise TypeError(msg)

        return ApiHttpMessage(
            performative=ApiHttpMessage.Performative.RESPONSE,
            status_code=status_code,
            status_text=status_text,
            headers="",
            version=message.version,
            body=json.dumps(body, default=datetime_handler).encode(),
        )

    def success_response(
        self, message: ApiHttpMessage, data: dict | list, status_code: int = 200, status_text: str = "OK"
    ) -> ApiHttpMessage:
        """Create a success response."""
        return self.create_response(message, status_code, status_text, data)

    def error_response(self, message: ApiHttpMessage, error_msg: str, status_code: int = 400) -> ApiHttpMessage:
        """Create an error response."""
        return self.create_response(message, status_code, "Error", {"error": error_msg})

    def not_found_response(self, message: ApiHttpMessage, resource_type: str, resource_id: str) -> ApiHttpMessage:
        """Create a not found response."""
        return self.error_response(message, f"{resource_type} with {resource_type}_id {resource_id} not found", 404)

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = Session(self._db_model.engine)
        try:
            yield session
        finally:
            session.close()

    def execute_db_query(
        self,
        session: Session,
        query,
        model_class,
        not_found_msg: str | None = None,
        not_found_id: str | None = None,
        is_list: bool = False,
    ) -> tuple[dict | list | None, int, str]:
        """Execute a database query and convert results to response model."""
        if is_list:
            results = session.execute(query).scalars().all()
            data = [model_class.model_validate(item).model_dump() for item in results]
            return data, 200, "OK"

        result = session.execute(query).scalar_one_or_none()
        if result is None and not_found_msg:
            return None, 404, not_found_msg.format(id=not_found_id)

        data = model_class.model_validate(result).model_dump()
        return data, 200, "OK"

    # Internal route mapping
    _ROUTE_MAP = {
        "GET": {
            "/api/reports": lambda self, msg: self._handle_list(Report, ReportResponse, msg),
            "/api/reports/report_id": lambda self, msg, report_id: self._handle_get_by_id(
                Report, ReportResponse, msg, report_id, "report_id"
            ),
            "/api/reports/asset/asset_id": lambda self, msg, asset_id: self._handle_get_latest_by_asset(
                Report, ReportResponse, msg, asset_id
            ),
            "/api/trigger/trigger_id": lambda self, msg, trigger_id: self._handle_get_by_id(
                Trigger, TriggerResponse, msg, trigger_id, "trigger_id"
            ),
            "/api/assets": lambda self, msg: self._handle_list(Asset, AssetResponse, msg),
            "/api/assets/asset_id": lambda self, msg, asset_id: self._handle_get_by_id(
                Asset, AssetResponse, msg, asset_id, "asset_id"
            ),
        },
        "POST": {
            "/api/trigger": lambda self, msg, body: self._handle_create_trigger(msg, body),
        },
    }

    def _handle_list(self, model_class, response_class, message: ApiHttpMessage) -> ApiHttpMessage:
        with self.get_session() as session:
            data, status_code, status_text = self.execute_db_query(
                session=session,
                query=select(model_class).order_by(model_class.created_at.desc()),
                model_class=response_class,
                is_list=True,
            )
            return self.success_response(message, data, status_code, status_text)

    def _handle_get_by_id(
        self, model_class, response_class, message: ApiHttpMessage, id_value: str, id_field: str
    ) -> ApiHttpMessage:
        with self.get_session() as session:
            data, status_code, status_text = self.execute_db_query(
                session=session,
                query=select(model_class).where(getattr(model_class, id_field) == id_value),
                model_class=response_class,
                not_found_msg=f"{model_class.__name__} with {id_field} {{id}} not found",
                not_found_id=id_value,
            )
            if status_code == 404:
                return self.not_found_response(message, model_class.__name__, id_value)
            return self.success_response(message, data, status_code, status_text)

    def _handle_get_latest_by_asset(
        self, model_class, response_class, message: ApiHttpMessage, asset_id: str
    ) -> ApiHttpMessage:
        with self.get_session() as session:
            data, status_code, status_text = self.execute_db_query(
                session=session,
                query=select(model_class)
                .where(model_class.asset_id == asset_id)
                .order_by(model_class.created_at.desc())
                .limit(1),
                model_class=response_class,
                not_found_msg=f"No {model_class.__name__} found for asset_id {{id}}",
                not_found_id=asset_id,
            )
            if status_code == 404:
                return self.error_response(message, f"No {model_class.__name__} found for asset_id {asset_id}", 404)
            return self.success_response(message, data, status_code, status_text)

    def _handle_create_trigger(self, message: ApiHttpMessage, body: bytes) -> ApiHttpMessage:
        try:
            trigger_create = TriggerCreate.model_validate(json.loads(body))
            with self.get_session() as session:
                new_trigger = Trigger(**trigger_create.model_dump(), processing_started_at=datetime.now(UTC))
                session.add(new_trigger)
                session.commit()
                session.refresh(new_trigger)
                data = TriggerResponse.model_validate(new_trigger).model_dump()
                return self.success_response(message, data, 201, "Trigger created")
        except json.JSONDecodeError:
            return self.error_response(message, "Invalid JSON in request body")
        except ValueError as e:
            return self.error_response(message, str(e))

    # Public handler methods (kept for compatibility)
    @handle_exception
    def handle_get_api_reports(self, message: ApiHttpMessage) -> ApiHttpMessage:
        """Handle GET request for /api/reports."""
        return self._ROUTE_MAP["GET"]["/api/reports"](self, message)

    @handle_exception
    def handle_get_api_reports_by_report_id(self, message: ApiHttpMessage, report_id: str) -> ApiHttpMessage:
        """Handle GET request for /api/reports/report_id."""
        return self._ROUTE_MAP["GET"]["/api/reports/report_id"](self, message, report_id)

    @handle_exception
    def handle_get_api_reports_asset_by_asset_id(self, message: ApiHttpMessage, asset_id: str) -> ApiHttpMessage:
        """Handle GET request for /api/reports/asset/asset_id."""
        return self._ROUTE_MAP["GET"]["/api/reports/asset/asset_id"](self, message, asset_id)

    @handle_exception
    def handle_post_api_trigger(self, message: ApiHttpMessage, body: bytes) -> ApiHttpMessage:
        """Handle POST request for /api/trigger."""
        return self._ROUTE_MAP["POST"]["/api/trigger"](self, message, body)

    @handle_exception
    def handle_get_api_trigger_by_trigger_id(self, message: ApiHttpMessage, trigger_id: str) -> ApiHttpMessage:
        """Handle GET request for /api/trigger/trigger_id."""
        return self._ROUTE_MAP["GET"]["/api/trigger/trigger_id"](self, message, trigger_id)

    @handle_exception
    def handle_get_api_assets(self, message: ApiHttpMessage) -> ApiHttpMessage:
        """Handle GET request for /api/assets."""
        return self._ROUTE_MAP["GET"]["/api/assets"](self, message)

    @handle_exception
    def handle_get_api_assets_by_asset_id(self, message: ApiHttpMessage, asset_id: str) -> ApiHttpMessage:
        """Handle GET request for /api/assets/asset_id."""
        return self._ROUTE_MAP["GET"]["/api/assets/asset_id"](self, message, asset_id)

    def handle_unexpected_message(self, message):
        """Handler for unexpected messages."""
        self.context.logger.info(f"Received unexpected message: {message}")
