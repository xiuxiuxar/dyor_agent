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

from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased
from aea.skills.base import Handler

from packages.eightballer.protocols.http.message import HttpMessage as ApiHttpMessage
from packages.xiuxiuxar.skills.simple_fsm.handlers import WebsocketHandler
from packages.eightballer.protocols.websockets.message import WebsocketsMessage as ApiWebsocketsMessage

from . import db, api


Asset = db.Asset
Report = db.Report
Trigger = db.Trigger
AssetResponse = api.AssetResponse
TriggerCreate = api.TriggerCreate
ReportResponse = api.ReportResponse
TriggerResponse = api.TriggerResponse


def handle_exception(handler_func):
    """Handle exception in the handler with standardized error responses."""

    def wrapper(self, message, *args, **kwargs):
        try:
            return handler_func(self, message, *args, **kwargs)
        except json.JSONDecodeError:
            self.context.logger.warning("Invalid JSON in request body")
            return self.error_response(
                message=message,
                error_msg="Invalid JSON in request body",
                status_code=400,
                error_code="BAD_REQUEST",
                details={"type": "json_decode_error"},
            )
        except ValueError as e:
            self.context.logger.warning(f"Validation error: {e!s}")
            return self.error_response(
                message=message,
                error_msg=str(e),
                status_code=400,
                error_code="BAD_REQUEST",
                details={"type": "validation_error"},
            )
        except Exception as e:
            self.context.logger.exception(f"Unhandled exception {e!s}")
            return self.error_response(
                message=message,
                error_msg="Internal server error",
                status_code=500,
                error_code="INTERNAL_SERVER_ERROR",
                details={"type": "unhandled_exception"},
            )

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

    def error_response(
        self,
        message: ApiHttpMessage,
        error_msg: str,
        status_code: int = 400,
        error_code: str | None = None,
        details: dict | None = None,
    ) -> ApiHttpMessage:
        """Create an error response following OpenAPI spec."""
        # Map status codes to error codes if not provided
        if error_code is None:
            error_code = {
                400: "BAD_REQUEST",
                401: "UNAUTHORIZED",
                403: "FORBIDDEN",
                404: "NOT_FOUND",
                429: "TOO_MANY_REQUESTS",
                500: "INTERNAL_SERVER_ERROR",
            }.get(status_code, "UNKNOWN_ERROR")

        error_body = {"code": error_code, "message": error_msg}
        if details:
            error_body["details"] = details

        return self.create_response(message, status_code, "Error", error_body)

    def not_found_response(self, message: ApiHttpMessage, resource_type: str, resource_id: str) -> ApiHttpMessage:
        """Create a not found response following OpenAPI spec."""
        return self.error_response(
            message=message,
            error_msg=f"{resource_type} with {resource_type}_id {resource_id} not found",
            status_code=404,
            error_code="NOT_FOUND",
        )

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

    def _parse_query_params(self, message: ApiHttpMessage) -> dict:
        """Parse and validate query parameters from the request."""
        params = {}
        query = urlparse(message.url).query
        if not query:
            return params

        query_dict = dict(param.split("=") for param in query.split("&") if "=" in param)

        params.update(self._parse_common_params(query_dict))
        params.update(self._parse_report_params(query_dict))
        params.update(self._parse_asset_params(query_dict))

        return params

    def _parse_common_params(self, query_dict: dict) -> dict:
        """Parse common pagination parameters."""
        params = {}
        for param in ["limit", "offset"]:
            if param in query_dict:
                try:
                    value = int(query_dict[param])
                    if param == "limit" and (value < 1 or value > 100):
                        msg = "limit must be between 1 and 100"
                        raise ValueError(msg)
                    if param == "offset" and value < 0:
                        msg = "offset must be non-negative"
                        raise ValueError(msg)
                    params[param] = value
                except ValueError as e:
                    msg = f"Invalid {param} parameter: {e}"
                    raise ValueError(msg) from e
        return params

    def _parse_report_params(self, query_dict: dict) -> dict:
        """Parse report-specific parameters."""
        params = {}
        if "asset_id" in query_dict:
            params["asset_id"] = query_dict["asset_id"]

        for date_param in ["start_date", "end_date"]:
            if date_param in query_dict:
                try:
                    params[date_param] = datetime.fromisoformat(query_dict[date_param])
                except ValueError:
                    msg = f"{date_param} must be in ISO 8601 format"
                    raise ValueError(msg) from None

        if "trigger_type" in query_dict:
            valid_types = {
                "volume_spike",
                "social_spike",
                "bollinger_band",
                "user_triggered",
                "moving_average_crossover",
            }
            if query_dict["trigger_type"] not in valid_types:
                msg = f"trigger_type must be one of: {', '.join(valid_types)}"
                raise ValueError(msg)
            params["trigger_type"] = query_dict["trigger_type"]

        return params

    def _parse_asset_params(self, query_dict: dict) -> dict:
        """Parse asset-specific parameters."""
        params = {}
        if "category" in query_dict:
            valid_categories = {"AI", "DeFi", "Memecoin", "Gaming", "Infrastructure", "Other"}
            if query_dict["category"] not in valid_categories:
                msg = f"category must be one of: {', '.join(valid_categories)}"
                raise ValueError(msg)
            params["category"] = query_dict["category"]

        if "has_report" in query_dict:
            if query_dict["has_report"].lower() not in {"true", "false"}:
                msg = "has_report must be 'true' or 'false'"
                raise ValueError(msg)
            params["has_report"] = query_dict["has_report"].lower() == "true"

        return params

    def _build_list_query(self, model_class: type, params: dict) -> select:
        """Build the query based on model type and parameters."""
        query = select(model_class)

        if model_class == Report:
            if "asset_id" in params:
                query = query.where(model_class.asset_id == params["asset_id"])
            if "start_date" in params:
                query = query.where(model_class.created_at >= params["start_date"])
            if "end_date" in params:
                query = query.where(model_class.created_at <= params["end_date"])
            if "trigger_type" in params:
                query = query.where(model_class.trigger_type == params["trigger_type"])
        elif model_class == Asset:
            if "category" in params:
                query = query.where(model_class.category == params["category"])
            if "has_report" in params:
                # Use a subquery to check if asset has any reports
                subquery = select(Report.asset_id).distinct()
                if params["has_report"]:
                    query = query.where(model_class.asset_id.in_(subquery))
                else:
                    query = query.where(~model_class.asset_id.in_(subquery))

        return query.order_by(model_class.created_at.desc())

    def _handle_list(self, model_class, response_class, message: ApiHttpMessage) -> ApiHttpMessage:
        """Handle list endpoints with query parameter support."""
        try:
            params = self._parse_query_params(message)
        except ValueError as e:
            return self.error_response(
                message=message,
                error_msg=str(e),
                status_code=400,
                error_code="BAD_REQUEST",
                details={"type": "query_parameter_validation"},
            )

        with self.get_session() as session:
            # Build base query
            query = self._build_list_query(model_class, params)

            # Get total count
            total_count = session.execute(select(func.count()).select_from(query.subquery())).scalar()

            # Apply pagination
            limit = params.get("limit", 10 if model_class == Report else 50)
            offset = params.get("offset", 0)
            query = query.limit(limit).offset(offset)

            # Execute query using existing pattern
            data, status_code, status_text = self.execute_db_query(
                session=session, query=query, model_class=response_class, is_list=True
            )

            if status_code != 200:
                return self.error_response(
                    message=message,
                    error_msg=status_text,
                    status_code=status_code,
                    error_code="INTERNAL_SERVER_ERROR" if status_code == 500 else "BAD_REQUEST",
                )

            # Construct pagination info
            base_url = urlparse(message.url)
            next_offset = offset + limit if offset + limit < total_count else None
            prev_offset = offset - limit if offset > 0 else None

            pagination = {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "next": f"{base_url.scheme}://{base_url.netloc}{base_url.path}?limit={limit}&offset={next_offset}"
                if next_offset is not None
                else None,
                "previous": f"{base_url.scheme}://{base_url.netloc}{base_url.path}?limit={limit}&offset={prev_offset}"
                if prev_offset is not None
                else None,
            }

            # Return response matching OpenAPI spec structure
            response_data = {"reports" if model_class == Report else "assets": data, "pagination": pagination}

            return self.success_response(message, response_data, status_code, status_text)

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

    def _handle_get_by_id(
        self, model_class, response_class, message: ApiHttpMessage, id_value: str, id_field: str
    ) -> ApiHttpMessage:
        with self.get_session() as session:
            # Special handling for Trigger: join Asset to get asset_symbol
            if model_class == Trigger:
                asset_alias = aliased(Asset)
                query = (
                    select(Trigger, asset_alias.symbol)
                    .join(asset_alias, Trigger.asset_id == asset_alias.asset_id)
                    .where(getattr(Trigger, id_field) == id_value)
                )
                result = session.execute(query).first()
                if result is None:
                    return self.not_found_response(message, model_class.__name__, id_value)
                trigger_obj, asset_symbol = result
                data = response_class.model_validate(trigger_obj).model_dump()
                data["asset_symbol"] = asset_symbol
                return self.success_response(message, data, 200, "OK")
            # Default for other models
            data, status_code, status_text = self.execute_db_query(
                session=session,
                query=select(model_class).where(getattr(model_class, id_field) == id_value),
                model_class=response_class,
                not_found_msg=f"{model_class.__name__} with {id_field} {{id}} not found",
                not_found_id=id_value,
            )
            if status_code == 404:
                return self.not_found_response(message, model_class.__name__, id_value)
            if status_code != 200:
                return self.error_response(
                    message=message,
                    error_msg=status_text,
                    status_code=status_code,
                    error_code="INTERNAL_SERVER_ERROR" if status_code == 500 else "BAD_REQUEST",
                )
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
            data = json.loads(body)
            # Validate input using TriggerCreate (which now allows asset_id or asset_symbol)
            trigger_create = TriggerCreate.model_validate(data)
            asset_id = trigger_create.asset_id
            asset_symbol = trigger_create.asset_symbol
            with self.get_session() as session:
                # If asset_symbol is provided, resolve or create asset
                if asset_symbol:
                    asset = session.execute(select(Asset).where(Asset.symbol == asset_symbol)).scalar_one_or_none()
                    if asset is None:
                        # Create new asset with symbol and a default name (use symbol as name if not provided)
                        asset_name = data.get("asset_name") or asset_symbol.upper()
                        new_asset = Asset(symbol=asset_symbol, name=asset_name)
                        session.add(new_asset)
                        session.commit()
                        session.refresh(new_asset)
                        asset_id = new_asset.asset_id
                    else:
                        asset_id = asset.asset_id
                elif asset_id is not None:
                    # Validate asset_id exists
                    asset = session.execute(select(Asset).where(Asset.asset_id == asset_id)).scalar_one_or_none()
                    if asset is None:
                        return self.not_found_response(message, "Asset", str(asset_id))
                else:
                    return self.error_response(
                        message=message,
                        error_msg="Either asset_id or asset_symbol must be provided.",
                        status_code=400,
                        error_code="BAD_REQUEST",
                    )
                # Prepare trigger creation dict
                trigger_dict = trigger_create.model_dump()
                trigger_dict["asset_id"] = asset_id
                trigger_dict.pop("asset_symbol", None)  # Not a DB column
                new_trigger = Trigger(**trigger_dict, processing_started_at=datetime.now(UTC))
                session.add(new_trigger)
                session.commit()
                session.refresh(new_trigger)
                data = TriggerResponse.model_validate(new_trigger).model_dump()
                return self.success_response(message, data, 202, "Report generation triggered successfully")
        except json.JSONDecodeError:
            return self.error_response(
                message=message,
                error_msg="Invalid JSON in request body",
                status_code=400,
                error_code="BAD_REQUEST",
                details={"type": "json_decode_error"},
            )
        except ValueError as e:
            return self.error_response(
                message=message,
                error_msg=str(e),
                status_code=400,
                error_code="BAD_REQUEST",
                details={"type": "validation_error"},
            )

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


class DyorWsHandler(WebsocketHandler):
    """Implements the WebSocket handler for DYOR API."""

    SUPPORTED_PROTOCOL = ApiWebsocketsMessage.protocol_id

    def handle(self, message: ApiWebsocketsMessage) -> None:
        """Handle incoming WebSocket messages with action-based routing."""
        try:
            data = json.loads(message.data)
            action = data.get("action")
            payload = data.get("payload", {})
            dialogue = self._find_dialogue_for_message(message)
            handler = getattr(self, f"handle_{action}", None)
            if handler:
                return handler(message, payload, dialogue)
            return self.ws_error(message, f"Unknown action: {action}", dialogue)
        except Exception as e:
            dialogue = self._find_dialogue_for_message(message)
            self.context.logger.exception(f"Error handling WebSocket message: {e}")
            return self.ws_error(message, f"Internal error: {e!s}", dialogue)

    def handle_ping(self, message, _payload, dialogue):
        """Respond to ping with pong."""
        return self.ws_success(message, "pong", {"timestamp": datetime.now(UTC).isoformat()}, dialogue)
