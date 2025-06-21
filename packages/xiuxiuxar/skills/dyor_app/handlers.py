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

"""This module contains the handler for the DYOR App skill."""

import json
import secrets
from enum import Enum
from typing import TYPE_CHECKING, Any, cast
from datetime import UTC, datetime
from urllib.parse import urlparse

from aea.skills.base import Handler
from aea.protocols.base import Message

from packages.xiuxiuxar.skills.dyor_app import PUBLIC_ID
from packages.eightballer.protocols.default import DefaultMessage
from packages.eightballer.protocols.http.message import HttpMessage
from packages.xiuxiuxar.skills.dyor_app.dialogues import (
    HttpDialogue,
    HttpDialogues,
    DefaultDialogues,
    WebsocketsDialogue,
    ApiWebSocketDialogues,
)
from packages.eightballer.protocols.websockets.message import WebsocketsMessage


if TYPE_CHECKING:
    from packages.xiuxiuxar.skills.dyor_app.models import APIClientStrategy


def _parse_headers(header_str: str) -> dict[str, str]:
    """Parses a string of HTTP headers into a dictionary."""
    headers = {}
    for line in header_str.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return headers


class SystemStatus(Enum):
    """System status Enum."""

    OPERATIONAL = "operational"
    ERROR = "error"
    INITIALIZING = "initializing"


class HttpHandler(Handler):
    """This implements the echo handler."""

    SUPPORTED_PROTOCOL = HttpMessage.protocol_id
    SUPPORTED_METHODS = {"get"}
    SYSTEM_ENDPOINTS = {"/status", "/metrics"}
    WS_ENDPOINT = "/ws"

    def __init__(self, **kwargs):
        """Initialise the handler."""
        self.enable_cors = kwargs.pop("enable_cors", False)
        self.expected_api_key = kwargs.pop("expected_api_key", None)
        super().__init__(**kwargs)

    def setup(self) -> None:
        """Implement the setup."""

    @property
    def strategy(self) -> str | None:
        """Get the strategy."""
        return cast("APIClientStrategy", self.context.api_client_strategy)

    def handle(self, message: Message) -> None:
        """Handle the message."""
        self.context.logger.debug("Handling new connection message in skill")
        http_msg = cast("HttpMessage", message)
        http_dialogues = cast("HttpDialogues", self.context.http_dialogues)
        http_dialogue = cast("HttpDialogue", http_dialogues.update(http_msg))

        if http_dialogue is None:
            self._handle_unidentified_dialogue(http_msg)
            return

        if http_msg.performative != HttpMessage.Performative.REQUEST:
            self._handle_invalid(http_msg, http_dialogue)
            return

        # Validate API key for all requests
        headers = _parse_headers(http_msg.headers)
        provided_key = headers.get("x-api-key")
        if not self._is_valid_api_key(provided_key):
            self._handle_unauthorized(http_msg, http_dialogue)
            return

        method = http_msg.method.lower()
        url_path = self._normalize_path(http_msg.url)

        # Handle WebSocket upgrade request
        if url_path == self.WS_ENDPOINT and method == "get" and headers.get("upgrade", "").lower() == "websocket":
            # Let the WebSocket server handle the upgrade
            return

        # Route to API handler if path starts with /api
        if url_path.startswith("/api/"):
            for handler in self.strategy.http_handlers:
                result = handler.handle(message)
                if result is not None:
                    self._send_response(
                        http_dialogue,
                        http_msg,
                        result.status_code,
                        json.loads(result.body.decode("utf-8")),
                        result.status_text,
                    )
                    return

            self._handle_invalid(http_msg, http_dialogue)
            return

        # Handle system endpoints
        if url_path in self.SYSTEM_ENDPOINTS:
            if method not in self.SUPPORTED_METHODS:
                self._handle_invalid(http_msg, http_dialogue)
            elif url_path == "/status":
                self._handle_status(http_msg, http_dialogue)
        else:
            self._handle_invalid(http_msg, http_dialogue)

    def _normalize_path(self, url: str) -> str:
        """Normalize URL path."""
        path = urlparse(url).path.rstrip("/")
        return path or "/"

    def _get_system_status(self) -> tuple[SystemStatus, str]:
        """Get current system status and FSM state."""
        fsm_behaviour = self.context.behaviours.main
        current_state = fsm_behaviour.current or "initializing"

        if current_state == "handleerrorround":
            return SystemStatus.ERROR, current_state
        if current_state == "setupdyorround":
            return SystemStatus.INITIALIZING, current_state
        return SystemStatus.OPERATIONAL, current_state

    def _build_response_headers(self, original_headers: str | None = None) -> str:
        """Build response headers with CORS if enabled."""
        headers = ["Content-Type: application/json"]

        if self.enable_cors:
            headers.extend(
                [
                    "Access-Control-Allow-Origin: *",
                    "Access-Control-Allow-Methods: GET",
                    "Access-Control-Allow-Headers: Content-Type,Accept",
                ]
            )

        if original_headers:
            headers.append(original_headers)

        return "\n".join(headers)

    def _is_valid_api_key(self, provided_key: str | None) -> bool:
        """Validate API key."""
        if not provided_key or not self.expected_api_key:
            return False
        return secrets.compare_digest(provided_key, self.expected_api_key)

    def _handle_internal_error(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle internal errors."""
        self._send_response(http_dialogue, http_msg, 500, {"error": "Internal server error"}, "Internal Server Error")
        self.context.logger.error("Internal server error")

    def _handle_unauthorized(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle unauthorized requests."""
        self._send_response(http_dialogue, http_msg, 401, {"error": "Unauthorized"}, "Unauthorized")
        self.context.logger.warning("Unauthorized request received")

    def _send_response(
        self,
        dialogue: HttpDialogue,
        msg: HttpMessage,
        status_code: int,
        body: dict[str, Any],
        status_text: str = "Success",
    ) -> None:
        """Send HTTP response."""
        headers = self._build_response_headers(msg.headers)
        response = dialogue.reply(
            performative=HttpMessage.Performative.RESPONSE,
            target_message=msg,
            version=msg.version,
            status_code=status_code,
            status_text=status_text,
            headers=headers,
            body=json.dumps(body).encode("utf-8"),
        )
        self.context.logger.info(f"responding with: {response}")
        self.context.outbox.put_message(message=response)

    def _handle_status(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle the status endpoint."""
        status, current_state = self._get_system_status()
        metrics = self.context.strategy.get_metrics()

        status_data = {"status": status.value, "fsm_state": current_state, "version": str(PUBLIC_ID.version), **metrics}

        self._send_response(http_dialogue, http_msg, 200, status_data)

    def _handle_unidentified_dialogue(self, http_msg: HttpMessage) -> None:
        """Handle an unidentified dialogue."""
        self.context.logger.info(f"received invalid http message={http_msg}, unidentified dialogue.")
        default_dialogues = cast("DefaultDialogues", self.context.default_dialogues)
        default_msg, _ = default_dialogues.create(
            counterparty=http_msg.sender,
            performative=DefaultMessage.Performative.ERROR,
            error_code=DefaultMessage.ErrorCode.INVALID_DIALOGUE,
            error_msg="Invalid dialogue.",
            error_data={"http_message": http_msg.encode()},
        )
        self.context.outbox.put_message(message=default_msg)

    def _handle_invalid(self, http_msg: HttpMessage, http_dialogue: HttpDialogue) -> None:
        """Handle an invalid requests."""
        self._send_response(http_dialogue, http_msg, 400, {"error": "Invalid request"}, "Bad Request")

        self.context.logger.warning(f"Invalid request received for path: {http_msg.url}")

    def teardown(self) -> None:
        """Implement the handler teardown."""


class WebsocketHandler(Handler):
    """Base WebSocket handler with action-based routing and standardized responses."""

    SUPPORTED_PROTOCOL = WebsocketsMessage.protocol_id

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context.logger.info("WebsocketHandler initialized")

    @property
    def strategy(self) -> "APIClientStrategy":
        """Get the strategy."""
        return cast("APIClientStrategy", self.context.api_client_strategy)

    def setup(self) -> None:
        """Implement the setup."""
        self.context.logger.info("WebsocketHandler setup")

    def handle(self, message: Message) -> None:
        """Handle the message with action-based routing."""
        if message.performative == WebsocketsMessage.Performative.CONNECT:
            return self._handle_connect(message)

        dialogue = self.websocket_dialogues.get_dialogue(message)
        if message.performative == WebsocketsMessage.Performative.DISCONNECT:
            return self._handle_disconnect(message, dialogue)

        if dialogue is None:
            self.context.logger.error(f"Could not locate dialogue for message={message}")
            return None

        if message.performative == WebsocketsMessage.Performative.SEND:
            try:
                data = json.loads(message.data)
                action = data.get("action")
                payload = data.get("payload", {})
                handler = getattr(self, f"handle_{action}", None)
                if handler:
                    handler(message, payload, dialogue)
                else:
                    self.ws_error(message, "Unknown action", dialogue)
            except Exception as e:
                self.context.logger.exception(f"Error handling WebSocket message: {e}")
                self.ws_error(message, f"Internal error: {e!s}", dialogue)
            return None

        self.context.logger.warning(f"Cannot handle websockets message of performative={message.performative}")
        return None

    # Example action handler
    def handle_ping(self, message: Message, _payload: dict, dialogue) -> None:
        """Handle the ping action."""
        return self.ws_success(message, "pong", {"timestamp": datetime.now(UTC).isoformat()}, dialogue)

    def ws_success(self, _message: Message, event_type: str, payload: Any, dialogue) -> None:
        """Send a standardized success response."""
        ws_msg = dialogue.reply(
            performative=WebsocketsMessage.Performative.SEND,
            target_message=dialogue.last_message,
            data=json.dumps(
                {
                    "type": event_type,
                    "status": "success",
                    "payload": payload,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ),
        )
        self.context.outbox.put_message(message=ws_msg)

    def ws_error(self, _message: Message, error_msg: str, dialogue, details: dict | None = None) -> None:
        """Send a standardized error response."""
        ws_msg = dialogue.reply(
            performative=WebsocketsMessage.Performative.SEND,
            target_message=dialogue.last_message,
            data=json.dumps(
                {
                    "type": "error",
                    "status": "error",
                    "message": error_msg,
                    "details": details,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ),
        )
        self.context.outbox.put_message(message=ws_msg)

    def broadcast_event(self, event_type: str, payload: dict) -> None:
        """Broadcast an event to all connected clients."""
        event = {
            "type": event_type,
            "status": "success",
            "payload": payload,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if not self.strategy.clients:
            self.context.logger.warning("No connected clients for event publishing.")
            return
        for client_id, dialogue in self.strategy.clients.items():
            try:
                ws_msg = dialogue.reply(
                    performative=WebsocketsMessage.Performative.SEND,
                    target_message=dialogue.last_message,
                    data=json.dumps(event),
                )
                self.context.outbox.put_message(message=ws_msg)
            except Exception as e:
                self.context.logger.exception(f"Error sending event to client {client_id}: {e}")

    @property
    def websocket_dialogues(self) -> "ApiWebSocketDialogues":
        """Get the websocket dialogues."""
        return cast("ApiWebSocketDialogues", self.context.api_ws_dialogues)

    def _handle_connect(self, message: Message) -> None:
        dialogue: WebsocketsDialogue = self.websocket_dialogues.get_dialogue(message)
        if dialogue:
            self.context.logger.debug(f"Already have a dialogue for message={message}")
            return
        client_reference = message.url
        dialogue = self.websocket_dialogues.update(message)
        response_msg = dialogue.reply(
            performative=WebsocketsMessage.Performative.CONNECTION_ACK,
            success=True,
            target_message=message,
        )
        if not hasattr(self.strategy, "clients"):
            self.strategy.clients = {}
        self.strategy.clients[client_reference] = dialogue
        self.context.logger.info(f"New WebSocket client connected. Total clients: {len(self.strategy.clients)}")
        self.context.outbox.put_message(message=response_msg)

    def _handle_disconnect(self, message: Message, dialogue: WebsocketsDialogue) -> None:
        self.context.logger.info(f"Handling disconnect message in skill: {message}")
        ws_dialogues_to_connections = {v.incomplete_dialogue_label: k for k, v in self.strategy.clients.items()}
        if dialogue.incomplete_dialogue_label in ws_dialogues_to_connections:
            client_id = ws_dialogues_to_connections[dialogue.incomplete_dialogue_label]
            del self.strategy.clients[client_id]
            self.context.logger.info(f"WebSocket client disconnected. Remaining clients: {len(self.strategy.clients)}")
        else:
            self.context.logger.warning(f"Could not find dialogue to disconnect: {dialogue.incomplete_dialogue_label}")

    def get_connected_clients_count(self) -> int:
        """Get the number of connected clients."""
        return len(self.strategy.clients)

    def teardown(self) -> None:
        """Implement the handler teardown."""
        self.context.logger.info("WebsocketHandler teardown")
