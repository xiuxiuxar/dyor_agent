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

"""This module contains the handler for the 'metrics' skill."""

import json
import secrets
from enum import Enum
from typing import Any, cast
from urllib.parse import urlparse

from aea.skills.base import Handler
from aea.protocols.base import Message

from packages.xiuxiuxar.skills.simple_fsm import PUBLIC_ID
from packages.eightballer.protocols.default import DefaultMessage
from packages.eightballer.protocols.http.message import HttpMessage
from packages.xiuxiuxar.skills.simple_fsm.dialogues import (
    HttpDialogue,
    HttpDialogues,
    DefaultDialogues,
)


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
    SUPPORTED_ENDPOINTS = {"/status", "/metrics"}

    def __init__(self, **kwargs):
        """Initialise the handler."""
        self.enable_cors = kwargs.pop("enable_cors", False)
        self.expected_api_key = kwargs.pop("expected_api_key", None)
        super().__init__(**kwargs)

    def setup(self) -> None:
        """Implement the setup."""

    def handle(self, message: Message) -> None:
        """Handle the message."""
        self.context.logger.debug("Handling new connection message in skill")
        http_msg = cast(HttpMessage, message)
        http_dialogues = cast(HttpDialogues, self.context.http_dialogues)
        http_dialogue = cast(HttpDialogue, http_dialogues.update(http_msg))

        if http_dialogue is None:
            self._handle_unidentified_dialogue(http_msg)
            return

        if http_msg.performative != HttpMessage.Performative.REQUEST:
            self._handle_invalid(http_msg, http_dialogue)
            return

        headers = _parse_headers(http_msg.headers)
        provided_key = headers.get("x-api-key")
        if not self._is_valid_api_key(provided_key):
            self._handle_unauthorized(http_msg, http_dialogue)
            return

        method = http_msg.method.lower()
        url_path = urlparse(http_msg.url).path.rstrip("/")
        if method not in self.SUPPORTED_METHODS or url_path not in self.SUPPORTED_ENDPOINTS:
            self._handle_invalid(http_msg, http_dialogue)
            return

        handlers = {
            "/status": self._handle_status,
            # "/metrics": self._handle_metrics,
        }

        try:
            handlers[url_path](http_msg, http_dialogue)
        except Exception as e:
            self.context.logger.exception(f"Error handling request: {e!s}")
            self._handle_internal_error(http_msg, http_dialogue)

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
        self.context.logger.exception("Internal server error")

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
        default_dialogues = cast(DefaultDialogues, self.context.default_dialogues)
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
