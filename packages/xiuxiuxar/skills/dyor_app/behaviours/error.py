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

import json
import time
import random
from typing import Any
from datetime import UTC, datetime

import requests
import sqlalchemy
from sqlalchemy import text

from packages.xiuxiuxar.skills.dyor_app.behaviours.base import BaseState, DyorabciappEvents, DyorabciappStates


class HandleErrorRound(BaseState):
    """This class implements the behaviour of the state HandleErrorRound."""

    # Error classification rules
    RETRYABLE_ERRORS = {
        "database_error": True,
        "storage_error": True,
        "api_error": True,
        "timeout_error": True,
        "llm_api_error": True,
        "llm_rate_limit": True,
        "llm_generation_error": True,  # Sometimes retryable
        "scraping_error": True,
    }
    NON_RETRYABLE_ERRORS = {
        "configuration_error": False,
        "validation_error": False,
        "data_validation_error": False,
        "asset_validation_error": False,
        "internal_logic_error": False,
        "resource_exhaustion": False,
        "llm_content_filter": False,
    }
    # Default retry/backoff config
    BASE_DELAY = 5  # seconds
    MAX_DELAY = 300  # seconds
    MAX_ATTEMPTS = 5
    JITTER = 0.2  # 20% jitter

    def __init__(self, **kwargs: Any) -> None:
        self.ntfy_topic = kwargs.pop("ntfy_topic", "alerts")
        super().__init__(**kwargs)
        self._state = DyorabciappStates.HANDLEERRORROUND
        self._retry_states = {}  # Store retry states in the class instance

    def _classify_error(self, error_context: dict) -> tuple[bool, str]:
        """Classify error and determine if retryable."""
        error_type = (error_context.get("error_type") or "").lower()
        if error_type in self.RETRYABLE_ERRORS:
            return True, error_type
        if error_type in self.NON_RETRYABLE_ERRORS:
            return False, error_type
        # Fallback: retry on API/DB/timeout, not on validation/config
        if "api" in error_type or "timeout" in error_type or "storage" in error_type or "database" in error_type:
            return True, error_type
        if "validation" in error_type or "config" in error_type or "resource" in error_type:
            return False, error_type
        return False, error_type or "unknown"

    def _get_retry_state(self, trigger_id: int) -> dict:
        """Get or initialize retry state for this trigger."""
        return self._retry_states.setdefault(trigger_id, {"attempt": 0, "last_ts": None})

    def _increment_error_metrics(self, error_type: str) -> None:
        """Increment error metrics in strategy if available."""
        if hasattr(self.context, "strategy") and hasattr(self.context.strategy, "_metrics"):
            metrics = self.context.strategy._metrics  # noqa: SLF001
            key = f"errors_{error_type}"
            metrics[key] = metrics.get(key, 0) + 1

    def _log_json_error(self, error_context: dict, retryable: bool, attempt: int, max_attempts: int) -> None:
        log_record = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "level": "ERROR" if retryable else "CRITICAL",
            "round": error_context.get("originating_round", "unknown"),
            "asset_id": error_context.get("asset_id"),
            "trigger_id": error_context.get("trigger_id"),
            "error_type": error_context.get("error_type"),
            "message": error_context.get("error_message"),
            "stack": error_context.get("stack_trace"),
            "retryable": retryable,
            "attempt": attempt,
            "max_attempts": max_attempts,
        }
        self.context.logger.error(json.dumps(log_record, default=str))

    def _send_alert(self, error_context: dict, critical: bool = False) -> None:
        # Integrate with ntfy.sh alerting system, else log CRITICAL
        alert_msg = (
            f"ALERT: {error_context.get('error_type')} | {error_context.get('error_message')} | "
            f"Trigger: {error_context.get('trigger_id')} | Asset: {error_context.get('asset_id')}"
        )
        try:
            requests.post(
                f"https://ntfy.sh/{self.ntfy_topic}",
                data=alert_msg.encode("utf-8"),
                headers={
                    "Title": f"{error_context.get('error_type', 'Error').replace('_', ' ').title()} detected",
                    "Priority": "urgent" if critical else "high",
                    "Tags": "warning,skull" if critical else "warning",
                },
                timeout=5,
            )
        except (requests.RequestException, requests.Timeout, requests.ConnectionError) as e:
            self.context.logger.critical(f"Failed to send alert to ntfy.sh: {e!s} | {alert_msg}")
            self.context.logger.critical(alert_msg)

    def _update_trigger_status_error(self, trigger_id: int, error_message: str | None = None) -> None:
        # Mark the trigger as errored in the DB
        try:
            with self.context.db_model.engine.connect() as conn:
                conn.execute(
                    text("""
                        UPDATE triggers
                        SET status = 'error', completed_at = NOW(), error_message = :error_message
                        WHERE trigger_id = :trigger_id
                    """),
                    {"trigger_id": trigger_id, "error_message": error_message},
                )
                conn.commit()
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.context.logger.critical(f"Failed to update trigger status to error for trigger_id={trigger_id}: {e!s}")

    def _calculate_backoff(self, attempt: int) -> float:
        base = self.BASE_DELAY * (2**attempt)
        jitter = base * self.JITTER * (random.random() * 2 - 1)  # +/- jitter  # noqa: S311
        return min(self.MAX_DELAY, max(1, base + jitter))

    def act(self) -> None:
        """Handle error: log, increment metrics, retry or mark as failed, alert if critical."""
        self.context.logger.info(f"Entering state: {self._state}")
        error_context = getattr(self.context, "error_context", {}) or {}
        trigger_id = error_context.get("trigger_id")

        originating_round = error_context.get("originating_round", "unknown")

        retryable, error_type = self._classify_error(error_context)
        retry_state = self._get_retry_state(trigger_id) if trigger_id is not None else {"attempt": 0}
        attempt = retry_state["attempt"]

        self._log_json_error(error_context, retryable, attempt, self.MAX_ATTEMPTS)

        self._increment_error_metrics(error_type)

        if not retryable or attempt >= self.MAX_ATTEMPTS:
            self._send_alert(error_context, critical=True)

        if retryable and attempt < self.MAX_ATTEMPTS:
            delay = self._calculate_backoff(attempt)
            retry_state["attempt"] += 1
            retry_state["last_ts"] = datetime.now(tz=UTC).isoformat()
            self.context.logger.info(
                f"Retrying {originating_round} for trigger_id={trigger_id} in {delay:.1f}s"
                f"(attempt {attempt + 1}/{self.MAX_ATTEMPTS})"
            )
            time.sleep(delay)
            self._event = DyorabciappEvents.RETRY
        else:
            if trigger_id is not None:
                error_message = error_context.get("error_message", "Unknown error")
                self._update_trigger_status_error(trigger_id, error_message)
            self._event = DyorabciappEvents.DONE

        # Clear error context to prevent state leakage to subsequent triggers
        self.context.error_context = None
        self._is_done = True
