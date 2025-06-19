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

from typing import Any

import sqlalchemy
from sqlalchemy import text

from packages.xiuxiuxar.skills.simple_fsm.behaviours.base import BaseState, DyorabciappEvents, DyorabciappStates


class TriggerRound(BaseState):
    """This class implements the behaviour of the state TriggerRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.TRIGGERROUND

    def _validate_asset(self, conn) -> tuple[bool, str]:
        """Validate asset exists and is active."""
        try:
            result = conn.execute(
                text("""
                    SELECT a.asset_id, a.symbol, a.name
                    FROM assets a
                    WHERE a.asset_id = :asset_id
                """),
                {"asset_id": self.context.trigger_context["asset_id"]},
            )
            asset = result.fetchone()

            if not asset:
                return False, "Asset not found"

            return True, ""
        except sqlalchemy.exc.SQLAlchemyError as e:
            return False, f"Database error: {e!s}"

    def _check_recent_report(self, conn, asset_id: int) -> tuple[bool, str]:
        """Check if a recent report exists (within last 24h unless force_refresh)."""
        trigger_details = self.context.trigger_context.get("trigger_details", {})
        if trigger_details.get("force_refresh"):
            return True, ""

        try:
            result = conn.execute(
                text("""
                    SELECT r.report_id, r.created_at
                    FROM reports r
                    WHERE r.asset_id = :asset_id
                    AND r.created_at > NOW() - INTERVAL '24 hours'
                    ORDER BY r.created_at DESC
                    LIMIT 1
                """),
                {"asset_id": asset_id},
            )
            report = result.fetchone()

            if report:
                return False, f"Recent report exists from {report[1]}"
            return True, ""
        except sqlalchemy.exc.SQLAlchemyError as e:
            return False, f"Database error checking recent report: {e!s}"

    def act(self) -> None:
        """Process trigger request and prepare for data ingestion."""
        self.context.logger.info(f"Entering state: {self._state}")

        try:
            with self.context.db_model.engine.connect() as conn:
                # Validate asset and check recent report
                is_valid, error_msg = self._validate_asset(conn)
                if not is_valid:
                    raise ValueError(error_msg)

                # Update trigger context with validated asset info
                self.context.trigger_context.update(
                    {
                        "asset_id": self.context.trigger_context["asset_id"],
                        "asset_symbol": self.context.trigger_context["asset_symbol"],
                        "asset_name": self.context.trigger_context["asset_name"],
                    }
                )

                can_proceed, error_msg = self._check_recent_report(conn, self.context.trigger_context["asset_id"])
                if not can_proceed:
                    raise ValueError(error_msg)

                # Update metrics
                self.context.strategy.increment_active_triggers()
                self._event = DyorabciappEvents.DONE

        except ValueError as e:
            self.context.logger.warning(f"Validation error: {e!s}")
            self.context.error_context = {
                "error_type": "validation_error",
                "error_message": str(e),
                "trigger_id": self.context.trigger_context.get("trigger_id"),
                "asset_id": self.context.trigger_context.get("asset_id"),
            }
            self._event = DyorabciappEvents.ERROR

        except Exception as e:
            self.context.logger.exception(f"Unexpected error in TriggerRound: {e}")
            self.context.error_context = {
                "error_type": "unexpected_error",
                "error_message": str(e),
                "trigger_id": self.context.trigger_context.get("trigger_id"),
                "asset_id": self.context.trigger_context.get("asset_id"),
            }
            self._event = DyorabciappEvents.ERROR

        self._is_done = True
