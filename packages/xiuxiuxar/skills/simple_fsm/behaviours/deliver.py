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


class DeliverReportRound(BaseState):
    """This class implements the behaviour of the state DeliverReportRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.DELIVERREPORTROUND

    def _update_trigger_status(self, conn) -> bool:
        """Update trigger status to processed."""
        try:
            conn.execute(
                text("""
                    UPDATE triggers
                    SET status = 'processed', completed_at = NOW()
                    WHERE trigger_id = :trigger_id
                """),
                {"trigger_id": self.context.trigger_context["trigger_id"]},
            )
            conn.commit()
            return True
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.context.logger.exception(f"Failed to update trigger status: {e}")
            return False

    def _publish_report_event(self) -> bool:
        """Publish report_generated event via WebSocket."""
        try:
            for handler in self.context.api_client_strategy.ws_handlers:
                if hasattr(handler, "broadcast_event"):
                    handler.broadcast_event(
                        "report_generated",
                        {
                            "trigger_id": self.context.trigger_context["trigger_id"],
                            "asset_id": self.context.trigger_context["asset_id"],
                            "asset_symbol": self.context.trigger_context["asset_symbol"],
                            "report_id": self.context.report_context["report_id"],
                            "status": "success",
                        },
                    )
            return True
        except (ConnectionError, TimeoutError, ValueError) as e:
            self.context.logger.warning(f"Error publishing report event: {e}")
            return False

    def act(self) -> None:
        """Update trigger status and notify about report completion."""
        self.context.logger.info(f"Entering state: {self._state}")

        try:
            # Update trigger status
            with self.context.db_model.engine.connect() as conn:
                if not self._update_trigger_status(conn):
                    msg = "Failed to update trigger status"
                    raise RuntimeError(msg)

            self._publish_report_event()

            self.context.logger.info("Report published successfully")

            self._event = DyorabciappEvents.DONE

        except Exception as e:
            self.context.logger.exception(f"Error in DeliverReportRound: {e}")
            self.context.error_context = {
                "error_type": "delivery_error",
                "error_message": str(e),
                "trigger_id": self.context.trigger_context.get("trigger_id"),
                "asset_id": self.context.trigger_context.get("asset_id"),
                "critical": True,  # DB failure is critical
            }
            self._event = DyorabciappEvents.ERROR

        self._is_done = True