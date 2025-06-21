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

from typing import Any

import sqlalchemy
from sqlalchemy import text

from packages.xiuxiuxar.skills.dyor_app.behaviours.base import BaseState, DyorabciappEvents, DyorabciappStates


class WatchingRound(BaseState):
    """This class implements the behaviour of the state WatchingRound."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.WATCHINGROUND

    def act(self) -> None:
        """Act:
        1. Check if there are any triggers in the database.
        2. If there are, set the event to TRIGGER.
        3. If there are no triggers, set the event to NO_TRIGGER.
        """
        self.context.logger.debug(f"Entering state: {self._state}")

        # Clear any stale error context from previous trigger processing cycles
        if hasattr(self.context, "error_context"):
            self.context.error_context = None

        try:
            # Query for unprocessed triggers
            with self.context.db_model.engine.begin() as conn:
                row = conn.execute(
                    text("""
                    WITH next AS (
                        SELECT trigger_id
                        FROM   triggers
                        WHERE  status = 'pending'
                        ORDER  BY created_at
                        LIMIT  1
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE triggers AS t
                    SET    status = 'processing',
                        processing_started_at = NOW()
                    FROM   next
                    WHERE  t.trigger_id = next.trigger_id
                    RETURNING t.trigger_id,
                            t.asset_id,
                            (SELECT symbol FROM assets WHERE asset_id = t.asset_id)   AS symbol,
                            (SELECT name   FROM assets WHERE asset_id = t.asset_id)   AS name,
                            t.trigger_details
                """)
                ).fetchone()

                if row:
                    # Found a trigger, set up context and transition
                    self.context.trigger_context = {
                        "trigger_id": row.trigger_id,
                        "asset_id": row.asset_id,
                        "asset_symbol": row.symbol,
                        "asset_name": row.name,
                        "trigger_details": row.trigger_details,
                    }
                    self.context.logger.info(
                        f"Found trigger {row.trigger_id} for asset {row.symbol} - set to processing"
                    )
                    self._event = DyorabciappEvents.TRIGGER
                else:
                    self.context.logger.debug("No pending triggers found")
                    self._event = DyorabciappEvents.NO_TRIGGER

        except sqlalchemy.exc.SQLAlchemyError as e:
            self.context.logger.exception(f"Database error while checking triggers: {e}")
            self._event = DyorabciappEvents.ERROR

        self._is_done = True
