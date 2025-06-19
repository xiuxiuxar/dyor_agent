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

import os
from typing import Any

from aea.skills.behaviours import FSMBehaviour

from packages.xiuxiuxar.skills.simple_fsm.behaviours.base import DyorabciappEvents, DyorabciappStates
from packages.xiuxiuxar.skills.simple_fsm.behaviours.error import HandleErrorRound
from packages.xiuxiuxar.skills.simple_fsm.behaviours.setup import SetupDYORRound
from packages.xiuxiuxar.skills.simple_fsm.behaviours.ingest import IngestDataRound
from packages.xiuxiuxar.skills.simple_fsm.behaviours.deliver import DeliverReportRound
from packages.xiuxiuxar.skills.simple_fsm.behaviours.process import ProcessDataRound
from packages.xiuxiuxar.skills.simple_fsm.behaviours.trigger import TriggerRound
from packages.xiuxiuxar.skills.simple_fsm.behaviours.generate import GenerateReportRound
from packages.xiuxiuxar.skills.simple_fsm.behaviours.watching import WatchingRound


class DyorabciappFsmBehaviour(FSMBehaviour):
    """This class implements a simple Finite State Machine behaviour."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.register_state(DyorabciappStates.SETUPDYORROUND.value, SetupDYORRound(**kwargs), True)

        self.register_state(DyorabciappStates.WATCHINGROUND.value, WatchingRound(**kwargs))
        self.register_state(DyorabciappStates.PROCESSDATAROUND.value, ProcessDataRound(**kwargs))
        self.register_state(DyorabciappStates.DELIVERREPORTROUND.value, DeliverReportRound(**kwargs))
        self.register_state(DyorabciappStates.TRIGGERROUND.value, TriggerRound(**kwargs))
        self.register_state(DyorabciappStates.INGESTDATAROUND.value, IngestDataRound(**kwargs))
        self.register_state(DyorabciappStates.GENERATEREPORTROUND.value, GenerateReportRound(**kwargs))
        self.register_state(DyorabciappStates.HANDLEERRORROUND.value, HandleErrorRound(**kwargs))

        self.register_transition(
            source=DyorabciappStates.DELIVERREPORTROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.DELIVERREPORTROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.DELIVERREPORTROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.DELIVERREPORTROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.GENERATEREPORTROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.DELIVERREPORTROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.GENERATEREPORTROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.GENERATEREPORTROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.GENERATEREPORTROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.HANDLEERRORROUND.value,
            event=DyorabciappEvents.RETRY,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.HANDLEERRORROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.INGESTDATAROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.PROCESSDATAROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.INGESTDATAROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.INGESTDATAROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.INGESTDATAROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.PROCESSDATAROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.GENERATEREPORTROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.PROCESSDATAROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.PROCESSDATAROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.PROCESSDATAROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.SETUPDYORROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.SETUPDYORROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.SETUPDYORROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.SETUPDYORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.TRIGGERROUND.value,
            event=DyorabciappEvents.DONE,
            destination=DyorabciappStates.INGESTDATAROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.TRIGGERROUND.value,
            event=DyorabciappEvents.ERROR,
            destination=DyorabciappStates.HANDLEERRORROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.TRIGGERROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.TRIGGERROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.WATCHINGROUND.value,
            event=DyorabciappEvents.NO_TRIGGER,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.WATCHINGROUND.value,
            event=DyorabciappEvents.TIMEOUT,
            destination=DyorabciappStates.WATCHINGROUND.value,
        )
        self.register_transition(
            source=DyorabciappStates.WATCHINGROUND.value,
            event=DyorabciappEvents.TRIGGER,
            destination=DyorabciappStates.TRIGGERROUND.value,
        )

    def setup(self) -> None:
        """Implement the setup."""
        self.context.logger.info("Setting up Dyorabciapp FSM behaviour.")

    def teardown(self) -> None:
        """Implement the teardown."""
        self.context.logger.info("Tearing down Dyorabciapp FSM behaviour.")

    def act(self) -> None:
        """Implement the act."""
        super().act()
        if self.current is None:
            self.context.logger.info("No state to act on.")
            self.terminate()

    def terminate(self) -> None:
        """Implement the termination."""
        os._exit(0)
