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

import importlib
from abc import ABC
from enum import Enum
from typing import Any

from aea.skills.behaviours import State


PROTOCOL_HTTP = "eightballer/http:0.1.0"
PROTOCOL_WEBSOCKETS = "eightballer/websockets:0.1.0"
PROTOCOL_HANDLER_MAP = {
    PROTOCOL_HTTP: "http_handlers",
    PROTOCOL_WEBSOCKETS: "ws_handlers",
}


def dynamic_import(component_name, module_name):
    """Dynamically import a module."""

    module = importlib.import_module(component_name)
    return getattr(module, module_name)


class DyorabciappEvents(Enum):
    """Events for the fsm."""

    NO_TRIGGER = "NO_TRIGGER"
    DONE = "DONE"
    RETRY = "RETRY"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    TRIGGER = "TRIGGER"


class DyorabciappStates(Enum):
    """States for the fsm."""

    WATCHINGROUND = "watchinground"
    PROCESSDATAROUND = "processdataround"
    SETUPDYORROUND = "setupdyorround"
    DELIVERREPORTROUND = "deliverreportround"
    TRIGGERROUND = "triggerround"
    INGESTDATAROUND = "ingestdataround"
    GENERATEREPORTROUND = "generatereportround"
    HANDLEERRORROUND = "handleerrorround"


class BaseState(State, ABC):
    """Base class for states."""

    _state: DyorabciappStates = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._event = None
        self._is_done = False

    def act(self) -> None:
        """Perform the act."""
        self._is_done = True
        self._event = DyorabciappEvents.DONE

    def is_done(self) -> bool:
        """Is done."""
        return self._is_done

    @property
    def event(self) -> str | None:
        """Current event."""
        return self._event
