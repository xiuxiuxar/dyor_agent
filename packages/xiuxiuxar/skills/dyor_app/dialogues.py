# ------------------------------------------------------------------------------
#
#   Copyright 2022 Valory AG
#   Copyright 2018-2021 Fetch.AI Limited
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

"""This module contains the classes required for dialogue management.

- DefaultDialogue: The dialogue class maintains state of a dialogue of type default and manages it.
- DefaultDialogues: The dialogues class keeps track of all dialogues of type default.
- HttpDialogue: The dialogue class maintains state of a dialogue of type http and manages it.
- HttpDialogues: The dialogues class keeps track of all dialogues of type http.
"""

from typing import Any

from aea.skills.base import Model
from aea.protocols.base import Address, Message
from aea.protocols.dialogue.base import Dialogue as BaseDialogue

from packages.eightballer.protocols.http.dialogues import (
    HttpDialogue as BaseHttpDialogue,
    HttpDialogues as BaseHttpDialogues,
)
from packages.eightballer.protocols.default.dialogues import (
    DefaultDialogue as BaseDefaultDialogue,
    DefaultDialogues as BaseDefaultDialogues,
)
from packages.eightballer.protocols.websockets.dialogues import (
    WebsocketsDialogue as BaseWebsocketsDialogue,
    WebsocketsDialogues as BaseWebsocketsDialogues,
)


class ApiWebSocketDialogues(Model, BaseWebsocketsDialogues):
    """Dialogues class for the ui_loader_abci skill."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize dialogues.

        Args:
        ----
            **kwargs: Keyword arguments

        """
        Model.__init__(self, **kwargs)

        def role_from_first_message(  # pylint: disable=unused-argument
            message: Message, receiver_address: Address
        ) -> BaseDialogue.Role:
            """Infer the role of the agent from an incoming/outgoing first message.

            Args:
            ----
                message (Message): an incoming/outgoing first message
                receiver_address (Address): the address of the receiving agent

            Returns:
            -------
                BaseDialogue.Role: The role of the agent

            """
            del message, receiver_address
            return BaseWebsocketsDialogue.Role.SERVER

        BaseWebsocketsDialogues.__init__(
            self,
            self_address=str(self.skill_id),
            role_from_first_message=role_from_first_message,
        )


DefaultDialogue = BaseDefaultDialogue
DefaultDialogues = BaseDefaultDialogues


HttpDialogue = BaseHttpDialogue
HttpDialogues = BaseHttpDialogues

WebsocketsDialogue = BaseWebsocketsDialogue
WebsocketsDialogues = BaseWebsocketsDialogues
