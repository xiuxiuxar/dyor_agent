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

"""This package contains a scaffold of a model."""

from typing import Any

from aea.skills.base import Model

class DYORParams(Model):
    """This class scaffolds a model."""

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model."""
        self.POSTGRES_USER = kwargs.pop("POSTGRES_USER", "default_user")
        self.POSTGRES_PASSWORD = kwargs.pop("POSTGRES_PASSWORD", "default_password")
        self.POSTGRES_HOST = kwargs.pop("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = kwargs.pop("POSTGRES_PORT", 5432)
        self.POSTGRES_DB = kwargs.pop("POSTGRES_DB", "default_db")

        super().__init__(**kwargs)
