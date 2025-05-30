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
"""This module contains the prompt for the simple_fsm skill."""

from typing import Any
from pathlib import Path
from datetime import UTC, datetime

from jinja2 import Environment, FileSystemLoader, select_autoescape


# Path to the template file
TEMPLATE_FILENAME = "prompt_template.jinja"
TEMPLATE_DIR = Path(__file__).parent


def timestamp_to_date(timestamp):
    """Convert Unix timestamp to formatted date string."""
    try:
        return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(timestamp)


def intcomma(value):
    """Format number with commas as thousands separators."""
    try:
        return f"{float(value):,.0f}"
    except (ValueError, TypeError):
        return str(value)


env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["jinja"]),
    trim_blocks=True,
    lstrip_blocks=True,
)
env.filters["timestamp_to_date"] = timestamp_to_date
env.filters["intcomma"] = intcomma


def build_report_prompt(context: dict[str, Any]) -> str:
    """Render the report prompt using the Jinja2 template and provided context."""
    template = env.get_template(TEMPLATE_FILENAME)
    return template.render(**context)
