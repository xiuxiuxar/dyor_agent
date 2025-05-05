from typing import Any
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


# Path to the template file
TEMPLATE_FILENAME = "prompt_template_v2.jinja"
TEMPLATE_DIR = Path(__file__).parent

# Set up Jinja2 environment
env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["jinja"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_report_prompt(context: dict[str, Any]) -> str:
    """Render the report prompt using the Jinja2 template and provided context."""
    template = env.get_template(TEMPLATE_FILENAME)
    return template.render(**context)
