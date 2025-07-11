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

import re
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import markdown

from packages.xiuxiuxar.skills.dyor_app.models import LLMServiceError
from packages.xiuxiuxar.skills.dyor_app.prompt import build_report_prompt
from packages.xiuxiuxar.skills.dyor_app.behaviours.base import BaseState, DyorabciappEvents, DyorabciappStates


class GenerateReportRound(BaseState):
    """This class implements the behaviour of the state GenerateReportRound."""

    REQUIRED_SECTIONS = [
        "Overview",
        "Key Recent Changes",
        "Recent News/Events",
        "Community & Social Chatter",
        "Unlock Events",
        "Analysis",
        "Conclusion",
    ]

    MODEL_CONFIG = {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = DyorabciappStates.GENERATEREPORTROUND
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._prompt = None
        self._payload = None
        self._attempt = 0
        self._max_retries = 3
        self._last_error = None

    def _fetch_structured_payload(self):
        """Fetch the structured payload from the database."""
        trigger_id = self.context.trigger_context.get("trigger_id")
        asset_id = self.context.trigger_context.get("asset_id")

        payload = self.context.db_model.get_structured_payload(trigger_id, asset_id)
        if not payload:
            msg = "No structured payload found for this trigger/asset."
            raise ValueError(msg)
        return payload

    def _validate_markdown(self, text: str) -> bool:
        """Check if the text is valid Markdown (basic check: can be parsed)."""
        try:
            html = markdown.markdown(text)
            return bool(html.strip())
        except (ValueError, TypeError):
            return False

    def _check_required_sections(self, text: str) -> list[str]:
        """Return a list of missing required section headers."""
        missing = []
        for section in self.REQUIRED_SECTIONS:
            # Look for a Markdown header (### Section)
            if not re.search(rf"^###\s*{re.escape(section)}", text, re.MULTILINE | re.IGNORECASE):
                missing.append(section)
        return missing

    def _setup_run(self) -> bool:
        """Set up a new report generation run. Returns False if report already exists."""
        self._future = None
        self._prompt = None
        self._payload = None
        self._attempt = 1
        self._last_error = None
        self._is_done = False  # Reset done state for new trigger

        trigger_id = self.context.trigger_context.get("trigger_id")
        if self.context.db_model.report_exists(trigger_id):
            self.context.logger.warning(f"Report already exists for trigger_id={trigger_id}")
            return False

        self._payload = self._fetch_structured_payload()
        self._prompt = build_report_prompt(self._payload)
        self.context.logger.info("Built report prompt.")
        self.context.logger.debug(f"Prompt: {self._prompt}")
        return True

    def _submit_llm_request(self) -> None:
        """Submit a request to the LLM service."""
        self.context.logger.info(f"Submitting LLM request (attempt {self._attempt}/{self._max_retries})")
        self._future = self._executor.submit(self.context.llm_service.generate_summary, self._prompt, self.MODEL_CONFIG)

    def _process_llm_result_and_store(self, llm_result: dict) -> None:
        """Process, validate, and store the LLM result."""
        llm_output = llm_result["content"]
        self.context.logger.info(f"LLM output (attempt {self._attempt}): {llm_output[:500]}...")

        if not self._validate_markdown(llm_output):
            msg = f"LLM output is not valid Markdown (attempt {self._attempt})."
            raise ValueError(msg)

        missing_sections = self._check_required_sections(llm_output)
        if missing_sections:
            msg = f"LLM output missing required sections: {', '.join(missing_sections)} (attempt {self._attempt})"
            raise ValueError(msg)

        report_id = self.context.db_model.store_report(
            trigger_id=self.context.trigger_context.get("trigger_id"),
            asset_id=self.context.trigger_context.get("asset_id"),
            report_content_markdown=llm_output,
            report_data_json=self._payload,
            llm_model_used=llm_result["llm_model_used"],
            generation_time_ms=llm_result["generation_time_ms"],
            token_usage=llm_result["token_usage"],
        )

        if not hasattr(self.context, "report_context"):
            self.context.report_context = {}
        self.context.report_context["report_id"] = report_id
        self.context.strategy.record_report_generated()
        self.context.logger.info(f"Stored report with ID {report_id}.")

    def _create_error_context(self, error_type: str, error_message: str) -> dict:
        """Create a standardized error context dictionary."""
        return {
            "error_type": error_type,
            "error_message": error_message,
            "trigger_id": self.context.trigger_context.get("trigger_id"),
            "asset_id": self.context.trigger_context.get("asset_id"),
            "originating_round": str(self._state),
        }

    def _handle_max_retries_exceeded(self) -> None:
        """Log and set context when max retries are exceeded."""
        error_message = f"Report generation failed after {self._max_retries} attempts: {self._last_error}"
        critical_info = self._create_error_context("report_generation_error", error_message)
        critical_info["level"] = "CRITICAL"
        self.context.logger.critical(error_message, extra=critical_info)
        self.context.error_context = critical_info

    def _finish_run(self, success: bool = True) -> None:
        """Finalize the run and set the event."""
        self._event = DyorabciappEvents.DONE if success else DyorabciappEvents.ERROR
        self._is_done = True
        self._future = None
        self._prompt = None
        self._payload = None
        self._attempt = 0
        self._last_error = None

    def act(self) -> None:
        """Generate a report using a non-blocking, asynchronous pattern."""
        if self._attempt == 0:  # First call for this trigger
            self.context.logger.info(f"Entering state: {self._state}")
            if not self._setup_run():
                self._finish_run(success=True)  # Report exists, consider it done
                return
            self._submit_llm_request()
            return  # Defer processing to the next tick

        if self._future is None:
            self.context.logger.error("GenerateReportRound: _future is None unexpectedly.")
            self.context.error_context = self._create_error_context("internal_error", "Future was not set.")
            self._finish_run(success=False)
            return

        if not self._future.done():
            self.context.logger.debug("Waiting for LLM response...")
            return

        # Future is done, process it
        try:
            llm_result = self._future.result()
            self._process_llm_result_and_store(llm_result)
            self._finish_run(success=True)
        except (LLMServiceError, ValueError) as e:
            self._last_error = e
            self.context.logger.warning(
                f"LLM report generation failed (attempt {self._attempt}/{self._max_retries}): {e}"
            )
            self._attempt += 1
            if self._attempt > self._max_retries:
                self._handle_max_retries_exceeded()
                self._finish_run(success=True)  # Finished with critical error, but "done" per original logic
            else:
                self._submit_llm_request()  # Retry
        except Exception as e:
            self.context.logger.exception(f"Unexpected error in GenerateReportRound: {e}")
            self.context.error_context = self._create_error_context("report_generation_error", str(e))
            self._finish_run(success=False)
