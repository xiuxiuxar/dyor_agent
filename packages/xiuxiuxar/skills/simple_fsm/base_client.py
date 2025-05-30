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

"""Base Client implementation."""

import json
import time
import logging
from typing import Any
from urllib.parse import urlparse

import requests
from curl_adapter import CurlCffiAdapter


logger = logging.getLogger(__name__)


class BaseAPIError(Exception):
    """Base exception for API related errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class BaseClientConfig:
    """Base configuration for clients."""

    api_key: str | None = None
    base_url: str
    timeout: int = 15
    retry_config: dict = {
        "max_retries": 3,
        "backoff_factor": 0.5,
        "status_forcelist": (429, 500, 502, 503, 504),
        "connect": 5,
        "read": 5,
    }
    default_headers: dict = {"Content-Type": "application/json"}


class BaseClient:
    """Base class for clients."""

    def __init__(
        self,
        base_url: str,
        timeout: int,
        max_retries: int,
        backoff_factor: float,
        status_forcelist: tuple[int, ...],
        headers: dict[str, str],
        api_key: str | None = None,
        error_class: type = BaseAPIError,
    ):
        parsed = urlparse(base_url) if base_url else None
        if not base_url or parsed.scheme not in {"http", "https"}:
            msg = f"Invalid {self.__class__.__name__} base URL {base_url}"
            raise ValueError(msg)

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.error_class = error_class
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.status_forcelist = status_forcelist

        self.session = requests.Session()
        if api_key:
            headers["Api-key"] = api_key
        self.session.headers.update(headers)

        def log_req(r: requests.Response):
            self.context.logger.debug(f"[HOOK] Sent headers: {r.request.headers}")
            return r

        self.session.hooks["response"].append(log_req)

        adapter = CurlCffiAdapter(impersonate_browser_type="chrome")

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._last_health_status: bool = True
        logger.info(f"{self.__class__.__name__} initialized")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        base_url_override: str | None = None,
        expect_json: bool = True,
    ) -> dict[str, Any] | str | None:
        """Make a request to the API."""
        url = endpoint if base_url_override == "" else f"{base_url_override or self.base_url}/{endpoint.lstrip('/')}"
        log_params = params or (json_data or {})
        logger.debug(f"Sending {method} request to {url} with params: {log_params}")

        for attempt in range(self.max_retries):
            try:
                if headers:
                    self.session.headers.update(headers)

                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    timeout=self.timeout,
                )

                logger.info(f"Response headers: {response.headers}")
                logger.info(f"Response body (first 200 chars): {response.text[:200]}")

                if self._should_retry(response, url, attempt):
                    continue

                self._update_health(True)
                return self._parse_response(response, endpoint, url, expect_json=expect_json)

            except requests.exceptions.Timeout as e:
                self._handle_timeout(e, method, url, endpoint)
            except requests.exceptions.HTTPError as e:
                self._handle_http_error(e, url, endpoint)
            except requests.exceptions.RequestException as e:
                self._handle_request_exception(e, url)
            except Exception as e:
                if self._handle_unexpected_exception(e, attempt, endpoint):
                    continue
                raise
        return None

    def _should_retry(self, response, url, attempt):
        if response.status_code in self.status_forcelist:
            logger.warning(
                f"Received retryable status {response.status_code} for {url} "
                f"(attempt {attempt + 1}/{self.max_retries})"
            )
            if attempt == self.max_retries - 1:
                response.raise_for_status()
            sleep_time = self.backoff_factor * (2**attempt)
            time.sleep(sleep_time)
            return True
        response.raise_for_status()
        return False

    def _parse_response(self, response, endpoint, url, expect_json: bool = True):
        if not expect_json:
            logger.info(f"Returning raw text response for {url}")
            return {"text": response.text, "headers": dict(response.headers)}
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            try:
                cleaned_text = response.text.replace("\\'", "'")
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                logger.exception(
                    f"Failed to parse JSON from {url} Status: {response.status_code}."
                    f"Response text: {response.text[:200]}..."
                )
                self._update_health(False, f"JSON Decode Error from {endpoint}")
                msg = f"Invalid JSON response from {endpoint}"
                raise self.error_class(msg, response.status_code) from None

    def _handle_timeout(self, e, method, url, endpoint):
        logger.error(f"Request timed out for {method} {url}: {e}")
        self._update_health(False, f"Timeout accessing {endpoint}")
        msg = f"Request timed out after {self.timeout}s"
        raise self.error_class(msg) from e

    def _handle_http_error(self, e, url, endpoint):
        status_code = e.response.status_code
        logger.error(f"HTTP {status_code} error for {url}: {e.response.text[:200]}...")
        self._update_health(False, f"HTTP Error {status_code} from {endpoint}")
        msg = f"API request failed with status {status_code}"
        raise self.error_class(msg, status_code) from e

    def _handle_request_exception(self, e, url):
        logger.error(f"Request failed for {url}: {e}")
        self._update_health(False, f"Request Exception: {type(e).__name__}")
        msg = "Failed to communicate with API"
        raise self.error_class(msg) from e

    def _handle_unexpected_exception(self, e, attempt, endpoint):
        logger.error(f"Unexpected error during API request to {endpoint}: {e}")
        self._update_health(False, f"Unexpected error: {type(e).__name__}")
        if attempt == self.max_retries - 1:
            return False
        sleep_time = self.backoff_factor * (2**attempt)
        logger.warning(f"Retrying ({attempt + 1}/{self.max_retries}) after error: {e}. Sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
        return True

    def _update_health(self, is_healthy: bool, message: str = "") -> None:
        """Updates and logs the API health status."""
        if self._last_health_status != is_healthy:
            if is_healthy:
                logger.info(f"{self.__class__.__name__} connection healthy")
            else:
                logger.warning(f"{self.__class__.__name__} connection unhealthy. Reason: {message}")
            self._last_health_status = is_healthy

    def check_api_health(self) -> bool:
        """Returns the last known health status of the API."""
        logger.info(f"Reporting API Health Status: {'Healthy' if self._last_health_status else 'Unhealthy'}")
        return self._last_health_status

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.session.close()
