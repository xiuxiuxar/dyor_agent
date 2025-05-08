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

"""LookOnChain Client."""

import os
from typing import Any
from datetime import UTC, datetime

from aea.skills.base import Model

from packages.xiuxiuxar.skills.simple_fsm.models import ScrapedDataItem
from packages.xiuxiuxar.skills.simple_fsm.base_client import BaseClient, BaseAPIError


STATUS_FORCELIST = (429, 500, 502, 503, 504)
DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }


class LookOnChainAPIError(BaseAPIError):
    """LookOnChain API specific error."""

    def __init__(self, message: str, error_type: str = "network", status_code: int | None = None):
        super().__init__(message, status_code)
        self.error_type = error_type


class NetworkError(LookOnChainAPIError):
    """Network-related error."""

    def __init__(self, source: str, message: str, status_code: int | None = None):
        super().__init__(f"Failed to fetch data from {source}: {message}", "network", status_code)


class ParsingError(LookOnChainAPIError):
    """Parsing-related error."""

    def __init__(self, source: str, message: str, status_code: int | None = None):
        super().__init__(f"Failed to parse data from {source}: {message}", "parsing", status_code)


class LookOnChainClient(Model, BaseClient):
    """Client for interacting with the LookOnChain."""

    def __init__(
        self, **kwargs: Any
        # base_url: str = LookOnChainConfig.base_url,
        # search_endpoint: str = LookOnChainConfig.search_endpoint,
        # max_retries: int = LookOnChainConfig.retry_config["max_retries"],
        # backoff_factor: float = LookOnChainConfig.retry_config["backoff_factor"],
        # timeout: int = LookOnChainConfig.timeout,
        # status_forcelist: tuple[int, ...] = LookOnChainConfig.retry_config["status_forcelist"],
    ):
        base_url = kwargs.pop("base_url", None)
        search_endpoint = kwargs.pop("search_endpoint", None)
        timeout = kwargs.pop("timeout", 15)
        max_retries = kwargs.pop("max_retries", 3)
        backoff_factor = kwargs.pop("backoff_factor", 0.5)

        Model.__init__(self, **kwargs)
        BaseClient.__init__(
            self,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=STATUS_FORCELIST,
            headers=DEFAULT_HEADERS,
        )
        self.search_endpoint = search_endpoint
        self.source_name = "lookonchain"
        self.error_class = LookOnChainAPIError

    def search(self, query: str, page: int = 1, count: int = 20) -> list[ScrapedDataItem]:
        """Search for content using the LookOnChain API.

        Args:
        ----
            query: Search term
            page: Page number for pagination
            count: Number of items per page

        Returns:
        -------
            List of items matching the search query

        Raises:
        ------
            NetworkError: If the API request fails due to network issues
            ParsingError: If the response cannot be parsed

        """
        try:
            params = {
                "max_time": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "keyword": query,
                "protype": "feeds",
                "page": page,
                "count": count,
            }

            headers = {"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"}
            response = self._make_request(
                method="GET",
                endpoint=self.search_endpoint,
                params=params,
                headers=headers,
                base_url_override="",  # Use the full search_endpoint URL
            )

            if not response:
                return []

            return self._process_response_data(response)

        except LookOnChainAPIError as e:
            if "Invalid JSON" in str(e):
                raise ParsingError(self.source_name, str(e)) from e
            raise NetworkError(self.source_name, str(e)) from e

    def _process_response_data(self, data: dict) -> list[ScrapedDataItem]:
        """Process parsed JSON data into ScrapedDataItems.

        Args:
        ----
            data: Parsed JSON data

        Returns:
        -------
            List of ScrapedDataItems

        """
        if not isinstance(data, dict) or data.get("success") != "Y":
            self.context.logger.error(f"API returned error or invalid format: {data}")
            return []

        content = data.get("content", [])
        if not isinstance(content, list):
            self.context.logger.error(f"Invalid content format: {content}")
            return []

        self.context.logger.debug(f"Found {len(content)} items in response")
        return [item for item in (self._create_scraped_item(item_data) for item_data in content) if item is not None]

    def _create_scraped_item(self, item_data: dict) -> ScrapedDataItem | None:
        """Create a ScrapedDataItem from raw item data.

        Args:
        ----
            item_data: Raw item data from API

        Returns:
        -------
            ScrapedDataItem if successful, None if creation fails

        """
        try:
            abstract = item_data.get("sabstract", "").replace("<br/>", "\n").replace("<br>", "\n")

            return ScrapedDataItem(
                source=self.source_name,
                title=item_data["stitle"],
                url=f"{self.base_url}/feeds/{item_data['id']}",
                summary=abstract,
                timestamp=item_data.get("dcreate_time"),
                metadata={
                    "author": item_data.get("sauthor_name", ""),
                    "type": item_data.get("stype", ""),
                    "image": item_data.get("spic", ""),
                    "full_content": abstract,
                },
            )
        except (KeyError, Exception) as e:
            self.context.logger.debug(f"Failed to process item: {e}")
            return None

    def validate_config(self) -> None:
        """Validate required environment variables are set."""
        required_vars = ["LOOKONCHAIN_BASE_URL", "LOOKONCHAIN_SEARCH_ENDPOINT"]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            msg = f"Missing required environment variables: {', '.join(missing)}"
            raise ValueError(msg)
