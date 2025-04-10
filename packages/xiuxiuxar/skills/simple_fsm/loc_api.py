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

"""LookOnChain API Client."""

import os
import logging
from datetime import UTC, datetime

from models import ScrapedDataItem
from base_api import BaseAPIError, BaseAPIClient, BaseAPIConfig


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
LOOKONCHAIN_BASE_URL = os.getenv("LOOKONCHAIN_BASE_URL", "https://www.lookonchain.com")
LOOKONCHAIN_SEARCH_ENDPOINT = os.getenv(
    "LOOKONCHAIN_SEARCH_ENDPOINT", "https://www.lookonchain.com/ashx/search_list.ashx"
)


class LookOnChainConfig(BaseAPIConfig):
    """Configuration for LookOnChain API."""

    base_url = LOOKONCHAIN_BASE_URL
    search_endpoint = LOOKONCHAIN_SEARCH_ENDPOINT
    timeout = 20
    retry_config = {"max_retries": 3, "backoff_factor": 0.5, "status_forcelist": (500, 502, 503, 504)}
    default_headers = {
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


class LookOnChainClient(BaseAPIClient):
    """Client for interacting with the LookOnChain API."""

    def __init__(
        self,
        base_url: str = LookOnChainConfig.base_url,
        search_endpoint: str = LookOnChainConfig.search_endpoint,
        max_retries: int = LookOnChainConfig.retry_config["max_retries"],
        backoff_factor: float = LookOnChainConfig.retry_config["backoff_factor"],
        timeout: int = LookOnChainConfig.timeout,
        status_forcelist: tuple[int, ...] = LookOnChainConfig.retry_config["status_forcelist"],
    ):
        """Initialize the LookOnChain API client.

        Args:
        ----
            base_url: Base URL for the API
            search_endpoint: Endpoint for search operations
            max_retries: Maximum number of retry attempts
            backoff_factor: Factor for exponential backoff
            timeout: Request timeout in seconds
            status_forcelist: HTTP status codes to retry on

        """
        if not search_endpoint or not search_endpoint.startswith("https://"):
            msg = "Invalid LookOnChain search endpoint URL"
            raise ValueError(msg)

        self.search_endpoint = search_endpoint
        self.source_name = "lookonchain"

        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            headers=LookOnChainConfig.default_headers,
            error_class=LookOnChainAPIError,
        )

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
            logger.error(f"API returned error or invalid format: {data}")
            return []

        content = data.get("content", [])
        if not isinstance(content, list):
            logger.error(f"Invalid content format: {content}")
            return []

        logger.debug(f"Found {len(content)} items in response")
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
            logger.debug(f"Failed to process item: {e}")
            return None
