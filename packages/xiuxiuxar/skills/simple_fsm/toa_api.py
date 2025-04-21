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

"""Tree of Alpha API Client."""

import os
import time
import logging
from typing import Any
from datetime import UTC, datetime, timedelta

import requests
from base_api import BaseAPIError, BaseAPIClient, BaseAPIConfig


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
TREE_OF_ALPHA_BASE_URL = os.environ["TREE_OF_ALPHA_BASE_URL"]
TREE_OF_ALPHA_NEWS_ENDPOINT = os.environ["TREE_OF_ALPHA_NEWS_ENDPOINT"]
TREE_OF_ALPHA_CACHE_TTL = int(os.environ["TREE_OF_ALPHA_CACHE_TTL"])


class TreeOfAlphaConfig(BaseAPIConfig):
    """Configuration for Tree of Alpha API."""

    base_url = TREE_OF_ALPHA_BASE_URL
    news_endpoint = TREE_OF_ALPHA_NEWS_ENDPOINT
    cache_ttl = TREE_OF_ALPHA_CACHE_TTL
    timeout = 30  # Increased timeout for large response
    retry_config = {
        "max_retries": 3,
        "backoff_factor": 0.5,
        "status_forcelist": (429, 500, 502, 503, 504),
        "connect": 5,
        "read": 5,
    }
    default_headers = {"Content-Type": "application/json"}


class TreeOfAlphaAPIError(BaseAPIError):
    """Tree of Alpha API specific error."""


class TreeOfAlphaAPI(BaseAPIClient):
    """Client for interacting with the Tree of Alpha API."""

    def __init__(
        self,
        base_url: str = TreeOfAlphaConfig.base_url,
        news_endpoint: str = TreeOfAlphaConfig.news_endpoint,
        cache_ttl: int = TreeOfAlphaConfig.cache_ttl,
        max_retries: int = TreeOfAlphaConfig.retry_config["max_retries"],
        backoff_factor: float = TreeOfAlphaConfig.retry_config["backoff_factor"],
        timeout: int = TreeOfAlphaConfig.timeout,
        status_forcelist: tuple[int, ...] = TreeOfAlphaConfig.retry_config["status_forcelist"],
    ):
        """Initialize the Tree of Alpha API client.

        Args:
        ----
            base_url: Base URL for the API
            news_endpoint: Endpoint for news operations
            cache_ttl: Time to live for cache in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Factor for exponential backoff
            timeout: Request timeout in seconds
            status_forcelist: HTTP status codes to retry on

        """
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            headers=TreeOfAlphaConfig.default_headers,
            error_class=TreeOfAlphaAPIError,
        )

        self.news_endpoint = news_endpoint
        self.cache_ttl = cache_ttl
        self._cache: dict[str, Any] = {}
        self._last_fetch_time: float = 0
        self._cached_news: list[dict[str, Any]] = []

    def get_news(self, limit: int = 3000, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Fetch news from Tree of Alpha API with caching.

        Args:
        ----
            limit: Number of news items to fetch
            force_refresh: Force refresh cache even if TTL hasn't expired

        Returns:
        -------
            List of news items

        Raises:
        ------
            TreeOfAlphaAPIError: If the API request fails

        """
        current_time = time.time()
        cache_age = current_time - self._last_fetch_time

        # Return cached data if within TTL and not forced refresh
        if not force_refresh and cache_age < self.cache_ttl and self._cached_news:
            logger.debug("Returning cached news data")
            return self._cached_news

        try:
            params = {"limit": limit}
            response = self._make_request(
                method="GET",
                endpoint=self.news_endpoint,
                params=params,
            )

            if not response:
                logger.warning("No news data received from API")
                return []

            if not isinstance(response, list):
                msg = f"Unexpected response format. Expected list, got {type(response)}"
                raise TreeOfAlphaAPIError(msg)

            # Update cache
            self._cached_news = response
            self._last_fetch_time = current_time
            logger.info(f"Successfully fetched and cached {len(response)} news items")
            return response

        except requests.exceptions.RequestException as e:
            logger.exception("Failed to fetch news data")
            msg = "Failed to communicate with API"
            raise TreeOfAlphaAPIError(msg) from e
        except Exception as e:
            logger.exception("Failed to fetch news data")
            raise TreeOfAlphaAPIError(str(e)) from e

    def search_news(
        self,
        query: str,
        case_sensitive: bool = False,
        limit: int = 3000,
    ) -> list[dict[str, Any]]:
        """Search news items for a specific query.

        Args:
        ----
            query: Search term
            case_sensitive: Whether to perform case-sensitive search
            limit: Maximum number of news items to search through

        Returns:
        -------
            List of matching news items

        Raises:
        ------
            TreeOfAlphaAPIError: If the news fetch fails

        """
        news_items = self.get_news(limit=limit)

        if not query:
            return news_items

        def matches_query(item: dict[str, Any]) -> bool:
            """Check if item matches search query."""
            if not case_sensitive:
                search_query = query.lower()
                title = item.get("title", "").lower()
                source = item.get("source", "").lower()
                symbols = [s.lower() for s in item.get("symbols", [])]
            else:
                search_query = query
                title = item.get("title", "")
                source = item.get("source", "")
                symbols = item.get("symbols", [])

            return search_query in title or search_query in source or any(search_query in symbol for symbol in symbols)

        matching_items = [item for item in news_items if matches_query(item)]
        logger.info(f"Found {len(matching_items)} items matching query: {query}")
        return matching_items

    def get_news_by_symbol(self, symbol: str, limit: int = 3000) -> list[dict[str, Any]]:
        """Get news items related to a specific symbol.

        Args:
        ----
            symbol: Trading symbol to search for
            limit: Maximum number of news items to search through

        Returns:
        -------
            List of news items related to the symbol

        """
        news_items = self.get_news(limit=limit)
        symbol = symbol.upper()

        matching_items = [item for item in news_items if symbol in (item.get("symbols", []) or [])]

        logger.info(f"Found {len(matching_items)} items for symbol: {symbol}")
        return matching_items

    def get_latest_news(self, hours: int = 24, limit: int = 3000) -> list[dict[str, Any]]:
        """Get news items from the last N hours.

        Args:
        ----
            hours: Number of hours to look back
            limit: Maximum number of news items to search through

        Returns:
        -------
            List of recent news items

        """
        news_items = self.get_news(limit=limit)
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        recent_items = [
            item for item in news_items if datetime.fromtimestamp(item.get("time", 0) / 1000, UTC) > cutoff_time
        ]

        logger.info(f"Found {len(recent_items)} items from last {hours} hours")
        return recent_items
