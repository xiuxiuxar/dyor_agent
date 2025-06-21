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

"""Tree of Alpha Client."""

import re
import time
from typing import Any
from datetime import UTC, datetime, timedelta

import requests
from aea.skills.base import Model

from packages.xiuxiuxar.skills.dyor_app.base_client import BaseClient, BaseAPIError


STATUS_FORCELIST = (429, 500, 502, 503, 504)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.3"
    ),
    "Content-Type": "application/json",
}


class TreeOfAlphaAPIError(BaseAPIError):
    """Tree of Alpha API specific error."""


class TreeOfAlphaClient(Model, BaseClient):
    """Client for interacting with the Tree of Alpha."""

    def __init__(self, **kwargs: Any):
        name = kwargs.pop("name", "treeofalpha_client")
        skill_context = kwargs.pop("skill_context", None)
        base_url = kwargs.pop("base_url", None)
        news_endpoint = kwargs.pop("news_endpoint", None)
        cache_ttl = kwargs.pop("cache_ttl", 3600)  # Default 1 hour
        max_retries = kwargs.pop("max_retries", 3)  # Default 3 retries
        backoff_factor = kwargs.pop("backoff_factor", 0.5)  # Default 0.5 seconds
        timeout = kwargs.pop("timeout", 15)  # Default 15 seconds

        Model.__init__(self, name=name, skill_context=skill_context, **kwargs)
        BaseClient.__init__(
            self,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=STATUS_FORCELIST,
            headers=DEFAULT_HEADERS,
            error_class=TreeOfAlphaAPIError,
        )
        self.cache_ttl = cache_ttl
        self.news_endpoint = news_endpoint
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
            self.context.logger.debug("Returning cached news data")
            return self._cached_news

        try:
            params = {"limit": limit}
            headers = {"Content-Type": "application/json"}
            response = self._make_request(
                method="GET",
                endpoint=self.news_endpoint,
                params=params,
                headers=headers,
            )

            if not response:
                self.context.logger.warning("No news data received from API")
                return []

            if not isinstance(response, list):
                msg = f"Unexpected response format. Expected list, got {type(response)}"
                raise TreeOfAlphaAPIError(msg)

            # Update cache
            self._cached_news = response
            self._last_fetch_time = current_time
            self.context.logger.info(f"Successfully fetched and cached {len(response)} news items")
            return response

        except requests.exceptions.RequestException as e:
            self.context.logger.exception("Failed to fetch news data")
            msg = "Failed to communicate with API"
            raise TreeOfAlphaAPIError(msg) from e
        except Exception as e:
            self.context.logger.exception("Failed to fetch news data")
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

        # Use regex to match the query as a whole word (case-insensitive by default)
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(rf"\b{re.escape(query)}\b", flags)

        def matches_query(item: dict[str, Any]) -> bool:
            """Check if item matches search query as a whole word."""
            title = item.get("title", "")
            source = item.get("source", "")
            symbols = item.get("symbols", [])
            # Check in title, source, and each symbol
            return (
                bool(pattern.search(title))
                or bool(pattern.search(source))
                or any(pattern.search(symbol) for symbol in symbols)
            )

        matching_items = [item for item in news_items if matches_query(item)]
        self.context.logger.info(f"Found {len(matching_items)} items matching query: {query}")
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

        self.context.logger.info(f"Found {len(matching_items)} items for symbol: {symbol}")
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

        self.context.logger.info(f"Found {len(recent_items)} items from last {hours} hours")
        return recent_items
