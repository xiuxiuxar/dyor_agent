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

"""Research Agent API Client."""

import os
import logging
from typing import Any

from base_api import BaseAPIError, BaseAPIClient, BaseAPIConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ResearchAgentConfig(BaseAPIConfig):
    """Configuration for Research Agent API."""

    base_url = os.getenv("RESEARCH_AGENT_BASE_URL", "https://docker.trendmoon.io/v1")
    api_key = os.getenv("RESEARCH_AGENT_API_KEY")
    timeout = 30
    retry_config = {
        "max_retries": 3,
        "backoff_factor": 1.0,
        "status_forcelist": (429, 500, 502, 503, 504),
        "connect": 5,
        "read": 5,
    }
    default_headers = {"Content-Type": "application/json"}


class ResearchAgentAPIError(BaseAPIError):
    """Research Agent API specific error."""


class ResearchAgentAPI(BaseAPIClient):
    """Client for interacting with the Research Agent API."""

    def __init__(
        self,
        api_key: str = ResearchAgentConfig.api_key,
        base_url: str = ResearchAgentConfig.base_url,
        max_retries: int = ResearchAgentConfig.retry_config["max_retries"],
        backoff_factor: float = ResearchAgentConfig.retry_config["backoff_factor"],
        timeout: int = ResearchAgentConfig.timeout,
        status_forcelist: tuple[int, ...] = ResearchAgentConfig.retry_config["status_forcelist"],
    ):
        if not api_key:
            msg = "Research Agent API key is required"
            raise ValueError(msg)

        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            headers=ResearchAgentConfig.default_headers,
            api_key=api_key,
            error_class=ResearchAgentAPIError,
        )

    def _validate_limit(self, limit: int | None) -> int:
        """Validate and normalize limit parameter."""
        default_limit = 10
        max_limit = 50
        if limit is None:
            return default_limit
        return min(limit, max_limit)

    def get_tweets_multi_account(
        self,
        limit: int | None = None,
        since: str | None = None,
    ) -> dict[str, Any] | None:
        """Get tweets from multiple crypto research accounts.

        Args:
        ----
            limit: Maximum number of tweets to return
            since: Only return tweets after this timestamp

        Returns:
        -------
            Tweet data from multiple accounts or None on error

        """
        limit = self._validate_limit(limit)
        endpoint = "tweets/multi-account"
        params = {"limit": limit, "since": since}
        logger.info(f"Getting tweets from multiple accounts: {params}")
        return self._make_request("GET", endpoint, params=params)

    def get_tweets_research(
        self,
        limit: int | None = None,
        summarize: bool | None = False,
    ) -> dict[str, Any] | None:
        """Get latest tweets from spotonchain.

        Args:
        ----
            limit: Maximum number of tweets to return
            summarize: Whether to return summarized tweets

        Returns:
        -------
            Research tweets data or None on error

        """
        limit = self._validate_limit(limit)
        endpoint = "tweets/research"
        params = {"limit": limit, "summarize": summarize}
        logger.info(f"Getting research tweets: {params}")
        return self._make_request("GET", endpoint, params=params)

    def get_tweets_filter(
        self,
        account: str,
        filter: str,
        limit: int | None = None,
        format: str | None = None,
    ) -> dict[str, Any] | None:
        """Get tweets from a specific account with a filter.

        Args:
        ----
            account: Twitter account name
            filter: Filter string to apply
            limit: Maximum number of tweets to return
            format: Output format ('summary' or 'detailed')

        Returns:
        -------
            Filtered tweet data or None on error

        Raises:
        ------
            ValueError: If format is invalid

        """
        if format and format not in {"summary", "detailed"}:
            msg = "format must be 'summary' or 'detailed'"
            raise ValueError(msg)

        limit = self._validate_limit(limit)
        endpoint = "tweets/filter"
        params = {"account": account, "filter": filter, "limit": limit, "format": format}
        logger.info(f"Getting filtered tweets for {account}: {params}")
        return self._make_request("GET", endpoint, params=params)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
