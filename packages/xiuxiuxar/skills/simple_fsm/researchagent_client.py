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

"""Research Agent Client."""

from typing import Any

from aea.skills.base import Model

from packages.xiuxiuxar.skills.simple_fsm.base_client import BaseClient, BaseAPIError


STATUS_FORCELIST = (429, 500, 502, 503, 504)
TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5


class ResearchAgentAPIError(BaseAPIError):
    """Research Agent API specific error."""


class ResearchAgentClient(Model, BaseClient):
    """Client for interacting with the Research Agent."""

    def __init__(self, **kwargs: Any):
        base_url = kwargs.pop("base_url", None)
        api_key = kwargs.pop("api_key", None)

        # Initialize Model (for context, etc.)
        Model.__init__(self, **kwargs)
        # Initialize BaseClient
        BaseClient.__init__(
            self,
            base_url=base_url,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=STATUS_FORCELIST,
            headers={"Content-Type": "application/json"},
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
        self.context.logger.info(f"Getting tweets from multiple accounts: {params}")
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
        self.context.logger.info(f"Getting research tweets: {params}")
        return self._make_request("GET", endpoint, params=params)

    def get_tweets_filter(
        self,
        account: str | list[str],
        filter: str,
        limit: int | None = None,
        format: str | None = None,
    ) -> dict[str, Any] | None:
        """Get tweets from specific account(s) with a filter.

        Args:
        ----
            account: Single Twitter account name or list of account names
            filter: Filter string to apply
            limit: Maximum number of tweets to return
            format: Output format ('summary' or 'detailed')

        Returns:
        -------
            Filtered tweet data or None on error

        Raises:
        ------
            ValueError: If format is invalid or filter length exceeds 50 chars

        """
        if format and format not in {"summary", "detailed"}:
            msg = "format must be 'summary' or 'detailed'"
            raise ValueError(msg)

        if len(filter) > 50:
            msg = "filter must not exceed 50 characters"
            raise ValueError(msg)

        limit = self._validate_limit(limit)
        endpoint = "tweets/filter"
        params = {"account": account, "filter": filter, "limit": limit, "format": format}
        self.context.logger.info(f"Getting filtered tweets for {account}: {params}")
        return self._make_request("GET", endpoint, params=params)
