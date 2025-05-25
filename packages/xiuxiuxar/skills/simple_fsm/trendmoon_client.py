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

"""Trendmoon Base Client."""

from typing import Any
from inspect import signature
from functools import wraps
from urllib.parse import urlparse
from collections.abc import Callable

from aea.skills.base import Model
from dateutil.parser import ParserError, parse

from packages.xiuxiuxar.skills.simple_fsm.base_client import BaseClient, BaseAPIError


STATUS_FORCELIST = (429, 500, 502, 503, 504)


def validate_iso_dates(*date_param_names: str) -> Callable:
    """Decorator to validate ISO 8601 date parameters."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name in date_param_names:
                if param_name in bound_args.arguments:
                    date_value = bound_args.arguments[param_name]
                    if date_value is not None and not _validate_iso_date(date_value):
                        msg = f"Invalid {param_name} format. Must be ISO 8601 (e.g., 2023-10-26T00:00:00)"
                        raise ValueError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class TimeIntervals:
    """Time intervals for Trendmoon API."""

    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    THREE_DAYS = "3d"
    ONE_WEEK = "1w"

    @classmethod
    def valid_intervals(cls) -> set[str]:
        """Get all valid time intervals."""
        return {
            cls.ONE_HOUR,
            cls.FOUR_HOURS,
            cls.TWELVE_HOURS,
            cls.ONE_DAY,
            cls.THREE_DAYS,
            cls.ONE_WEEK,
        }


class SocialTrendIntervals:
    """Social trend intervals for Trendmoon API."""

    ONE_HOUR = "1h"
    ONE_DAY = "1d"

    @classmethod
    def valid_intervals(cls) -> set[str]:
        """Get all valid social trend intervals."""
        return {cls.ONE_HOUR, cls.ONE_DAY}


class MatchModes:
    """Match modes for Trendmoon API."""

    EXACT = "exact"
    ANY = "any"
    ALL = "all"
    FUZZY = "fuzzy"
    PARTIAL = "partial"

    @classmethod
    def valid_modes(cls) -> set[str]:
        """Get all valid match modes."""
        return {cls.EXACT, cls.ANY, cls.ALL, cls.FUZZY, cls.PARTIAL}


class TrendmoonAPIError(BaseAPIError):
    """Trendmoon API specific error."""


class TrendmoonClient(Model, BaseClient):
    """Client for interacting with Trendmoon."""

    def __init__(self, **kwargs: Any):
        name = kwargs.pop("name", "trendmoon_client")
        skill_context = kwargs.pop("skill_context", None)
        api_key = kwargs.pop("api_key", None)
        base_url = kwargs.pop("base_url", None)
        insights_url = kwargs.pop("insights_url", None)
        max_retries = kwargs.pop("max_retries", 3)
        backoff_factor = kwargs.pop("backoff_factor", 0.5)
        timeout = kwargs.pop("timeout", 15)

        if not api_key:
            msg = "API key is required"
            raise ValueError(msg)

        Model.__init__(self, name=name, skill_context=skill_context, **kwargs)
        BaseClient.__init__(
            self,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=STATUS_FORCELIST,
            headers={"Content-Type": "application/json"},
            api_key=api_key,
            error_class=TrendmoonAPIError,
        )

        if insights_url:
            parsed = urlparse(insights_url)
            if parsed.scheme not in {"http", "https"}:
                msg = f"Invalid {self.__class__.__name__} insights URL {insights_url}"
                raise ValueError(msg)
            self.insights_url = insights_url.rstrip("/")
        else:
            self.insights_url = None

    def _make_insights_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Make a request to the insights API."""
        return self._make_request(
            method=method,
            endpoint=endpoint,
            params=params,
            json_data=json_data,
            base_url_override=self.insights_url,
        )

    # -- API Endpoint Methods

    # -- Processed Insights
    def get_project_summary(self, project_name: str) -> dict[str, Any] | None:
        """Retrieves a summary for a specific project."""
        endpoint = "/social/project_summary"
        params = {"project_name": project_name}
        self.context.logger.info(f"Fetching project summary for: {project_name}")
        return self._make_request("GET", endpoint, params=params)

    def get_top_categories_today(self) -> dict[str, Any] | None:
        """Retrieves the top categories for today."""
        endpoint = "/get_top_categories_today"
        self.context.logger.info("Fetching top categories for today")
        return self._make_insights_request("GET", endpoint)

    def get_top_alerts_today(self) -> dict[str, Any] | None:
        """Retrieves the top alerts for today."""
        endpoint = "/get_top_alerts_today"
        self.context.logger.info("Fetching top alerts for today")
        return self._make_insights_request("GET", endpoint)

    def get_category_dominance(self, category_name: str, duration: int) -> list[dict[str, Any]] | None:
        """Retrieves the dominance of a category over a specified duration.

        Args:
        ----
            category_name: Name of the category to analyze
            duration: Number of days to look back (e.g. 7, 30, 90)

        Returns:
        -------
            List of category dominance data points, each containing:
            - date: ISO 8601 timestamp
            - category_name: Name of the category
            - category_dominance: Dominance value
            - category_market_cap: Market cap of the category
            - dominance_pct_change: Percentage change in dominance
            - market_cap_pct_change: Percentage change in market cap

        Raises:
        ------
            ValueError: If category_name is empty or duration is invalid
            TrendmoonAPIError: If the API request fails

        """
        if not category_name:
            msg = "category_name must not be empty"
            raise ValueError(msg)
        if not isinstance(duration, int) or duration <= 0:
            msg = "duration must be a positive integer"
            raise ValueError(msg)

        endpoint = "/categories/dominance"
        params = {"category_name": category_name, "duration": duration}
        self.context.logger.info(f"Fetching category dominance for: {category_name} over duration: {duration}")
        return self._make_request("GET", endpoint, params=params)

    def get_top_category_alerts(self) -> dict[str, Any] | None:
        """Retrieves the top category alerts filtered by top category for the day."""
        endpoint = "/get_top_category_alerts"
        self.context.logger.info("Fetching top category alerts")
        return self._make_insights_request("GET", endpoint)

    # -- Social
    def get_social_trend(
        self,
        contract_address: str | None = None,
        symbol: str | None = None,
        project_name: str | None = None,
        coin_id: str | None = None,
        date_interval: int | None = None,
        time_interval: str | None = None,
    ) -> dict[str, Any] | None:
        """Retrieves social trend data."""

        if time_interval and time_interval not in SocialTrendIntervals.valid_intervals():
            msg = f"Invalid time_interval. Must be one of: {', '.join(SocialTrendIntervals.valid_intervals())}"
            raise ValueError(msg)

        endpoint = "/social/trend"
        params = {
            "contract_address": contract_address,
            "symbol": symbol,
            "project_name": project_name,
            "coin_id": coin_id,
            "date_interval": date_interval,
            "time_interval": time_interval,
        }
        provided_params = {k: v for k, v in params.items() if v is not None}
        self.context.logger.info(f"Fetching social trends with params: {provided_params}")
        return self._make_request("GET", endpoint, params=params)

    def get_keyword_trend(
        self,
        keyword: str,
        duration: int | None = 7,
        time_interval: str | None = "1d",
        match_mode: str | None = "exact",
    ) -> dict[str, Any] | None:
        """Retrieves aggregated trend data for keywords over time.

        Args:
        ----
            keyword: The keyword, exact phrase, or comma-separated terms to search for
                    (e.g., "bitcoin", "bitcoin moon", "btc,eth")
            duration: Number of days to look back from current time (default: 7)
            time_interval: Time interval for aggregation (default: "1d")
                          Accepted values: "1h", "4h", "12h", "1d", "3d", "1w"
            match_mode: Strategy for matching keywords (default: "exact")
                       Accepted values:
                       - "exact": Match the exact keyword or phrase
                       - "any": Match any of the comma-separated terms
                       - "all": Match all comma-separated terms (order doesn't matter)
                       - "fuzzy": Match similar spellings (handles typos)
                       - "partial": Match if keyword appears as part of a word

        Returns:
        -------
            Optional[dict[str, Any]]: Keyword trend data or None on error

        Raises:
        ------
            TrendmoonAPIError: If the request fails after retries
            ValueError: If invalid time_interval or match_mode is provided

        """
        if time_interval and time_interval not in TimeIntervals.valid_intervals():
            msg = f"Invalid time_interval. Must be one of: {', '.join(TimeIntervals.valid_intervals())}"
            raise ValueError(msg)

        if match_mode and match_mode not in MatchModes.valid_modes():
            msg = f"Invalid match_mode. Must be one of: {', '.join(MatchModes.valid_modes())}"
            raise ValueError(msg)

        endpoint = "/social/keyword"
        params = {"keyword": keyword, "duration": duration, "time_interval": time_interval, "match_mode": match_mode}
        provided_params = {k: v for k, v in params.items() if v is not None and v != ""}
        self.context.logger.info(f"Fetching keyword trends for '{keyword}' with params: {provided_params}")
        return self._make_request("GET", endpoint, params=params)

    # -- Messages

    @validate_iso_dates("start_date", "end_date")
    def search_messages(
        self,
        text: str | None = None,
        group_username: str | None = None,
        username: str | None = None,
        message_type: str | None = None,
        user_is_bot: bool | None = None,
        user_is_spammer: bool | None = None,
        spam_flag: bool | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        from_: int | None = 0,
        size: int | None = 100,
    ) -> dict[str, Any] | None:
        """Search messages based on various criteria.

        Args:
        ----
            text: Search in message text
            group_username: Filter by group username
            username: Filter by sender username
            message_type: Filter by message type ("raw" or "clean")
            user_is_bot: Filter by bot status
            user_is_spammer: Filter by spammer status
            spam_flag: Filter by spam flag
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format
            from_: Starting offset for pagination
            size: Number of messages to return

        Returns:
        -------
            Optional[dict[str, Any]]: Search results or None on error

        Raises:
        ------
            TrendmoonAPIError: If the request fails after retries

        """
        endpoint = "/messages/search"
        params = {
            "text": text,
            "group_username": group_username,
            "username": username,
            "message_type": message_type,
            "user_is_bot": user_is_bot,
            "user_is_spammer": user_is_spammer,
            "spam_flag": spam_flag,
            "start_date": start_date,
            "end_date": end_date,
            "from": from_,  # Note: Using 'from' instead of 'from_' as it's the API parameter name
            "size": size,
        }

        params = {k: v for k, v in params.items() if v is not None}
        self.context.logger.info(f"Searching messages with params: {params}")
        return self._make_request("GET", endpoint, params=params)

    @validate_iso_dates("start_date", "end_date")
    def get_user_messages(self, username: str, start_date: str, end_date: str) -> dict[str, Any] | None:
        """Retrieves messages for a specific user.

        Args:
        ----
            username: The username of the user to retrieve messages for
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format

        Returns:
        -------
            Optional[dict[str, Any]]: Messages for the user or None on error

        Raises:
        ------
            TrendmoonAPIError: If the request fails after retries
            ValueError: If invalid date format

        """
        endpoint = "/messages/user"
        params = {"username": username, "start_date": start_date, "end_date": end_date}
        self.context.logger.info(f"Fetching messages for user: {username} from {start_date} to {end_date}")
        return self._make_request("GET", endpoint, params=params)

    @validate_iso_dates("start_date", "end_date")
    def get_chat_messages(
        self,
        start_date: str,
        end_date: str,
        group_username: str | None = None,
        chat_id: int | None = None,
        message_type: str | None = None,
        with_spams: bool | None = None,
        from_: int | None = 0,
        size: int | None = 100,
    ) -> dict[str, Any] | None:
        """Retrieves messages from a specific chat group.

        Args:
        ----
            start_date: Start date in ISO 8601 format (e.g., 2023-10-26T00:00:00)
            end_date: End date in ISO 8601 format (e.g., 2023-10-27T23:59:59)
            group_username: The username of the group chat
            chat_id: The internal Telegram ID of the chat
            message_type: Filter by message type ("raw" or "clean")
            with_spams: Include spam messages if true, exclude if false
            from_: Starting offset for pagination
            size: Number of messages to return per page

        Returns:
        -------
            Optional[dict[str, Any]]: Chat messages or None on error

        Raises:
        ------
            TrendmoonAPIError: If the request fails after retries
            ValueError: If neither group_username nor chat_id is provided

        """
        if not group_username and not chat_id:
            msg = "Either group_username or chat_id must be provided"
            raise ValueError(msg)
        if message_type and message_type not in {"raw", "clean"}:
            msg = "message_type must be either 'raw' or 'clean'"
            raise ValueError(msg)

        endpoint = "/messages/chat"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "group_username": group_username,
            "chat_id": chat_id,
            "message_type": message_type,
            "with_spams": with_spams,
            "from": from_,  # Note: Using 'from' instead of 'from_' as it's the API parameter name
            "size": size,
        }
        params = {k: v for k, v in params.items() if v is not None}
        identifier = group_username or f"chat_id:{chat_id}"
        self.context.logger.info(f"Fetching messages for chat {identifier} from {start_date} to {end_date}")
        return self._make_request("GET", endpoint, params=params)

    @validate_iso_dates("start_date", "end_date")
    def get_messages_by_timeframe(
        self,
        start_date: str,
        end_date: str,
        message_type: str | None = None,
        with_spams: bool | None = None,
        from_: int | None = 0,
        size: int | None = 100,
    ) -> dict[str, Any] | None:
        """Retrieves all messages across monitored chats within a date range.

        Args:
        ----
            start_date: Start date in ISO 8601 format (e.g., 2023-01-01T00:00:00)
            end_date: End date in ISO 8601 format (e.g., 2023-01-31T23:59:59)
            message_type: Filter by message type ("raw" or "clean")
            with_spams: Include spam messages if true, exclude if false
            from_: Starting offset for pagination
            size: Number of messages to return per page

        Returns:
        -------
            Optional[dict[str, Any]]: Messages within the timeframe or None on error

        Raises:
        ------
            TrendmoonAPIError: If the request fails after retries
            ValueError: If invalid date format

        """
        endpoint = "/messages/timeframe"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "message_type": message_type,
            "with_spams": with_spams,
            "from": from_,  # Note: Using 'from' instead of 'from_' as it's the API parameter name
            "size": size,
        }
        params = {k: v for k, v in params.items() if v is not None}
        self.context.logger.info(f"Fetching messages from {start_date} to {end_date}")
        return self._make_request("GET", endpoint, params=params)

    # -- Chat
    def get_chat_information_by_group_username(self, group_username: str) -> dict[str, Any] | None:
        """Retrieves chat information by group_username.

        Args:
        ----
            group_username: Telegram group username

        Returns:
        -------
            Optional[dict[str, Any]]: Chat information or None on error

        Raises:
        ------
            TrendmoonAPIError: If the request fails after retries

        """
        endpoint = f"/chats/{group_username}"
        self.context.logger.info(f"Fetching chat information for group username: {group_username}")
        return self._make_request("GET", endpoint)

    @validate_iso_dates("start_date", "end_date")
    def get_chat_activity(
        self, group_username: str, start_date: str, end_date: str, from_: int | None = 0, size: int | None = 100
    ) -> dict[str, Any] | None:
        """Retrieves activity data for a specific chat group.

        Args:
        ----
            group_username: Telegram group username
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format
            from_: Starting offset for pagination
            size: Number of records to return per page

        Returns:
        -------
            Optional[dict[str, Any]]: Chat activity data or None on error

        Raises:
        ------
            TrendmoonAPIError: If the request fails after retries

        """
        endpoint = "/chat_activity"
        params = {
            "group_username": group_username,
            "start_date": start_date,
            "end_date": end_date,
            "from": from_,  # Note: Using 'from' instead of 'from_' as it's the API parameter name
            "size": size,
        }
        params = {k: v for k, v in params.items() if v is not None}
        self.context.logger.info(f"Fetching chat activity for group {group_username} from {start_date} to {end_date}")
        return self._make_request("GET", endpoint, params=params)

    # -- Coins
    def search_coin(
        self,
        name: str | None = None,
        symbol: str | None = None,
        category: str | None = None,
        chain: str | None = None,
        contract_address: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Searches for a coin/token by its symbol."""
        endpoint = "/coins/search"
        params = {
            "name": name,
            "symbol": symbol,
            "category": category,
            "chain": chain,
            "contract_address": contract_address,
        }
        provided_params = {k: v for k, v in params.items() if v is not None and v != ""}
        self.context.logger.info(f"Searching for coin with params: {provided_params}")
        return self._make_request("GET", endpoint, params=params)

    def get_platforms(self) -> dict[str, Any] | None:
        """Retrieves all available platforms that have coins in the index."""
        endpoint = "/coins/platforms"
        self.context.logger.info("Fetching all platforms")
        return self._make_request("GET", endpoint)

    def get_coin_details(
        self,
        contract_address: str | None = None,
        coin_id: str | None = None,
        symbol: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Retrieves detailed information for a specific coin/token."""
        endpoint = "/coins/details"
        params = {
            "contract_address": contract_address,
            "coin_id": coin_id,
            "symbol": symbol,
            "project_name": project_name,
        }
        if not any(params.values()):
            msg = "At least one parameter (contract_address, coin_id, symbol, or project_name) must be provided"
            raise ValueError(msg)

        provided_params = {k: v for k, v in params.items() if v is not None and v != ""}
        self.context.logger.info(f"Fetching coin details for: {provided_params}")
        return self._make_request("GET", endpoint, params=params)


def _validate_iso_date(date_str: str) -> bool:
    """Validate ISO 8601 date string."""
    try:
        parse(date_str, yearfirst=True)
        return True
    except (ParserError, TypeError, ValueError):
        return False
