"""Trendmoon API Client."""

import os
import logging
from typing import Any
from inspect import signature
from functools import wraps
from collections.abc import Callable

import requests
from dateutil.parser import ParserError, parse
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


TRENDMOON_API_KEY = os.getenv("TRENDMOON_API_KEY")
TRENDMOON_BASE_URL = os.getenv("TRENDMOON_BASE_URL")
TRENDMOON_INSIGHTS_URL = os.getenv("TRENDMOON_INSIGHTS_URL")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


class TrendmoonAPIError(Exception):
    """Custom exception for Trendmoon API related errors."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class TrendmoonConfig:
    """Configuration for Trendmoon API."""

    api_key = TRENDMOON_API_KEY
    base_url = TRENDMOON_BASE_URL
    insights_url = TRENDMOON_INSIGHTS_URL
    timeout = 15
    retry_config = {"max_retries": 3, "backoff_factor": 0.5, "status_forcelist": (429, 500, 502, 503, 504)}


class TrendmoonAPI:
    """Client for interacting with the Trendmoon API."""

    def __init__(
        self,
        api_key: str = TrendmoonConfig.api_key,
        base_url: str = TrendmoonConfig.base_url,
        insights_url: str = TrendmoonConfig.insights_url,
        max_retries: int = TrendmoonConfig.retry_config["max_retries"],
        backoff_factor: float = TrendmoonConfig.retry_config["backoff_factor"],
        timeout: int = TrendmoonConfig.timeout,
        status_forcelist: tuple[int, ...] = TrendmoonConfig.retry_config["status_forcelist"],
    ):
        if not api_key:
            msg = "Trendmoon API key is required"
            raise ValueError(msg)
        if not base_url or not base_url.startswith("https://"):
            msg = "Invalid Trendmoon API base URL"
            raise ValueError(msg)
        if not insights_url or not insights_url.startswith("https://"):
            msg = "Invalid Trendmoon Insights API base URL"
            raise ValueError(msg)

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.insights_url = insights_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Api-key": self.api_key,
            }
        )

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

        self._last_health_status: bool = True
        logger.info("Trendmoon API client initialized")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        use_insights_url: bool = False,
    ) -> dict[str, Any] | None:
        """Make a request to the Trendmoon API."""
        base_url = self.insights_url if use_insights_url else self.base_url
        url = f"{base_url}/{endpoint.lstrip('/')}"
        log_params = params or (json_data or {})
        logger.debug(f"Sending {method} request to {url} with params: {log_params}")
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                timeout=self.timeout,
            )
            response.raise_for_status()

            self._update_health(True)
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                logger.exception(
                    f"Failed to parse JSON response from {url} Status: {response.status_code}."
                    f"Response text: {response.text[:200]}..."
                )
                self._update_health(False, f"JSON Decode Error from {endpoint}")
                msg = f"Invalid JSON response from API endpoint {endpoint}."
                raise TrendmoonAPIError(msg, status_code=response.status_code) from None

        except requests.exceptions.Timeout as e:
            logger.exception(f"Request timed out for {method} {url}: {e}")
            self._update_health(False, f"Timeout accessing {endpoint}")
            msg = f"Request timed out after {self.timeout}s for endpoint {endpoint}."
            raise TrendmoonAPIError(msg) from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            log_msg = f"HTTP Error {status_code} for {method} {url}: {e}. Response: {e.response.text[:200]}..."
            logger.exception(log_msg)
            self._update_health(False, f"HTTP Error {status_code} from {endpoint}")
            msg = f"API request failed for endpoint {endpoint} with status {status_code}."
            raise TrendmoonAPIError(msg, status_code=status_code) from e

        except requests.exceptions.RequestException as e:
            log_msg = f"General Request Exception for {method} {url}: {e}"
            logger.exception(log_msg)
            self._update_health(False, f"Request Exception accessing {endpoint}: {type(e).__name__}")
            msg = f"An unexpected error occurred communicating with the API endpoint {endpoint}."
            raise TrendmoonAPIError(msg) from e

        except Exception as e:
            logger.exception(f"An unexpected error occurred during API request to {endpoint}: {e}")
            self._update_health(False, f"Unexpected error: {type(e).__name__}")
            msg = f"An unexpected error occurred: {e}"
            raise TrendmoonAPIError(msg) from e

        return None

    def _update_health(self, is_healthy: bool, message: str = ""):
        """Updates and logs the API health status."""
        if self._last_health_status != is_healthy:
            if is_healthy:
                logger.info("Trendmoon API connection healthy.")
            else:
                logger.warning(f"Trendmoon API connection unhealthy. Reason: {message}")
            self._last_health_status = is_healthy

    def check_api_health(self) -> bool:
        """Returns the last known health status of the API based on request success.
        Note: This is reactive, based on the last call's success/failure.
        """
        logger.info(f"Reporting API Health Status: {'Healthy' if self._last_health_status else 'Unhealthy'}")
        return self._last_health_status

    # -- API Endpoint Methods

    # -- Processed Insights
    def get_project_summary(self, project_name: str) -> dict[str, Any] | None:
        """Retrieves a summary for a specific project."""
        endpoint = "/get_project_summary"
        params = {"project_name": project_name}
        logger.info(f"Fetching project summary for: {project_name}")
        return self._make_request("GET", endpoint, params=params)

    def get_top_categories_today(self) -> dict[str, Any] | None:
        """Retrieves the top categories for today."""
        endpoint = "/get_top_categories_today"
        logger.info("Fetching top categories for today")
        return self._make_request("GET", endpoint, use_insights_url=True)

    def get_top_alerts_today(self) -> dict[str, Any] | None:
        """Retrieves the top alerts for today."""
        endpoint = "/get_top_alerts_today"
        logger.info("Fetching top alerts for today")
        return self._make_request("GET", endpoint, use_insights_url=True)

    def get_category_dominance(self, category_names: list[str], duration: str) -> list[dict[str, Any]] | None:
        """Retrieves the dominance of a category over a specified duration.

        Args:
        ----
            category_names: List of category names to analyze
            duration: Time period to analyze (e.g. "7d", "30d", "90d")

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
            ValueError: If category_names is empty or duration is invalid
            TrendmoonAPIError: If the API request fails

        """
        if not category_names:
            msg = "category_names must not be empty"
            raise ValueError(msg)
        if not duration or not duration.endswith("d"):
            msg = "duration must be a valid time period (e.g. '7d', '30d', '90d')"
            raise ValueError(msg)

        endpoint = "/categories/dominance"
        params = {"category_names": category_names, "duration": duration}
        logger.info(f"Fetching category dominance for: {category_names} over duration: {duration}")
        return self._make_request("GET", endpoint, params=params, use_insights_url=True)

    def get_top_category_alerts(self) -> dict[str, Any] | None:
        """Retrieves the top category alerts filtered by top category for the day."""
        endpoint = "/get_top_category_alerts"
        logger.info("Fetching top category alerts")
        return self._make_request("GET", endpoint, use_insights_url=True)

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
        logger.info(f"Fetching social trends with params: {provided_params}")
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
        logger.info(f"Fetching keyword trends for '{keyword}' with params: {provided_params}")
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
        logger.info(f"Searching messages with params: {params}")
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
        logger.info(f"Fetching messages for user: {username} from {start_date} to {end_date}")
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
        logger.info(f"Fetching messages for chat {identifier} from {start_date} to {end_date}")
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
        logger.info(f"Fetching messages from {start_date} to {end_date}")
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
        logger.info(f"Fetching chat information for group username: {group_username}")
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
        logger.info(f"Fetching chat activity for group {group_username} from {start_date} to {end_date}")
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
        logger.info(f"Searching for coin with params: {provided_params}")
        return self._make_request("GET", endpoint, params=params)

    def get_platforms(self) -> dict[str, Any] | None:
        """Retrieves all available platforms that have coins in the index."""
        endpoint = "/coins/platforms"
        logger.info("Fetching all platforms")
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
        logger.info(f"Fetching coin details for: {provided_params}")
        return self._make_request("GET", endpoint, params=params)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.session.close()


def _validate_iso_date(date_str: str) -> bool:
    """Validate ISO 8601 date string."""
    try:
        parse(date_str, yearfirst=True)
        return True
    except (ParserError, TypeError, ValueError):
        return False
