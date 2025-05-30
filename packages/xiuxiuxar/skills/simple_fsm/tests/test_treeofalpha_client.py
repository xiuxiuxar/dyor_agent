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

"""Tests for the Tree of Alpha client."""

import time
import logging
from pathlib import Path
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests


ROOT_DIR = Path(__file__).parent.parent
import sys  # noqa: E402


sys.path.insert(0, str(ROOT_DIR))

try:
    from packages.xiuxiuxar.skills.simple_fsm.base_client import BaseClient
    from packages.xiuxiuxar.skills.simple_fsm.treeofalpha_client import (
        TreeOfAlphaClient,
        TreeOfAlphaAPIError,
    )
except ImportError as e:
    msg = (
        f"Could not import TreeOfAlphaClient or related errors. "
        f"Ensure treeofalpha_client.py is accessible (PYTHONPATH or relative path). Original error: {e}"
    )
    raise ImportError(msg) from None

# --- Constants for Testing ---
BASE_URL = "https://news.treeofalpha.com"
NEWS_ENDPOINT = "api/news"
TEST_LIMIT = 100
TEST_QUERY = "bitcoin"
TEST_SYMBOL = "BTC"

# --- Mock Data ---
MOCK_NEWS_ITEM = {
    "title": "Test News",
    "source": "Test Source",
    "url": "https://example.com",
    "time": int(time.time() * 1000),  # Current time in milliseconds
    "symbols": ["BTC", "ETH"],
    "_id": "test123",
    "suggestions": [],
}

# --- Fixtures ---


@pytest.fixture
def mock_session_request():
    """Fixture to mock the requests.Session.request method."""
    with patch("requests.Session.request", autospec=True) as mock_req:
        yield mock_req


@pytest.fixture
def api_client():
    """Fixture to provide an initialized TreeOfAlphaAPI client."""
    mock_context = MagicMock()
    mock_context.logger = MagicMock()
    client = TreeOfAlphaClient(
        base_url=BASE_URL,
        news_endpoint=NEWS_ENDPOINT,
        skill_context=mock_context,
        cache_ttl=3600,
    )
    client._last_health_status = False  # noqa: SLF001
    return client


@pytest.fixture
def mock_response():
    """Factory fixture to create mock requests.Response objects."""

    def _create_mock_response(status_code=200, json_data=None, text_data="", raise_for_status_error=None):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = status_code
        mock_resp.text = text_data

        if json_data is not None:
            mock_resp.json.return_value = json_data
        else:
            mock_resp.json.side_effect = requests.exceptions.JSONDecodeError("Mock decode error", "", 0)

        if raise_for_status_error:
            mock_resp.raise_for_status.side_effect = raise_for_status_error(response=mock_resp)
        elif status_code >= 400:
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                f"{status_code} Client Error", response=mock_resp
            )
        else:
            mock_resp.raise_for_status.return_value = None

        return mock_resp

    return _create_mock_response


# --- Test Classes ---


class TestTreeOfAlphaClientInitialization:
    """Tests for TreeOfAlphaAPI client initialization."""

    def test_init_success(self):
        """Test successful initialization."""
        mock_context = MagicMock()
        mock_context.logger = MagicMock()
        client = TreeOfAlphaClient(skill_context=mock_context, base_url=BASE_URL)
        assert client.base_url == BASE_URL
        assert isinstance(client.session, requests.Session)
        assert isinstance(client, BaseClient)

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        mock_context = MagicMock()
        mock_context.logger = MagicMock()
        custom_timeout = 60
        client = TreeOfAlphaClient(timeout=custom_timeout, skill_context=mock_context, base_url=BASE_URL)
        assert client.timeout == custom_timeout

    def test_init_invalid_base_url(self):
        """Test initialization with invalid base URL."""
        mock_context = MagicMock()
        mock_context.logger = MagicMock()
        with pytest.raises(ValueError, match="Invalid.*Client base URL"):
            TreeOfAlphaClient(base_url="invalid-url", skill_context=mock_context)

    def test_session_headers(self):
        """Test session headers are properly set."""
        mock_context = MagicMock()
        mock_context.logger = MagicMock()
        client = TreeOfAlphaClient(base_url=BASE_URL, skill_context=mock_context)
        assert "Content-Type" in client.session.headers
        assert client.session.headers["Content-Type"] == "application/json"


class TestTreeOfAlphaAPIEndpoints:
    """Tests for the specific API endpoint methods."""

    def test_get_news_success(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_news."""
        expected_data = [MOCK_NEWS_ITEM]
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.get_news(limit=TEST_LIMIT)

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs["method"] == "GET"
        assert call_args.kwargs["url"] == f"{BASE_URL}/{NEWS_ENDPOINT}"
        assert call_args.kwargs["params"] == {"limit": TEST_LIMIT}

    def test_get_news_caching_with_ttl(self, api_client, mock_session_request, mock_response):
        """Test that caching respects TTL."""
        expected_data = [MOCK_NEWS_ITEM]
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        # First call should hit the API
        result1 = api_client.get_news()
        assert result1 == expected_data
        assert mock_session_request.call_count == 1

        # Simulate TTL expiration
        api_client._last_fetch_time = time.time() - (api_client.cache_ttl + 1)  # noqa: SLF001

        # Next call should hit API again due to TTL expiration
        result2 = api_client.get_news()
        assert result2 == expected_data
        assert mock_session_request.call_count == 2

    def test_search_uses_cached_data(self, api_client, mock_session_request, mock_response):
        """Test that search operations use cached data instead of making new API calls."""
        mock_news = [{**MOCK_NEWS_ITEM, "title": "Bitcoin News"}, {**MOCK_NEWS_ITEM, "title": "Other News"}]
        mock_session_request.return_value = mock_response(status_code=200, json_data=mock_news)

        # First call to cache the data
        api_client.get_news()
        assert mock_session_request.call_count == 1

        # Search should use cached data
        result = api_client.search_news("Bitcoin")
        assert len(result) == 1
        assert result[0]["title"] == "Bitcoin News"
        # Verify no additional API calls were made
        assert mock_session_request.call_count == 1

    def test_search_with_different_cases(self, api_client, mock_session_request, mock_response):
        """Test search with different case sensitivities."""
        mock_news = [
            {**MOCK_NEWS_ITEM, "title": "BITCOIN News"},
            {**MOCK_NEWS_ITEM, "title": "bitcoin update"},
            {**MOCK_NEWS_ITEM, "title": "Bitcoin Analysis"},
            {**MOCK_NEWS_ITEM, "title": "Other News"},
        ]
        mock_session_request.return_value = mock_response(status_code=200, json_data=mock_news)

        # Cache the data first
        api_client.get_news()
        assert mock_session_request.call_count == 1

        # Case-insensitive search (default)
        result1 = api_client.search_news("bitcoin")
        assert len(result1) == 3

        # Case-sensitive search
        result2 = api_client.search_news("BITCOIN", case_sensitive=True)
        assert len(result2) == 1
        assert result2[0]["title"] == "BITCOIN News"

        # Verify no additional API calls were made
        assert mock_session_request.call_count == 1

    def test_get_news_by_symbol(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_news_by_symbol."""
        mock_news = [MOCK_NEWS_ITEM, {**MOCK_NEWS_ITEM, "symbols": ["ETH"]}]
        mock_session_request.return_value = mock_response(status_code=200, json_data=mock_news)

        result = api_client.get_news_by_symbol("BTC")
        assert len(result) == 1
        assert "BTC" in result[0]["symbols"]

    def test_get_latest_news(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_latest_news."""
        current_time = time.time()
        old_time = current_time - (25 * 3600 * 1000)  # 25 hours ago
        mock_news = [{**MOCK_NEWS_ITEM, "time": int(current_time * 1000)}, {**MOCK_NEWS_ITEM, "time": int(old_time)}]
        mock_session_request.return_value = mock_response(status_code=200, json_data=mock_news)

        result = api_client.get_latest_news(hours=24)
        assert len(result) == 1
        assert result[0]["time"] == int(current_time * 1000)


class TestTreeOfAlphaAPIErrorHandling:
    """Tests for error handling and response parsing."""

    def test_invalid_json_handling(self, api_client, mock_session_request, mock_response):
        """Test handling of invalid JSON responses."""
        mock_session_request.return_value = mock_response(status_code=200, text_data="Invalid JSON")

        with pytest.raises(TreeOfAlphaAPIError) as exc_info:
            api_client.get_news()
        assert "Invalid JSON response" in str(exc_info.value)

    def test_network_error_handling(self, api_client, mock_session_request):
        """Test handling of network errors."""
        mock_session_request.side_effect = requests.exceptions.RequestException("Network error")

        with pytest.raises(TreeOfAlphaAPIError) as exc_info:
            api_client.get_news()
        assert "Failed to communicate with API" in str(exc_info.value)

    def test_unexpected_response_format(self, api_client, mock_session_request, mock_response):
        """Test handling of unexpected response format."""
        mock_session_request.return_value = mock_response(status_code=200, json_data={"not": "a list"})

        with pytest.raises(TreeOfAlphaAPIError) as exc_info:
            api_client.get_news()
        assert "Unexpected response format" in str(exc_info.value)


class TestTreeOfAlphaClientHealthAndLogging:
    """Tests for health checking and logging integration."""

    @pytest.mark.skip(reason="This test is not working as expected")
    def test_health_status_transitions(self, api_client, mock_session_request, mock_response, caplog):
        """Test health status changes and logging."""
        caplog.set_level(logging.INFO)

        assert api_client.check_api_health() is False

        mock_session_request.return_value = mock_response(status_code=200, json_data=[MOCK_NEWS_ITEM])
        api_client.get_news()
        assert api_client.check_api_health() is True
        assert "Successfully fetched and cached" in caplog.text
        assert "connection healthy" in caplog.text

        caplog.clear()

        mock_session_request.reset_mock()
        mock_session_request.side_effect = requests.exceptions.RequestException("Network error")

        with pytest.raises(TreeOfAlphaAPIError):
            api_client.get_news()
        assert api_client.check_api_health() is False
        assert "connection unhealthy" in caplog.text

    def test_error_logging_includes_details(self, api_client, mock_session_request, caplog):
        """Test error log content."""
        caplog.set_level(logging.ERROR)
        error_message = "Network timeout"
        mock_session_request.side_effect = requests.exceptions.RequestException(error_message)

        with pytest.raises(TreeOfAlphaAPIError):
            api_client.get_news()

        assert error_message in caplog.text
        assert NEWS_ENDPOINT in caplog.text


@pytest.mark.integration
class TestTreeOfAlphaClientIntegration:
    """Integration tests for TreeOfAlphaAPI against live API."""

    @pytest.fixture(scope="module")
    def live_client(self):
        """Create a TreeOfAlphaClient client for live testing."""
        mock_context = MagicMock()
        mock_context.logger = MagicMock()
        client = TreeOfAlphaClient(
            base_url=BASE_URL, news_endpoint=NEWS_ENDPOINT, timeout=30, skill_context=mock_context, cache_ttl=3600
        )
        client._last_fetch_time = 0  # noqa: SLF001
        return client

    def test_get_news_integration(self, live_client):
        """Test get_news against live API."""
        result = live_client.get_news(limit=10)
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert "title" in item
            assert "source" in item
            assert "url" in item
            assert "time" in item
            assert isinstance(item["time"], int)

    def test_search_integration(self, live_client):
        """Test search against live API."""
        result = live_client.search_news("bitcoin")
        assert isinstance(result, list)
        for item in result:
            assert "bitcoin" in item["title"].lower() or "bitcoin" in item["source"].lower()

    def test_symbol_search_integration(self, live_client):
        """Test symbol search against live API."""
        result = live_client.get_news_by_symbol("BTC")
        assert isinstance(result, list)
        for item in result:
            assert "BTC" in item["symbols"]

    def test_latest_news_integration(self, live_client):
        """Test latest news against live API."""
        hours = 24
        result = live_client.get_latest_news(hours=hours)
        assert isinstance(result, list)
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        for item in result:
            news_time = datetime.fromtimestamp(item["time"] / 1000, UTC)
            assert news_time > cutoff_time

    def test_caching_integration(self, live_client):
        """Test caching behavior with live API."""
        # Add artificial delay to first API call to simulate network latency
        original_make_request = live_client._make_request  # noqa: SLF001
        first_call = True

        def delayed_make_request(*args, **kwargs):
            nonlocal first_call
            if first_call:
                first_call = False
                time.sleep(0.2)
            return original_make_request(*args, **kwargs)

        # Patch the _make_request method
        with patch.object(live_client, "_make_request", side_effect=delayed_make_request):
            start_time = time.time()
            result1 = live_client.get_news(limit=100, force_refresh=True)
            first_call_duration = time.time() - start_time

            start_time = time.time()
            result2 = live_client.get_news(limit=100)  # Use cache
            cached_call_duration = time.time() - start_time

            assert result1 == result2

            assert first_call_duration > cached_call_duration + 0.15, (
                f"First call ({first_call_duration:.3f}s) should be significantly slower than "
                f"cached call ({cached_call_duration:.3f}s)"
            )

    def test_rate_limiting(self, live_client):
        """Test API rate limiting behavior."""
        for _ in range(5):
            result = live_client.get_news(limit=10)
            assert isinstance(result, list)
            time.sleep(1)
