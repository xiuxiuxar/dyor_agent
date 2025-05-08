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

"""Tests for the LookOnChain client."""

import logging
from pathlib import Path
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
import requests


ROOT_DIR = Path(__file__).parent.parent
import sys  # noqa: E402


sys.path.insert(0, str(ROOT_DIR))

try:
    from packages.xiuxiuxar.skills.simple_fsm.base_client import BaseClient
    from packages.xiuxiuxar.skills.simple_fsm.lookonchain_client import (
        NetworkError,
        ParsingError,
        ScrapedDataItem,
        LookOnChainClient,
        LookOnChainAPIError,
    )
except ImportError as e:
    msg = (
        f"Could not import LookOnChainClient or related errors. "
        f"Ensure lookonchain_client.py is accessible (PYTHONPATH or relative path). Original error: {e}"
    )
    raise ImportError(msg) from None

# --- Constants for Testing ---
BASE_DOMAIN = "https://www.lookonchain.com"
SEARCH_ENDPOINT = f"{BASE_DOMAIN}/ashx/search_list.ashx"
TEST_QUERY = "ethereum"
TEST_PAGE = 1
TEST_COUNT = 20

# --- Fixtures ---


@pytest.fixture
def mock_session_request():
    """Fixture to mock the requests.Session.request method."""
    with patch("requests.Session.request", autospec=True) as mock_req:
        yield mock_req


@pytest.fixture
def api_client():
    """Fixture to provide an initialized LookOnChainClient."""
    client = LookOnChainClient(
        base_url=BASE_DOMAIN,
        search_endpoint=SEARCH_ENDPOINT,
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


@pytest.fixture(scope="module")
def live_client():
    """Create a LookOnChainClient for live testing."""
    return LookOnChainClient()


# --- Test Classes ---


class TestLookOnChainClientInitialization:
    """Tests for LookOnChainClient initialization."""

    def test_init_success(self):
        """Test successful initialization."""
        client = LookOnChainClient()
        assert client.source_name == "lookonchain"
        assert client.base_url == BASE_DOMAIN
        assert isinstance(client.session, requests.Session)
        assert isinstance(client, BaseClient)

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        custom_timeout = 30
        client = LookOnChainClient(timeout=custom_timeout)
        assert client.timeout == custom_timeout

    def test_init_invalid_search_endpoint(self):
        """Test initialization with invalid search endpoint."""
        with pytest.raises(ValueError, match="Invalid LookOnChain search endpoint URL"):
            LookOnChainClient(search_endpoint="invalid-url")

    def test_session_headers(self):
        """Test session headers are properly set."""
        client = LookOnChainClient()
        assert "User-Agent" in client.session.headers
        assert isinstance(client.session.headers["User-Agent"], str)


class TestLookOnChainAPIEndpoints:
    """Tests for the specific API endpoint methods."""

    def test_search_success(self, api_client, mock_session_request, mock_response):
        """Test successful search call."""
        expected_data = {
            "success": "Y",
            "content": [
                {
                    "id": "123",
                    "stitle": "Test Article",
                    "sabstract": "Test content",
                    "dcreate_time": datetime.now(UTC).isoformat(),
                    "sauthor_name": "Test Author",
                    "stype": "news",
                    "spic": "https://example.com/image.jpg",
                }
            ],
        }
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.search(TEST_QUERY)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ScrapedDataItem)
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert "params" in call_args.kwargs
        assert call_args.kwargs["params"]["keyword"] == TEST_QUERY

    def test_search_with_pagination(self, api_client, mock_session_request, mock_response):
        """Test search with pagination parameters."""
        mock_session_request.return_value = mock_response(status_code=200, json_data={"success": "Y", "content": []})

        api_client.search(TEST_QUERY, page=2, count=30)

        call_args = mock_session_request.call_args
        assert call_args.kwargs["params"]["page"] == 2
        assert call_args.kwargs["params"]["count"] == 30

    def test_search_failed_response(self, api_client, mock_session_request, mock_response):
        """Test handling of failed API response."""
        mock_session_request.return_value = mock_response(
            status_code=200, json_data={"success": "N", "message": "Error"}
        )

        result = api_client.search(TEST_QUERY)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_missing_required_fields(self, api_client, mock_session_request, mock_response):
        """Test handling of responses missing required fields."""
        mock_session_request.return_value = mock_response(
            status_code=200, json_data={"success": "Y", "content": [{"invalid": "data"}]}
        )

        result = api_client.search(TEST_QUERY)
        assert isinstance(result, list)
        assert len(result) == 0


class TestLookOnChainClientErrorHandling:
    """Tests for error handling and response parsing."""

    def test_network_error_handling(self, api_client, mock_session_request):
        """Test handling of network errors."""
        mock_session_request.side_effect = requests.exceptions.RequestException("Network error")

        with pytest.raises(NetworkError) as exc_info:
            api_client.search(TEST_QUERY)
        assert "Failed to fetch data" in str(exc_info.value)

    def test_invalid_json_handling(self, api_client, mock_session_request, mock_response):
        """Test handling of invalid JSON responses."""
        mock_session_request.return_value = mock_response(status_code=200, text_data="Invalid JSON")

        with pytest.raises(ParsingError) as exc_info:
            api_client.search(TEST_QUERY)
        assert "Failed to parse data from lookonchain" in str(exc_info.value)

    def test_api_error_conversion(self, api_client, mock_session_request):
        """Test conversion of LookOnChainAPIError to NetworkError."""
        mock_session_request.side_effect = LookOnChainAPIError("API Error", 500)

        with pytest.raises(NetworkError) as exc_info:
            api_client.search(TEST_QUERY)
        assert "API Error" in str(exc_info.value)


class TestLookOnChainClientHealthAndLogging:
    """Tests for health checking and logging integration."""

    def test_health_status_transitions(self, api_client, mock_session_request, mock_response, caplog):
        """Test health status changes and logging."""
        caplog.set_level(logging.INFO)

        # Start with unhealthy status to test transition to healthy
        api_client._last_health_status = False  # noqa: SLF001

        # Test transition to healthy
        mock_session_request.return_value = mock_response(status_code=200, json_data={"success": "Y", "content": []})
        api_client.search(TEST_QUERY)
        assert "connection healthy" in caplog.text
        assert api_client._last_health_status is True  # noqa: SLF001

        caplog.clear()

        mock_session_request.side_effect = requests.exceptions.RequestException("Network error")
        with pytest.raises(NetworkError):
            api_client.search(TEST_QUERY)
        assert "connection unhealthy" in caplog.text
        assert api_client._last_health_status is False  # noqa: SLF001

    def test_error_logging_includes_details(self, api_client, mock_session_request, caplog):
        """Test error log content."""
        caplog.set_level(logging.ERROR)
        error_message = "Invalid response"
        mock_session_request.side_effect = requests.exceptions.RequestException(error_message)

        with pytest.raises(NetworkError):
            api_client.search(TEST_QUERY)

        assert error_message in caplog.text
        assert SEARCH_ENDPOINT in caplog.text


@pytest.mark.integration
class TestLookOnChainClientIntegration:
    """Integration tests for LookOnChainClient against live API."""

    def test_search_integration(self, live_client):
        """Test search against live API."""
        result = live_client.search(TEST_QUERY)
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, ScrapedDataItem)
            assert item.title
            assert item.url.startswith(BASE_DOMAIN)
            assert item.source == "lookonchain"
            assert item.scraped_at
            assert isinstance(item.to_dict(), dict)

    def test_pagination_integration(self, live_client):
        """Test pagination against live API."""
        page_1 = live_client.search(TEST_QUERY, page=1, count=5)
        page_2 = live_client.search(TEST_QUERY, page=2, count=5)

        assert len(page_1) == 5
        assert len(page_2) == 5
        assert page_1 != page_2

    def test_rate_limiting(self, live_client):
        """Test API rate limiting behavior."""
        for _ in range(5):
            result = live_client.search(TEST_QUERY, count=1)
            assert isinstance(result, list)
