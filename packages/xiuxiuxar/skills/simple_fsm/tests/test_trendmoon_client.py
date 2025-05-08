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

"""Test Trendmoon client."""

import os
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
    from packages.xiuxiuxar.skills.simple_fsm.trendmoon_client import (
        MatchModes,
        TimeIntervals,
        TrendmoonClient,
        TrendmoonAPIError,
        SocialTrendIntervals,
    )
except ImportError as e:
    msg = (
        f"Could not import TrendmoonClient or TrendmoonClientError. "
        f"Ensure trendmoon_client.py is accessible (PYTHONPATH or relative path). Original error: {e}"
    )
    raise ImportError(msg) from None


# --- Constants for Testing ---
FAKE_API_KEY = "test-api-key-123"
BASE_URL = "https://mock.trendmoon.test/taimat"
INSIGHTS_URL = "https://mock.trendmoon.test/insights"
PROJECT_NAME = "testcoin"
SYMBOL = "TEST"
PERIOD = "1d"
CONTRACT_ADDRESS = "0x123..."
COIN_ID = "test-coin-123"
KEYWORD = "bitcoin moon"
GROUP_USERNAME = "test_group"
START_DATE = "2024-01-01T00:00:00"
END_DATE = "2024-01-02T00:00:00"

# Integration test constants
TEST_PROJECT = "taraxa"
TEST_SYMBOL = "BTC"
TEST_CONTRACT = "0xb8c77482e45f1f44de1745f52c74426c631bdd52"
TEST_COIN_ID = "taraxa"
TEST_KEYWORD = "bitcoin halving"
TEST_GROUP = "bitcoin_signals"

# --- Fixtures ---


@pytest.fixture
def mock_session_request():
    """Fixture to mock the requests.Session.request method."""
    with patch("requests.Session.request", autospec=True) as mock_req:
        yield mock_req


@pytest.fixture
def api_client():
    """Fixture to provide an initialized TrendmoonClient with a fake key."""
    return TrendmoonClient(
        api_key=FAKE_API_KEY,
        base_url=BASE_URL,
        insights_url=INSIGHTS_URL,
    )


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
def staging_client():
    """Create a TrendmoonClient configured for staging environment."""
    api_key = os.getenv("TRENDMOON_STAGING_API_KEY")
    base_url = os.getenv("TRENDMOON_STAGING_URL")
    if not api_key or not base_url:
        pytest.skip("Integration test environment variables not set")
    client = TrendmoonClient(api_key=api_key, base_url=base_url)
    assert client.check_api_health(), "Staging API is not healthy"
    return client


# --- Test Classes ---


class TestTimeIntervals:
    """Tests for TimeIntervals class."""

    def test_valid_intervals(self):
        """Test that valid_intervals returns all valid time intervals."""
        intervals = TimeIntervals.valid_intervals()
        assert isinstance(intervals, set)
        assert len(intervals) == 6
        assert TimeIntervals.ONE_HOUR in intervals
        assert TimeIntervals.FOUR_HOURS in intervals
        assert TimeIntervals.TWELVE_HOURS in intervals
        assert TimeIntervals.ONE_DAY in intervals
        assert TimeIntervals.THREE_DAYS in intervals
        assert TimeIntervals.ONE_WEEK in intervals


class TestMatchModes:
    """Tests for MatchModes class."""

    def test_valid_modes(self):
        """Test that valid_modes returns all valid match modes."""
        modes = MatchModes.valid_modes()
        assert isinstance(modes, set)
        assert len(modes) == 5
        assert MatchModes.EXACT in modes
        assert MatchModes.ANY in modes
        assert MatchModes.ALL in modes
        assert MatchModes.FUZZY in modes
        assert MatchModes.PARTIAL in modes


class TestTrendmoonClientInitialization:
    """Tests for TrendmoonClient initialization."""

    def test_init_success_with_env_var(self, monkeypatch):
        """Test successful initialization using environment variables."""
        monkeypatch.setenv("TRENDMOON_API_KEY", FAKE_API_KEY)
        monkeypatch.setenv("TRENDMOON_BASE_URL", BASE_URL)
        monkeypatch.setenv("TRENDMOON_INSIGHTS_URL", INSIGHTS_URL)
        client = TrendmoonClient(api_key=FAKE_API_KEY, base_url=BASE_URL, insights_url=INSIGHTS_URL)
        assert client.session.headers["Api-key"] == FAKE_API_KEY
        assert client.base_url == BASE_URL
        assert client.insights_url == INSIGHTS_URL

    def test_init_success_with_argument(self, monkeypatch):
        """Test successful initialization passing key as argument."""
        monkeypatch.delenv("TRENDMOON_API_KEY", raising=False)
        monkeypatch.setenv("TRENDMOON_BASE_URL", BASE_URL)
        monkeypatch.setenv("TRENDMOON_INSIGHTS_URL", INSIGHTS_URL)
        client = TrendmoonClient(api_key=FAKE_API_KEY, base_url=BASE_URL, insights_url=INSIGHTS_URL)
        assert client.session.headers["Api-key"] == FAKE_API_KEY

    def test_init_no_api_key_raises_value_error(self, monkeypatch):
        """Test ValueError is raised if no API key is found."""
        monkeypatch.delenv("TRENDMOON_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key is required"):
            TrendmoonClient(base_url=BASE_URL, insights_url=INSIGHTS_URL, api_key=None)

    def test_init_invalid_base_url_raises_value_error(self, monkeypatch):
        """Test ValueError is raised for invalid base URL."""
        monkeypatch.setenv("TRENDMOON_API_KEY", FAKE_API_KEY)
        with pytest.raises(ValueError, match="Invalid.*Client base URL"):
            TrendmoonClient(api_key=FAKE_API_KEY, base_url="invalid-url", insights_url=INSIGHTS_URL)
        with pytest.raises(ValueError, match="Invalid.*Client base URL"):
            TrendmoonClient(api_key=FAKE_API_KEY, base_url="", insights_url=INSIGHTS_URL)
        with pytest.raises(ValueError, match="Invalid.*Client base URL"):
            TrendmoonClient(api_key=FAKE_API_KEY, base_url="ftp://insecure.com", insights_url=INSIGHTS_URL)

    def test_init_invalid_insights_url_raises_value_error(self, monkeypatch):
        """Test ValueError is raised for invalid insights URL."""
        monkeypatch.setenv("TRENDMOON_API_KEY", FAKE_API_KEY)
        monkeypatch.setenv("TRENDMOON_BASE_URL", BASE_URL)
        with pytest.raises(ValueError, match="Invalid Trendmoon Insights API base URL"):
            TrendmoonClient(api_key=FAKE_API_KEY, base_url=BASE_URL, insights_url="invalid-url")
        with pytest.raises(ValueError, match="Invalid Trendmoon Insights API base URL"):
            TrendmoonClient(api_key=FAKE_API_KEY, base_url=BASE_URL, insights_url="")
        with pytest.raises(ValueError, match="Invalid Trendmoon Insights API base URL"):
            TrendmoonClient(api_key=FAKE_API_KEY, base_url=BASE_URL, insights_url="http://insecure.com")

    def test_inheritance(self):
        """Test that TrendmoonClient inherits from BaseClient."""
        client = TrendmoonClient(api_key=FAKE_API_KEY, base_url=BASE_URL, insights_url=INSIGHTS_URL)
        assert isinstance(client, BaseClient)


class TestTrendmoonAPIEndpoints:
    """Tests for the specific API endpoint methods."""

    def test_get_project_summary_success(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_project_summary."""
        expected_data = {"summary": "data", "project": PROJECT_NAME}
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.get_project_summary(PROJECT_NAME)

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs == {
            "method": "GET",
            "url": f"{BASE_URL}/social/project-summary",
            "params": {"project_name": PROJECT_NAME},
            "json": None,
            "timeout": api_client.timeout,
        }

    def test_get_top_categories_today_success(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_top_categories_today."""
        expected_data = {"categories": ["DeFi", "NFT", "Gaming"]}
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.get_top_categories_today()

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs["method"] == "GET"
        assert call_args.kwargs["url"] == f"{INSIGHTS_URL}/get_top_categories_today"
        assert call_args.kwargs["params"] is None
        assert call_args.kwargs["timeout"] == api_client.timeout

    def test_get_top_alerts_today_success(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_top_alerts_today."""
        expected_data = {"alerts": [{"id": 1, "message": "test alert"}]}
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.get_top_alerts_today()

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs["method"] == "GET"
        assert call_args.kwargs["url"] == f"{INSIGHTS_URL}/get_top_alerts_today"
        assert call_args.kwargs["params"] is None
        assert call_args.kwargs["timeout"] == api_client.timeout

    def test_get_social_trend_success(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_social_trend with multiple parameters."""
        expected_data = {"sentiment_score": 0.75, "mention_volume": 500}
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.get_social_trend(
            contract_address=CONTRACT_ADDRESS,
            symbol=SYMBOL,
            project_name=PROJECT_NAME,
            coin_id=COIN_ID,
            date_interval=7,
            time_interval=SocialTrendIntervals.ONE_DAY,
        )

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs == {
            "method": "GET",
            "url": f"{BASE_URL}/social/trend",
            "params": {
                "contract_address": CONTRACT_ADDRESS,
                "symbol": SYMBOL,
                "project_name": PROJECT_NAME,
                "coin_id": COIN_ID,
                "date_interval": 7,
                "time_interval": SocialTrendIntervals.ONE_DAY,
            },
            "json": None,
            "timeout": api_client.timeout,
        }

    def test_get_keyword_trend_success(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_keyword_trend."""
        expected_data = {"trend_data": {"volumes": [100, 200], "dates": ["2024-01-01", "2024-01-02"]}}
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.get_keyword_trend(
            keyword=KEYWORD, duration=7, time_interval=TimeIntervals.ONE_DAY, match_mode=MatchModes.EXACT
        )

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs == {
            "method": "GET",
            "url": f"{BASE_URL}/social/keyword",
            "params": {
                "keyword": KEYWORD,
                "duration": 7,
                "time_interval": TimeIntervals.ONE_DAY,
                "match_mode": MatchModes.EXACT,
            },
            "json": None,
            "timeout": api_client.timeout,
        }

    def test_get_keyword_trend_invalid_interval(self, api_client):
        """Test get_keyword_trend with invalid time interval."""
        with pytest.raises(ValueError, match="Invalid time_interval"):
            api_client.get_keyword_trend(KEYWORD, time_interval="invalid")

    def test_get_keyword_trend_invalid_match_mode(self, api_client):
        """Test get_keyword_trend with invalid match mode."""
        with pytest.raises(ValueError, match="Invalid match_mode"):
            api_client.get_keyword_trend(KEYWORD, match_mode="invalid")

    def test_search_messages_success(self, api_client, mock_session_request, mock_response):
        """Test successful call to search_messages with multiple parameters."""
        expected_data = {"messages": [{"id": 1, "text": "test message"}]}
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.search_messages(
            text="test",
            group_username=GROUP_USERNAME,
            message_type="clean",
            start_date=START_DATE,
            end_date=END_DATE,
            from_=0,
            size=100,
        )

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs == {
            "method": "GET",
            "url": f"{BASE_URL}/messages/search",
            "params": {
                "text": "test",
                "group_username": GROUP_USERNAME,
                "message_type": "clean",
                "start_date": START_DATE,
                "end_date": END_DATE,
                "from": 0,
                "size": 100,
            },
            "json": None,
            "timeout": api_client.timeout,
        }

    def test_get_chat_messages_no_identifier(self, api_client):
        """Test get_chat_messages with no group_username or chat_id."""
        with pytest.raises(ValueError, match="Either group_username or chat_id must be provided"):
            api_client.get_chat_messages(START_DATE, END_DATE)

    def test_get_coin_details_no_identifier(self, api_client):
        """Test get_coin_details with no identifier parameters."""
        with pytest.raises(ValueError, match="At least one parameter .* must be provided"):
            api_client.get_coin_details()

    def test_get_category_dominance_success(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_category_dominance."""
        expected_data = {"dominance": {"Infrastructure": 0.4}}
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.get_category_dominance(category_name="Infrastructure", duration=7)

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs == {
            "method": "GET",
            "url": f"{BASE_URL}/categories/dominance",
            "params": {"category_name": "Infrastructure", "duration": 7},
            "json": None,
            "timeout": api_client.timeout,
        }

    def test_get_top_category_alerts_success(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_top_category_alerts."""
        expected_data = {"alerts": [{"id": 1, "category": "DeFi", "message": "test alert"}]}
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.get_top_category_alerts()

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs["method"] == "GET"
        assert call_args.kwargs["url"] == f"{INSIGHTS_URL}/get_top_category_alerts"
        assert call_args.kwargs["params"] is None
        assert call_args.kwargs["timeout"] == api_client.timeout

    def test_get_chat_information_by_group_username(self, api_client, mock_session_request, mock_response):
        """Test successful call to get_chat_information_by_group_username."""
        expected_data = {"chat_info": {"username": GROUP_USERNAME, "member_count": 1000}}
        mock_session_request.return_value = mock_response(status_code=200, json_data=expected_data)

        result = api_client.get_chat_information_by_group_username(GROUP_USERNAME)

        assert result == expected_data
        mock_session_request.assert_called_once()
        call_args = mock_session_request.call_args
        assert call_args.kwargs == {
            "method": "GET",
            "url": f"{BASE_URL}/chats/{GROUP_USERNAME}",
            "params": None,
            "json": None,
            "timeout": api_client.timeout,
        }


class TestTrendmoonClientHealthAndLogging:
    """Tests for health checking and logging integration."""

    def test_health_status_transitions(self, api_client, mock_session_request, mock_response, caplog):
        """Verify health status changes correctly after success and failure."""
        caplog.set_level(logging.INFO)

        assert api_client.check_api_health() is True

        mock_session_request.return_value = mock_response(
            status_code=503, raise_for_status_error=requests.exceptions.HTTPError
        )
        with pytest.raises(TrendmoonAPIError):
            api_client.search_coin("FAIL")
        assert api_client.check_api_health() is False
        assert "connection unhealthy" in caplog.text
        assert "HTTP Error 503" in caplog.text

        caplog.clear()
        mock_session_request.return_value = mock_response(status_code=200, json_data={"ok": True})
        api_client.search_coin("RECOVER")
        assert api_client.check_api_health() is True
        assert "connection healthy" in caplog.text

    def test_error_logging_includes_details(self, api_client, mock_session_request, mock_response, caplog):
        """Verify specific error logs are generated with context."""
        caplog.set_level(logging.ERROR)
        status_code = 401
        response_text = '{"error": "Invalid API Key"}'
        mock_session_request.return_value = mock_response(
            status_code=status_code, text_data=response_text, raise_for_status_error=requests.exceptions.HTTPError
        )

        with pytest.raises(TrendmoonAPIError):
            api_client.get_project_summary("unauthorized_project")

        assert f"HTTP {status_code} error" in caplog.text
        assert f"{BASE_URL}/social/project-summary" in caplog.text
        assert response_text in caplog.text

    def test_info_logging_on_call(self, api_client, mock_session_request, mock_response, caplog):
        """Verify INFO level logs are generated for method calls."""
        caplog.set_level(logging.INFO)
        mock_session_request.return_value = mock_response(status_code=200, json_data={})

        api_client.get_social_trend(CONTRACT_ADDRESS, SYMBOL)

        assert (
            f"Fetching social trends with params: {{'contract_address': '{CONTRACT_ADDRESS}', 'symbol': '{SYMBOL}'}}"
            in caplog.text
        )


@pytest.mark.integration
class TestTrendmoonClientIntegration:
    """Integration tests for TrendmoonClient against staging environment."""

    @pytest.fixture(scope="class")
    def staging_client(self):
        """Create a TrendmoonClient configured for staging environment."""
        api_key = os.getenv("TRENDMOON_STAGING_API_KEY")
        base_url = os.getenv("TRENDMOON_STAGING_URL")
        if not api_key or not base_url:
            pytest.skip("Integration test environment variables not set")
        return TrendmoonClient(
            api_key=api_key,
            base_url=base_url,
            timeout=30,
            max_retries=5,
        )

    def test_get_project_summary(self, staging_client):
        """Test fetching project summary from staging."""
        result = staging_client.get_project_summary(TEST_PROJECT)
        assert result is not None
        assert isinstance(result, dict)
        assert "coin_id" in result
        assert result["coin_id"] == TEST_COIN_ID

    def test_get_top_categories_today(self, staging_client):
        """Test fetching top categories from staging."""
        result = staging_client.get_top_categories_today()
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        for category in result:
            assert isinstance(category, dict)
            assert "category_name" in category
            assert "category_dominance" in category
            assert "category_market_cap" in category
            assert "dominance_pct_change" in category

    def test_get_social_trend(self, staging_client):
        """Test fetching social trends from staging."""
        result = staging_client.get_social_trend(
            symbol=TEST_SYMBOL, project_name=TEST_PROJECT, date_interval=7, time_interval=TimeIntervals.ONE_DAY
        )
        assert result is not None
        assert isinstance(result, dict)
        assert "coin_id" in result or "trend_market_data" in result

    def test_get_keyword_trend(self, staging_client):
        """Test fetching keyword trends from staging."""
        result = staging_client.get_keyword_trend(
            keyword=TEST_KEYWORD, duration=7, time_interval=TimeIntervals.ONE_DAY, match_mode=MatchModes.EXACT
        )
        assert result is not None
        assert isinstance(result, dict)
        assert "data" in result
        for item in result["data"]:
            assert "date" in item
            assert "count" in item

    def test_search_messages(self, staging_client):
        """Test searching messages from staging."""
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=1)
        result = staging_client.search_messages(
            text=TEST_KEYWORD, start_date=start_date.isoformat(), end_date=end_date.isoformat(), size=10
        )
        assert result is not None
        assert isinstance(result, list)
        assert "text" in result[0]
        assert "group_username" in result[0]
        assert "message_type" in result[0]
        assert "date" in result[0]

    def test_get_category_dominance(self, staging_client):
        """Test fetching category dominance from staging."""
        result = staging_client.get_category_dominance(category_name="Infrastructure", duration=7)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        for category in result:
            assert "date" in category
            assert "category_name" in category
            assert "category_dominance" in category
            assert "category_market_cap" in category
            assert "dominance_pct_change" in category
            assert "market_cap_pct_change" in category

    def test_rate_limiting(self, staging_client):
        """Test API rate limiting behavior."""
        for _ in range(5):
            result = staging_client.get_top_categories_today()
            assert result is not None

        assert staging_client.check_api_health()

    @pytest.mark.parametrize("time_interval", [SocialTrendIntervals.ONE_HOUR, SocialTrendIntervals.ONE_DAY])
    def test_social_trend_intervals(self, staging_client, time_interval):
        """Test social trends with different time intervals."""
        result = staging_client.get_social_trend(symbol=TEST_SYMBOL, time_interval=time_interval, date_interval=3)
        assert result is not None
        assert isinstance(result, dict)
        if "message" in result:
            assert "No trend data found" in result["message"]
        else:
            assert "coin_id" in result or "trend_market_data" in result

    @pytest.mark.parametrize("match_mode", [MatchModes.EXACT, MatchModes.FUZZY, MatchModes.PARTIAL])
    def test_keyword_trend_match_modes(self, staging_client, match_mode):
        """Test keyword trends with different match modes."""
        result = staging_client.get_keyword_trend(
            keyword=TEST_KEYWORD, duration=3, time_interval=TimeIntervals.ONE_DAY, match_mode=match_mode
        )
        assert result is not None
        assert isinstance(result, dict)


@pytest.mark.integration
class TestTrendmoonClientErrorHandling:
    """Integration tests for error handling against staging environment."""

    def test_invalid_project(self, staging_client):
        """Test behavior with invalid project name."""
        with pytest.raises(TrendmoonAPIError):
            staging_client.get_project_summary("nonexistent_project_123456789")

    def test_invalid_date_range(self, staging_client):
        """Test behavior with invalid date range."""
        future_date = datetime.now(UTC) + timedelta(days=7)
        result = staging_client.search_messages(
            text=TEST_KEYWORD, start_date=future_date.isoformat(), end_date=future_date.isoformat()
        )
        assert result == []

    def test_invalid_category(self, staging_client):
        """Test behavior with invalid category name."""
        result = staging_client.get_category_dominance(category_name="InvalidCategory123", duration=7)
        assert result == []


class TestBaseClientIntegration:
    """Tests for BaseClient integration with TrendmoonClient."""

    def test_base_client_functionality(self):
        """Test that base client functionality works through TrendmoonClient."""
        client = TrendmoonClient(
            api_key=FAKE_API_KEY,
            base_url=BASE_URL,
            insights_url=INSIGHTS_URL,
        )
        assert isinstance(client, BaseClient)
        assert client.base_url == BASE_URL
        assert client.session.headers["Api-key"] == FAKE_API_KEY
        assert "Content-Type" in client.session.headers
