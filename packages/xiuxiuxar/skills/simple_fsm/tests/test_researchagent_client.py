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

"""Integration tests for the Research Agent client."""

import os
from pathlib import Path
from datetime import UTC, datetime, timedelta

import pytest


ROOT_DIR = Path(__file__).parent.parent
import sys  # noqa: E402


sys.path.insert(0, str(ROOT_DIR))

try:
    from packages.xiuxiuxar.skills.simple_fsm.researchagent_client import ResearchAgentClient, ResearchAgentAPIError
except ImportError as e:
    msg = (
        f"Could not import ResearchAgentClient or ResearchAgentAPIError. "
        f"Ensure researchagent_client.py is accessible (PYTHONPATH or relative path). Original error: {e}"
    )
    raise ImportError(msg) from None


# --- Constants for Testing ---
TEST_ACCOUNT = "spotonchain"
TEST_FILTER = "ethereum"
TEST_LIMIT = 5


@pytest.fixture(scope="module")
def live_client():
    """Create a ResearchAgentClient for live testing."""
    api_key = os.getenv("RESEARCH_AGENT_API_KEY")
    base_url = os.getenv("RESEARCH_AGENT_BASE_URL", "https://docker.trendmoon.io/v1")
    if not api_key:
        pytest.skip("Integration test environment variables not set")
    return ResearchAgentClient(api_key=api_key, base_url=base_url)


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping in CI")
@pytest.mark.integration
class TestResearchAgentAPIIntegration:
    """Integration tests for ResearchAgentClient against live environment."""

    def test_get_tweets_multi_account(self, live_client):
        """Test fetching tweets from multiple accounts."""
        since = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        result = live_client.get_tweets_multi_account(limit=TEST_LIMIT, since=since)

        assert result is not None
        assert isinstance(result, dict)
        assert "data" in result
        assert "tweets" in result["data"]
        assert isinstance(result["data"]["tweets"], list)
        # Each account can return up to TEST_LIMIT tweets
        num_accounts = 4  # spotonchain, lookonchain, aixbt_agent, cryptounfolded
        assert len(result["data"]["tweets"]) <= TEST_LIMIT * num_accounts

        for tweet in result["data"]["tweets"]:
            assert "id" in tweet
            assert "html" in tweet
            assert "timestamp" in tweet
            assert "account" in tweet

    def test_get_tweets_research(self, live_client):
        """Test fetching research tweets."""
        result = live_client.get_tweets_research(limit=TEST_LIMIT)

        assert result is not None
        assert isinstance(result, dict)
        assert "data" in result
        assert "tweets" in result["data"]
        assert isinstance(result["data"]["tweets"], list)
        assert len(result["data"]["tweets"]) <= TEST_LIMIT

        for tweet in result["data"]["tweets"]:
            assert "id" in tweet
            assert "html" in tweet
            assert "timestamp" in tweet

    def test_get_tweets_research_with_summary(self, live_client):
        """Test fetching research tweets with summarization."""
        result = live_client.get_tweets_research(limit=TEST_LIMIT, summarize=True)

        assert result is not None
        assert isinstance(result, dict)
        assert "data" in result
        assert "tweets" in result["data"]
        assert "summary" in result["data"]

    def test_get_tweets_filter(self, live_client):
        """Test fetching filtered tweets."""
        result = live_client.get_tweets_filter(account=TEST_ACCOUNT, filter=TEST_FILTER, limit=TEST_LIMIT)

        assert result is not None
        assert isinstance(result, dict)
        assert "data" in result
        assert "tweets" in result["data"]
        assert isinstance(result["data"]["tweets"], list)
        assert len(result["data"]["tweets"]) <= TEST_LIMIT

        for tweet in result["data"]["tweets"]:
            assert "id" in tweet
            assert "html" in tweet
            assert "timestamp" in tweet
            assert TEST_FILTER.lower() in tweet["html"].lower()

    def test_get_tweets_filter_with_format(self, live_client):
        """Test fetching filtered tweets with different formats."""
        for format_type in ["summary", "detailed"]:
            result = live_client.get_tweets_filter(
                account=TEST_ACCOUNT, filter=TEST_FILTER, limit=TEST_LIMIT, format=format_type
            )

            assert result is not None
            assert isinstance(result, dict)
            assert "data" in result
            assert "tweets" in result["data"]
            assert "metadata" in result
            assert "success" in result
            assert result["success"] is True

            if format_type == "summary":
                if result["data"]["tweets"]:  # Only check if there are tweets
                    assert "summaries" in result["data"]
            elif format_type == "detailed" and result["data"]["tweets"]:  # Only check if there are tweets
                assert "analysis" in result["data"]

    def test_rate_limiting(self, live_client):
        """Test API rate limiting behavior."""
        for _ in range(5):
            result = live_client.get_tweets_research(limit=1)
            assert result is not None
            assert isinstance(result, dict)
            assert "data" in result
            assert "tweets" in result["data"]

    def test_invalid_account(self, live_client):
        """Test behavior with invalid account name."""
        with pytest.raises(ResearchAgentAPIError):
            live_client.get_tweets_filter(account="nonexistent_account_123456789", filter=TEST_FILTER, limit=TEST_LIMIT)

    def test_invalid_format(self, live_client):
        """Test behavior with invalid format parameter."""
        with pytest.raises(ValueError, match="format must be 'summary' or 'detailed'"):
            live_client.get_tweets_filter(account=TEST_ACCOUNT, filter=TEST_FILTER, format="invalid_format")

    def test_pagination_limits(self, live_client):
        """Test pagination behavior with different limits."""
        limits = [1, 5, 10, 20]
        for limit in limits:
            result = live_client.get_tweets_research(limit=limit)
            assert result is not None
            assert isinstance(result, dict)
            assert "data" in result
            assert "tweets" in result["data"]
            assert len(result["data"]["tweets"]) <= limit

    @pytest.mark.skip(reason="This test is not working as expected")
    def test_date_filtering(self, live_client):
        """Test date-based filtering behavior."""
        time_ranges = [1, 7, 30]  # days
        for days in time_ranges:
            # Use a reference date in 2024 since that's when the tweets are from
            reference_date = datetime(2024, 10, 1, tzinfo=UTC)
            since = (reference_date - timedelta(days=days)).isoformat()

            result = live_client.get_tweets_multi_account(limit=TEST_LIMIT, since=since)
            assert result is not None
            assert isinstance(result, dict)
            assert "data" in result
            assert "tweets" in result["data"]

            if result["data"]["tweets"]:
                for tweet in result["data"]["tweets"]:
                    tweet_date = datetime.fromtimestamp(int(tweet["timestamp"]), tz=UTC)
                    assert tweet_date >= datetime.fromisoformat(since)

    def test_limit_enforcement(self, live_client):
        """Test that limits are properly enforced."""
        # Test default limit
        result = live_client.get_tweets_multi_account()
        assert len(result["data"]["tweets"]) <= 40

        # Test custom limit within bounds, 4 accounts * 5 tweets
        result = live_client.get_tweets_multi_account(limit=5)
        assert len(result["data"]["tweets"]) <= 20

        # Test limit exceeding max, 4 accounts * 50 tweets
        result = live_client.get_tweets_multi_account(limit=100)
        assert len(result["data"]["tweets"]) <= 200
