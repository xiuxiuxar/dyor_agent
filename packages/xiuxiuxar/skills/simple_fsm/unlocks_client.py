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

"""Unlocks Client using BeautifulSoup."""

import json
import logging
from typing import Any
from datetime import UTC, datetime, timedelta

import requests
from bs4 import BeautifulSoup
from aea.skills.base import Model

from packages.xiuxiuxar.skills.simple_fsm.models import ScrapedDataItem
from packages.xiuxiuxar.skills.simple_fsm.base_client import BaseClient, BaseAPIError


STATUS_FORCELIST = (429, 500, 502, 503, 504)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


class UnlocksClientError(BaseAPIError):
    """Unlocks Client specific error."""


class UnlocksClient(Model, BaseClient):
    """Client for scraping token unlock data."""

    def __init__(self, **kwargs: Any):
        name = kwargs.pop("name", "unlocks_client")
        skill_context = kwargs.pop("skill_context", None)
        base_url = kwargs.pop("base_url", None)
        timeout = kwargs.pop("timeout", 15)
        max_retries = kwargs.pop("max_retries", 3)
        backoff_factor = kwargs.pop("backoff_factor", 0.5)

        Model.__init__(self, name=name, skill_context=skill_context, **kwargs)
        BaseClient.__init__(
            self,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=STATUS_FORCELIST,
            headers=DEFAULT_HEADERS,
            error_class=UnlocksClientError,
        )
        self.source_name = "unlocks"
        self.logger = logging.getLogger(__name__)

    def fetch_unlocks(self, project: str) -> ScrapedDataItem:
        """Fetch unlock data for a given project from unlocks page.

        Args:
        ----
            project: The project name (e.g., 'celestia')

        Returns:
        -------
            ScrapedDataItem containing unlocks data

        Raises:
        ------
            UnlocksClientError: On network or parsing errors

        """
        project = project.lower().strip()

        url = f"{self.base_url.rstrip('/')}/{project}"

        self.logger.info(f"Fetching unlocks data for project '{project}' from {url}")
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=self.timeout)
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 500:
                    self.logger.warning(f"No unlock data available for {project} (500 error from {url}). Skipping unlocks fetch.")
                    # Return a ScrapedDataItem with empty/unavailable data
                    timestamp = datetime.now(UTC).isoformat()
                    item = ScrapedDataItem(
                        source=self.source_name,
                        title=f"{project.capitalize()} Unlocks",
                        url=url,
                        summary="No unlock data available.",
                        timestamp=timestamp,
                        metadata={},
                    )
                    return item
                else:
                    raise
        except Exception as e:
            self.logger.exception(f"Failed to fetch unlocks page: {e}")
            msg = f"Failed to fetch unlocks page: {e}"
            raise UnlocksClientError(msg) from e

        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
            if not script_tag or not script_tag.string:
                msg = "Could not find __NEXT_DATA__ script tag in HTML."
                raise UnlocksClientError(msg)
            data = json.loads(script_tag.string)
            emissions = data.get("props", {}).get("pageProps", {}).get("emissions", None)
            if not emissions:
                msg = "Could not find 'emissions' data in the expected path within the JSON."
                raise UnlocksClientError(msg)
        except Exception as e:
            self.logger.exception(f"Failed to parse unlocks data: {e}")
            msg = f"Failed to parse unlocks data: {e}"
            raise UnlocksClientError(msg) from e

        summary = (
            f"Token: {emissions.get('meta', {}).get('token', '')}\n"
            f"Circ Supply: {emissions.get('meta', {}).get('circSupply', None)}\n"
            f"Max Supply: {emissions.get('meta', {}).get('maxSupply', None)}\n"
            f"Unlock Events: {len(emissions.get('meta', {}).get('events', []))}"
        )
        timestamp = datetime.now(UTC).isoformat()
        item = ScrapedDataItem(
            source=self.source_name,
            title=f"{project.capitalize()} Unlocks",
            url=url,
            summary=summary,
            timestamp=timestamp,
            metadata={
                "token": emissions.get("meta", {}).get("token", ""),
                "circ_supply": emissions.get("meta", {}).get("circSupply", None),
                "max_supply": emissions.get("meta", {}).get("maxSupply", None),
                "chart_data": emissions.get("chartData", {}),
                "pie_chart": emissions.get("pieChartData", {}),
                "events": emissions.get("meta", {}).get("events", []),
                "sources": emissions.get("meta", {}).get("sources", []),
                "raw_emissions": emissions,
            },
        )
        self.logger.info(f"Successfully fetched unlocks data for {project}")
        return item

    def is_data_fresh(self, data: dict) -> bool:
        """Check if the unlock data is fresh enough to use (within 30 days)."""
        try:
            events = data.get("metadata", {}).get("raw_emissions", {}).get("meta", {}).get("events", [])

            if not events:
                return False

            latest_timestamp = max((event["timestamp"] for event in events if "timestamp" in event), default=0)

            latest_date = datetime.fromtimestamp(latest_timestamp, UTC)
            now = datetime.now(UTC)

            return (now - latest_date) < timedelta(days=30)

        except (KeyError, ValueError, TypeError) as e:
            self.logger.warning(f"Error checking data freshness: {e}")
            return False
