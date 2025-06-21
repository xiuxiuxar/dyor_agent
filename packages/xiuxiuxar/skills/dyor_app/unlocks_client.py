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
from typing import Any
from datetime import UTC, datetime, timedelta

from bs4 import BeautifulSoup
from aea.skills.base import Model

from packages.xiuxiuxar.skills.dyor_app.models import ScrapedDataItem
from packages.xiuxiuxar.skills.dyor_app.base_client import BaseClient, BaseAPIError


STATUS_FORCELIST = (429, 500, 502, 503, 504)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.3"
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

    def fetch_all_unlocks(self) -> ScrapedDataItem:
        """Fetch and store all unlocks data for all projects as a ScrapedDataItem."""
        url = f"{self.base_url.rstrip('/')}/"
        self.context.logger.info(f"Fetching all unlocks data from {url}")
        try:
            resp = self._make_request(
                method="GET",
                endpoint=url,
                headers=DEFAULT_HEADERS,
                base_url_override="",  # Use the full URL
                expect_json=False,
            )
            if not resp:
                self.context.logger.warning(f"No unlock data received from {url}")
                timestamp = datetime.now(UTC).isoformat()
                return ScrapedDataItem(
                    source=self.source_name,
                    title="Unlocks",
                    url=url,
                    summary="No unlock data available.",
                    timestamp=timestamp,
                    metadata={},
                )
        except Exception as e:
            self.context.logger.exception(f"Failed to fetch unlocks page: {e}")
            msg = f"Failed to fetch unlocks page: {e}"
            raise UnlocksClientError(msg) from e

        try:
            self.context.logger.info(f"Parsing HTML with BeautifulSoup. Type of resp: {type(resp)}")
            soup = BeautifulSoup(resp["text"], "html.parser")
            script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
            if not script_tag or not script_tag.string:
                self.context.logger.error("Could not find __NEXT_DATA__ script tag in HTML.")
                msg = "Could not find __NEXT_DATA__ script tag in HTML."
                raise UnlocksClientError(msg)
            self.context.logger.info(f"Found __NEXT_DATA__ script tag. Length: {len(script_tag.string)}")
            data = json.loads(script_tag.string)
            all_projects = data.get("props", {}).get("pageProps", {}).get("data", [])
            self.context.logger.info(f"Extracted {len(all_projects)} projects from unlocks JSON.")
            if not all_projects:
                self.context.logger.error("Could not find 'data' array in unlocks JSON.")
                msg = "Could not find 'data' array in unlocks JSON."
                raise UnlocksClientError(msg)
        except Exception as e:
            self.context.logger.exception(f"Failed to parse unlocks data: {e}")
            msg = f"Failed to parse unlocks data: {e}"
            raise UnlocksClientError(msg) from e

        timestamp = datetime.now(UTC).isoformat()
        item = ScrapedDataItem(
            source=self.source_name,
            title="All Unlocks Projects",
            url=url,
            summary=f"Unlocks data for {len(all_projects)} projects.",
            timestamp=timestamp,
            metadata={"all_projects": all_projects},
        )
        self.context.logger.info(f"Successfully fetched all unlocks data ({len(all_projects)} projects)")
        return item

    def get_project_from_all_unlocks(
        self,
        all_projects_data,
        coingecko_id: str | None = None,
        asset_name: str | None = None,
        symbol: str | None = None,
    ) -> dict | None:
        """Extract a single project's unlocks data from all_projects_data."""
        project = None
        if coingecko_id:
            project = next((p for p in all_projects_data if p.get("gecko_id") == coingecko_id), None)
        if not project and asset_name:
            project = next((p for p in all_projects_data if p.get("name", "").lower() == asset_name.lower()), None)
        if not project and symbol:
            project = next(
                (p for p in all_projects_data if p.get("token", "").split(":")[-1].lower() == symbol.lower()), None
            )
        return project

    def fetch_unlocks(
        self,
        symbol: str,
        asset_name: str | None = None,
        coingecko_id: str | None = None,
        all_unlocks_data: list | None = None,
    ) -> ScrapedDataItem:
        """Fetch unlock data for a single project, using cached all-projects data if provided."""
        if all_unlocks_data is None:
            all_unlocks_item = self.fetch_all_unlocks()
            all_unlocks_data = all_unlocks_item.metadata.get("all_projects", [])
        project = self.get_project_from_all_unlocks(all_unlocks_data, coingecko_id, asset_name, symbol)
        if not project:
            msg = f"Could not find project for gecko_id={coingecko_id}, asset_name={asset_name}, symbol={symbol}"
            raise UnlocksClientError(msg)

        events = project.get("events", [])

        summary = (
            f"Token: {project.get('name', '')}\n"
            f"Circ Supply: {project.get('circSupply', None)}\n"
            f"Max Supply: {project.get('maxSupply', None)}\n"
            f"Unlock Events: {len(events)}"
        )
        timestamp = datetime.now(UTC).isoformat()
        item = ScrapedDataItem(
            source=self.source_name,
            title=f"{project.get('name', '').capitalize()} Unlocks",
            url=self.base_url.rstrip("/") + "/",
            summary=summary,
            timestamp=timestamp,
            metadata={
                "token": project.get("name", ""),
                "circ_supply": project.get("circSupply", None),
                "max_supply": project.get("maxSupply", None),
                "events": events,
                "sources": project.get("sources", []),
                "raw_project": project,
            },
        )
        self.context.logger.info(f"Successfully fetched unlocks data for {project.get('name', '')}")
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
            self.context.logger.warning(f"Error checking data freshness: {e}")
            return False
