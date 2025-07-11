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

"""Metrics calculator for DYOR App skill."""

from datetime import datetime

from packages.xiuxiuxar.skills.dyor_app.utils import calculate_percentage_change
from packages.xiuxiuxar.skills.dyor_app.data_models import KeyMetrics, SocialSummary


class MetricsCalculator:
    """Handles all metrics calculations including weekly changes."""

    def __init__(self, context, logger):
        """Initialize with context and logger."""
        self.context = context
        self.logger = logger

    def get_latest_trend_values(self, trend_market_data: list, key: str, count: int = 2) -> list:
        """Get the latest N values for a specific key from trend market data."""
        filtered = [e for e in trend_market_data if key in e and isinstance(e[key], int | float) and e[key] is not None]
        filtered.sort(key=lambda x: datetime.fromisoformat(x["date"]), reverse=True)
        return filtered[:count]

    def calculate_weekly_average(
        self, data_points: list, start_idx: int, end_idx: int, key: str | None = None
    ) -> float:
        """Calculate average value for a week period."""
        if not data_points or len(data_points) <= max(start_idx, end_idx):
            return 0.0

        week_data = data_points[start_idx : end_idx + 1]
        if not week_data:
            return 0.0

        try:
            # Get the key dynamically from the first item if not provided
            if key is None:
                sample_item = week_data[0]
                for possible_key in ["total_volume", "lc_social_dominance", "social_mentions", "lc_sentiment"]:
                    if possible_key in sample_item:
                        key = possible_key
                        break

            if not key:
                return 0.0

            values = [float(item[key]) for item in week_data if key in item and item[key] is not None]
            return sum(values) / len(values) if values else 0.0
        except (ValueError, TypeError, KeyError):
            return 0.0

    def calculate_weekly_change(self, trend_data: list, key: str, use_percentage: bool = True) -> float | None:
        """Calculate weekly change."""
        data_points = self.get_latest_trend_values(trend_data, key, 14)
        if len(data_points) < 14:
            self.logger.warning(
                f"Insufficient data points (got {len(data_points)}) "
                f"for weekly change calculation of '{key}'. Returning None."
            )
            return None
        result = None
        try:
            # True week-over-week averages (14+ days)
            current_avg = self.calculate_weekly_average(data_points, 0, 6, key)
            previous_avg = self.calculate_weekly_average(data_points, 7, 13, key)
            if use_percentage and previous_avg > 0:
                result = calculate_percentage_change(current_avg, previous_avg)
            elif not use_percentage:
                result = round(current_avg - previous_avg, 2)
        except (ValueError, TypeError, KeyError) as e:
            self.logger.warning(f"Failed to calculate {key} weekly change: {e}")
        return result

    def build_key_metrics(self, trend_market_data) -> KeyMetrics:
        """Build key metrics including price, volume, and mindshare changes."""
        self.logger.info(f"Building key metrics with {len(trend_market_data)} data points")

        # Price change 24h
        price_points = self.get_latest_trend_values(trend_market_data, "price")
        price_change_24h = (
            calculate_percentage_change(float(price_points[0]["price"]), float(price_points[1]["price"]))
            if len(price_points) == 2
            else 0.0
        )

        # Volume changes (24h and 7d)
        volume_points_24h = self.get_latest_trend_values(trend_market_data, "total_volume")
        volume_change_24h = (
            calculate_percentage_change(
                float(volume_points_24h[0]["total_volume"]), float(volume_points_24h[1]["total_volume"])
            )
            if len(volume_points_24h) == 2
            else 0.0
        )
        volume_change_7d = self.calculate_weekly_change(trend_market_data, "total_volume")

        # Mindshare current value and changes (24h and 7d)
        mindshare_points_24h = self.get_latest_trend_values(trend_market_data, "lc_social_dominance")
        latest_mindshare = float(mindshare_points_24h[0]["lc_social_dominance"]) if mindshare_points_24h else 0.0

        mindshare_change_24h = (
            calculate_percentage_change(
                float(mindshare_points_24h[0]["lc_social_dominance"]),
                float(mindshare_points_24h[1]["lc_social_dominance"]),
            )
            if len(mindshare_points_24h) == 2
            else 0.0
        )
        mindshare_change_7d = self.calculate_weekly_change(trend_market_data, "lc_social_dominance")

        return KeyMetrics(
            mindshare=latest_mindshare,
            mindshare_24h=mindshare_change_24h,
            mindshare_7d=mindshare_change_7d,
            volume_change_24h=volume_change_24h,
            volume_change_7d=volume_change_7d,
            price_change_24h=price_change_24h,
        )

    def build_social_summary(self, social_data) -> SocialSummary:
        """Build social summary including sentiment and mention changes."""
        if social_data is None:
            social_data = {}
        trend_market_data = social_data.get("trend_market_data", [])

        # Latest sentiment
        latest_sentiment = 0.0
        sentiment_points = self.get_latest_trend_values(trend_market_data, "lc_sentiment", 1)
        if sentiment_points:
            latest_sentiment = sentiment_points[0]["lc_sentiment"]

        # Sentiment weekly change (use absolute change, not percentage for sentiment scores)
        sentiment_change_7d = self.calculate_weekly_change(trend_market_data, "lc_sentiment", use_percentage=False)

        # Mention changes (24h and 7d) - handle zero values carefully
        mention_points = self.get_latest_trend_values(trend_market_data, "social_mentions")
        if len(mention_points) >= 2 and mention_points[0]["social_mentions"] == 0:
            # Skip zero values by getting next two points
            all_mentions = self.get_latest_trend_values(trend_market_data, "social_mentions", 3)
            mention_points = all_mentions[1:3] if len(all_mentions) >= 3 else mention_points

        mention_change_24h = 0.0
        if len(mention_points) == 2:
            try:
                latest_mentions = float(mention_points[0]["social_mentions"])
                previous_mentions = float(mention_points[1]["social_mentions"])
                mention_change_24h = calculate_percentage_change(latest_mentions, previous_mentions)
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse social mentions for 24h change: {e}")

        mention_change_7d = self.calculate_weekly_change(trend_market_data, "social_mentions")

        return SocialSummary(
            sentiment_score=latest_sentiment,
            top_keywords=social_data.get("symbols", []),
            mention_change_24h=mention_change_24h,
            mention_change_7d=mention_change_7d,
            sentiment_change_7d=sentiment_change_7d,
        )
