"""Data sources for the simple FSM."""

from packages.xiuxiuxar.skills.simple_fsm.unlocks_client import UnlocksClient
from packages.xiuxiuxar.skills.simple_fsm.trendmoon_client import TrendmoonClient, TrendmoonAPIError
from packages.xiuxiuxar.skills.simple_fsm.lookonchain_client import LookOnChainClient
from packages.xiuxiuxar.skills.simple_fsm.treeofalpha_client import TreeOfAlphaClient
from packages.xiuxiuxar.skills.simple_fsm.researchagent_client import ResearchAgentClient


def trendmoon_social(context, symbol, **_):
    """Fetch social trends from Trendmoon."""
    cfg = context.api_client_configs["trendmoon"]
    return TrendmoonClient(
        name="trendmoon_client",
        base_url=cfg["base_url"],
        insights_url=cfg.get("insights_url"),
        api_key=cfg["api_key"],
        max_retries=cfg.get("max_retries", 3),
        backoff_factor=cfg.get("backoff_factor", 0.5),
        timeout=cfg.get("timeout", 15),
        skill_context=context,
    ).get_social_trend(symbol=symbol, time_interval="1d", date_interval=3)


def trendmoon_coin_details(context, symbol, **_):
    """Fetch coin details from Trendmoon."""
    cfg = context.api_client_configs["trendmoon"]
    return TrendmoonClient(
        name="trendmoon_client",
        base_url=cfg["base_url"],
        insights_url=cfg.get("insights_url"),
        api_key=cfg["api_key"],
        max_retries=cfg.get("max_retries", 3),
        backoff_factor=cfg.get("backoff_factor", 0.5),
        timeout=cfg.get("timeout", 15),
        skill_context=context,
    ).get_coin_details(symbol=symbol)


def trendmoon_topic_summary(context, topic, **_):
    """Fetch topic summary from Trendmoon."""
    cfg = context.api_client_configs["trendmoon"]
    return TrendmoonClient(
        name="trendmoon_client",
        base_url=cfg["base_url"],
        insights_url=cfg.get("insights_url"),
        api_key=cfg["api_key"],
        max_retries=cfg.get("max_retries", 3),
        backoff_factor=cfg.get("backoff_factor", 0.5),
        timeout=cfg.get("timeout", 15),
        skill_context=context,
    ).get_topic_summary(topic=topic)


def trendmoon_project_summary(context, symbol, **_):
    """Fetch project summary from Trendmoon."""
    cfg = context.api_client_configs["trendmoon"]
    try:
        return TrendmoonClient(
            base_url=cfg["base_url"],
            insights_url=cfg.get("insights_url"),
            api_key=cfg["api_key"],
            max_retries=cfg.get("max_retries", 3),
            backoff_factor=cfg.get("backoff_factor", 0.5),
            timeout=cfg.get("timeout", 15),
            skill_context=context,
        ).get_project_summary(symbol=symbol)
    except TrendmoonAPIError as e:
        context.logger.warning(f"Error fetching project summary: {e}")
        return {}


def lookonchain_fetcher(context, symbol, **_):
    """Fetch data from LookOnChain."""
    cfg = context.api_client_configs["lookonchain"]
    return LookOnChainClient(
        name="lookonchain_client",
        base_url=cfg["base_url"],
        search_endpoint=cfg.get("search_endpoint"),
        timeout=cfg.get("timeout", 15),
        max_retries=cfg.get("max_retries", 3),
        backoff_factor=cfg.get("backoff_factor", 0.5),
        skill_context=context,
    ).search(query=symbol, count=25)


def treeofalpha_fetcher(context, symbol, **_):
    """Fetch data from TreeOfAlpha."""
    cfg = context.api_client_configs["treeofalpha"]
    return TreeOfAlphaClient(
        name="treeofalpha_client",
        base_url=cfg["base_url"],
        news_endpoint=cfg.get("news_endpoint"),
        cache_ttl=cfg.get("cache_ttl"),
        max_retries=cfg.get("max_retries"),
        backoff_factor=cfg.get("backoff_factor"),
        timeout=cfg.get("timeout"),
        skill_context=context,
    ).search_news(query=symbol)


def researchagent_fetcher(context, symbol, asset_name=None, **_):
    """Fetch data from ResearchAgent."""
    cfg = context.api_client_configs["researchagent"]
    return ResearchAgentClient(
        name="researchagent_client",
        base_url=cfg["base_url"],
        api_key=cfg.get("api_key"),
        skill_context=context,
    ).get_tweets_filter(account="aixbt_agent", filter=asset_name or symbol, limit=25)


def unlocks_fetcher(context, *_):
    """Fetch data from Unlocks."""
    cfg = context.api_client_configs["unlocks"]
    # Only fetch the full unlocks dataset
    return UnlocksClient(
        name="unlocks_client",
        base_url=cfg["base_url"],
        timeout=cfg.get("timeout", 15),
        max_retries=cfg.get("max_retries", 3),
        backoff_factor=cfg.get("backoff_factor", 0.5),
        skill_context=context,
    ).fetch_all_unlocks()


def unlocks_project_filter(all_projects, coingecko_id=None, asset_name=None, symbol=None):
    """Filter all_projects for a specific project and return filtered events (cliff, insiders, privateSale)."""

    def get_project():
        if coingecko_id:
            return next((p for p in all_projects if p.get("gecko_id") == coingecko_id), None)
        if asset_name:
            return next((p for p in all_projects if p.get("name", "").lower() == asset_name.lower()), None)
        if symbol:
            return next((p for p in all_projects if p.get("token", "").split(":")[-1].lower() == symbol.lower()), None)
        return None

    project = get_project()
    if not project:
        return None, []
    events = project.get("events", [])
    filtered_events = [
        e for e in events if e.get("unlockType") == "cliff" and e.get("category") in {"insiders", "privateSale"}
    ]
    return project, filtered_events


DATA_SOURCES = {
    "trendmoon": {
        "fetchers": {
            "social": trendmoon_social,
            "coin_details": trendmoon_coin_details,
            "project_summary": trendmoon_project_summary,
            "topic_summary": trendmoon_topic_summary,
        },
        "processor": "serialize_trendmoon_data",
        "data_type_handler": "multi",
    },
    "lookonchain": {
        "fetcher": lookonchain_fetcher,
        "processor": "serialize_lookonchain_data",
        "data_type_handler": "single",
    },
    "treeofalpha": {
        "fetcher": treeofalpha_fetcher,
        "processor": "serialize_treeofalpha_data",
        "data_type_handler": "single",
    },
    "researchagent": {
        "fetcher": researchagent_fetcher,
        "processor": "serialize_researchagent_data",
        "data_type_handler": "single",
    },
    "unlocks": {
        "fetcher": unlocks_fetcher,
        "processor": "serialize_unlocks_data",
        "data_type_handler": "single",
    },
}
