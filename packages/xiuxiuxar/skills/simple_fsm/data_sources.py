"""Data sources for the simple FSM."""

DATA_SOURCES = {
    "trendmoon": {
        "fetchers": {
            "social": lambda client, symbol, **_: client.get_social_trend(
                symbol=symbol, time_interval="1d", date_interval=3
            ),
            "coin_details": lambda client, symbol, **_: client.get_coin_details(symbol=symbol),
            "project_summary": lambda client, symbol, **_: client.get_project_summary(symbol=symbol),
        },
        "processor": "serialize_trendmoon_data",
        "data_type_handler": "multi",
    },
    "lookonchain": {
        "fetcher": lambda client, symbol, **_: client.search(query=symbol, count=25),
        "processor": "serialize_lookonchain_data",
        "data_type_handler": "single",
    },
    "treeofalpha": {
        "fetcher": lambda client, symbol, **_: client.search_news(query=symbol),
        "processor": "serialize_treeofalpha_data",
        "data_type_handler": "single",
    },
    "researchagent": {
        "fetcher": lambda client, symbol, asset_name=None, **_: client.get_tweets_filter(
            account="aixbt_agent", filter=asset_name or symbol, limit=25
        ),
        "processor": "serialize_researchagent_data",
        "data_type_handler": "single",
    },
}
