"""Data sources for the simple FSM."""

DATA_SOURCES = {
    "trendmoon": {
        "fetchers": {
            "social": lambda client_class, client_kwargs, symbol, _asset_name=None: (
                client_class(**client_kwargs).get_social_trend(symbol=symbol, time_interval="1d", date_interval=3)
            ),
            "coin_details": lambda client_class, client_kwargs, symbol, _asset_name=None: (
                client_class(**client_kwargs).get_coin_details(symbol=symbol)
            ),
        },
        "processor": "serialize_trendmoon_data",
        "data_type_handler": "multi",
    },
    "lookonchain": {
        "fetcher": lambda client_class, client_kwargs, symbol, _asset_name=None: (
            client_class(**client_kwargs).search(query=symbol, count=25)
        ),
        "processor": "serialize_lookonchain_data",
        "data_type_handler": "single",
    },
    "treeofalpha": {
        "fetcher": lambda client_class, client_kwargs, symbol, _asset_name=None: (
            client_class(**client_kwargs).search_news(query=symbol)
        ),
        "processor": "serialize_treeofalpha_data",
        "data_type_handler": "single",
    },
    "researchagent": {
        "fetcher": lambda client_class, client_kwargs, symbol, asset_name=None: (
            client_class(**client_kwargs).get_tweets_filter(
                account="aixbt_agent", filter=asset_name or symbol, limit=25
            )
        ),
        "processor": "serialize_researchagent_data",
        "data_type_handler": "single",
    },
}
