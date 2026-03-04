import os
import pandas as pd
import streamlit as st

KNOWN_FILES = {
    "users": "users.csv",
    "sessions": "sessions.csv",
    "market_trades": "market_trades.csv",
    "token_ledger": "token_ledger.csv",

    # enriched
    "price_oracle_daily": "price_oracle_daily.csv",
    "econ_events": "econ_events.csv",
    "resource_production_daily": "resource_production_daily.csv",
    "planet_state_daily": "planet_state_daily.csv",
    "guild_membership": "guild_membership.csv",
    "guild_events": "guild_events.csv",
    "nft_ownership": "nft_ownership.csv",
    "nft_usage_events": "nft_usage_events.csv",
    "funnel_events": "funnel_events.csv",
    "marketing_spend": "marketing_spend.csv",

    # optional (you mentioned)
    "planets": "planets.csv",
    "alliances": "aliance.csv",   # если у тебя именно так
    "alliance": "alliance.csv",   # fallback
}

@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_all_tables(data_dir: str, verbose: bool = False) -> dict:
    tables = {}
    if not data_dir:
        return tables

    for key, fn in KNOWN_FILES.items():
        path = os.path.join(data_dir, fn)
        if os.path.exists(path):
            try:
                df = _read_csv(path)
                tables[key] = df
                if verbose:
                    print(f"[load] {key}: {df.shape} from {path}")
            except Exception as e:
                if verbose:
                    print(f"[load] FAIL {key} from {path}: {e}")

    # alias logic for alliances
    if "alliances" not in tables and "alliance" in tables:
        tables["alliances"] = tables["alliance"]

    return tables

def list_loaded_tables(tables: dict) -> pd.DataFrame:
    rows = []
    for k, df in tables.items():
        rows.append({
            "table": k,
            "rows": len(df),
            "cols": len(df.columns),
            "columns": ", ".join(list(df.columns)[:18]) + (" …" if len(df.columns) > 18 else "")
        })
    return pd.DataFrame(rows).sort_values(["table"])
