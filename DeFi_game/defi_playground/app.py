# app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DeFi Playground", layout="wide")

# --- твои папки с данными ---
BASE_MAIN = Path(r"C:\Users\Shaim\defi game\synthetic_defi_game_data")
BASE_EXTRA = Path(r"C:\Users\Shaim\defi game\synthetic_data")

# какие файлы считаем "основными" / "дополнительными"
EXPECTED = {
    # core
    "users": "users.csv",
    "sessions": "sessions.csv",
    "market_trades": "market_trades.csv",
    "token_ledger": "token_ledger.csv",

    # enriched / extra
    "price_oracle": "price_oracle_daily.csv",  # или price_oracle_daily.csv если так назвал
    "econ_events": "econ_events.csv",
    "resource_production_daily": "resource_production_daily.csv",
    "planet_state_daily": "planet_state_daily.csv",
    "guild_membership": "guild_membership.csv",
    "guild_events": "guild_events.csv",
    "nft_ownership": "nft_ownership.csv",
    "nft_usage_events": "nft_usage_events.csv",
    "funnel_events": "funnel_events.csv",
    "marketing_spend": "marketing_spend.csv",

    # если у тебя есть отдельные:
    "planets": "planets.csv",
    "alliances": "alliances.csv",   # если файл называется иначе (aliance.csv) — поправь тут
}

# колонки, которые пытаемся парсить как datetime если они есть
DT_COLS = ["ts", "start_ts", "end_ts", "created_at", "day", "date", "start_day", "end_day", "event_ts", "timestamp"]


def _read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    for c in DT_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_tables(main_dir: str, extra_dir: str):
    main_dir = Path(main_dir)
    extra_dir = Path(extra_dir)

    tables = {}
    missing = []

    # 1) сначала пробуем найти ожидаемые файлы в main, потом в extra
    for key, fname in EXPECTED.items():
        p1 = main_dir / fname
        p2 = extra_dir / fname

        if p1.exists():
            df = _read_csv_safe(p1)
            tables[key] = _coerce_datetimes(df)
        elif p2.exists():
            df = _read_csv_safe(p2)
            tables[key] = _coerce_datetimes(df)
        else:
            missing.append((key, fname))

    # 2) дополнительно: подхватим ВСЕ CSV из папок (если ты генерил новые таблицы и забыл добавить в EXPECTED)
    #    ключ = имя файла без расширения
    for base in [main_dir, extra_dir]:
        if not base.exists():
            continue
        for p in base.glob("*.csv"):
            stem = p.stem
            if stem not in tables:
                try:
                    df = _read_csv_safe(p)
                    tables[stem] = _coerce_datetimes(df)
                except Exception:
                    pass

    return tables, missing


# =========================
# UI
# =========================

st.title("DeFi Game — Analytics Playground")

with st.sidebar:
    st.header("Data folders")
    main_dir = st.text_input("Main data dir", str(BASE_MAIN))
    extra_dir = st.text_input("Extra data dir", str(BASE_EXTRA))
    reload_btn = st.button("🔄 Reload data", type="primary")

if reload_btn:
    load_tables.clear()

with st.spinner("Loading CSV datasets…"):
    tables, missing = load_tables(main_dir, extra_dir)

st.session_state["TABLES"] = tables

st.success(f"Loaded tables: {len(tables)}")
if missing:
    st.warning("Some expected files not found (ok if you didn't generate them):")
    st.write(pd.DataFrame(missing, columns=["table_key", "expected_file"]))

# quick overview
core_need = ["users", "sessions", "market_trades", "token_ledger"]
core_ok = all(k in tables and tables[k] is not None and len(tables[k]) for k in core_need)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Core status")
    st.write({k: ("OK" if k in tables else "MISSING") for k in core_need})
with c2:
    st.subheader("Shapes")
    shapes = []
    for k in core_need:
        if k in tables and tables[k] is not None:
            shapes.append((k, tables[k].shape))
    st.write(shapes)

st.divider()
st.info("Open needed page from menu in Streamlit (example, Policy Simulator).")
