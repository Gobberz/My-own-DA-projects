# pages/01_Economy_and_Market_Desk.py
# Economy Desk + Market Quality (combined)
# All UI labels and comments are in English.

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# =========================
# App setup
# =========================
st.set_page_config(
    page_title="Economy Desk + Market Quality",
    layout="wide",
)

st.title("Economy Desk + Market Quality")
st.caption("Sinks/Sources • Velocity • Issuance • Flows • Slippage/Spread/Volatility • Liquidity stress")


# =========================
# Helpers
# =========================
def _to_dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

def _floor_day(ts: pd.Series) -> pd.Series:
    return _to_dt(ts).dt.floor("D")

def _pick(df: pd.DataFrame, candidates: list[str], required: bool = True, df_name: str = "df") -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"{df_name}: missing one of {candidates}. Available: {list(df.columns)}")
    return None

def _find_csv(roots: list[str], filename: str) -> str | None:
    """
    Search filename in multiple roots. Returns first found path or None.
    """
    for r in roots:
        p = Path(r) / filename
        if p.exists():
            return str(p)
    return None

@st.cache_data(show_spinner=False)
def load_csv_any(roots: tuple[str, ...], filename: str) -> pd.DataFrame | None:
    p = _find_csv(list(roots), filename)
    if not p:
        return None
    df = pd.read_csv(p)
    return df

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd

def safe_pct(x, q: float):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.quantile(q))


# =========================
# Sidebar: paths & filters
# =========================
st.sidebar.header("Data paths")

default_extra = r"C:\Users\Shaim\defi game\synthetic_data"
default_main  = r"C:\Users\Shaim\defi game\synthetic_defi_game_data"

ROOT_EXTRA = st.sidebar.text_input("Extra synthetic data folder", value=default_extra)
ROOT_MAIN  = st.sidebar.text_input("Main synthetic data folder", value=default_main)

ROOTS = (ROOT_MAIN, ROOT_EXTRA)

st.sidebar.divider()
st.sidebar.header("Files (auto)")
st.sidebar.write("Looking for CSVs in:")
st.sidebar.code("\n".join(ROOTS), language="text")

# Core files
users         = load_csv_any(ROOTS, "users.csv")
sessions      = load_csv_any(ROOTS, "sessions.csv")
market_trades = load_csv_any(ROOTS, "market_trades.csv")
token_ledger  = load_csv_any(ROOTS, "token_ledger.csv")

# Enriched files
price_oracle  = load_csv_any(ROOTS, "price_oracle.csv")  # expected: day, resource, price_token, volatility
econ_events   = load_csv_any(ROOTS, "econ_events.csv")   # optional

if market_trades is None or token_ledger is None:
    st.error("Missing required tables: market_trades.csv and/or token_ledger.csv. Put them into one of the folders above.")
    st.stop()

with st.sidebar.expander("Loaded tables (shapes)", expanded=True):
    def _shape(df):
        return None if df is None else df.shape
    st.write({
        "users": _shape(users),
        "sessions": _shape(sessions),
        "market_trades": _shape(market_trades),
        "token_ledger": _shape(token_ledger),
        "price_oracle": _shape(price_oracle),
        "econ_events": _shape(econ_events),
    })

st.sidebar.divider()
st.sidebar.header("Filters")

# We will infer date bounds from trades/ledger after parsing
# placeholders (set later)
date_min_ui = None
date_max_ui = None


# =========================
# Step 1: Normalize inputs
# =========================
status = st.status("Preparing data…", expanded=True)

status.write("Parsing market_trades…")
mt = market_trades.copy()

mt_ts = _pick(mt, ["ts", "timestamp"], df_name="market_trades")
mt["ts"] = _to_dt(mt[mt_ts])
mt["day"] = mt["ts"].dt.floor("D")

# resource key
res_col = _pick(mt, ["asset_base", "resource", "base_asset", "asset"], df_name="market_trades")
mt["resource"] = mt[res_col].astype(str).str.upper()

# gross volume proxy (token-denominated)
amt_base = _pick(mt, ["amount_base", "qty_base", "base_amount"], df_name="market_trades")
price_col = _pick(mt, ["price", "px"], df_name="market_trades")

mt[amt_base] = pd.to_numeric(mt[amt_base], errors="coerce").fillna(0.0)
mt[price_col] = pd.to_numeric(mt[price_col], errors="coerce").fillna(0.0)
mt["gross_token"] = mt[amt_base] * mt[price_col]

maker = _pick(mt, ["maker_user_id", "user_id"], required=False, df_name="market_trades")

# fees/taxes (may exist in trades)
fee_col = _pick(mt, ["fee_amount"], required=False, df_name="market_trades")
tax_col = _pick(mt, ["tax_amount"], required=False, df_name="market_trades")
slip_col = _pick(mt, ["slippage_bps"], required=False, df_name="market_trades")
spr_col  = _pick(mt, ["spread_bps"], required=False, df_name="market_trades")

for c in [fee_col, tax_col, slip_col, spr_col]:
    if c and c in mt.columns:
        mt[c] = pd.to_numeric(mt[c], errors="coerce")

status.write("Parsing token_ledger…")
tl = token_ledger.copy()
tl_ts = _pick(tl, ["ts", "timestamp"], df_name="token_ledger")
tl["ts"] = _to_dt(tl[tl_ts])
tl["day"] = tl["ts"].dt.floor("D")

tl_user = _pick(tl, ["user_id"], df_name="token_ledger")
tx_type = _pick(tl, ["tx_type", "type"], df_name="token_ledger")
amt_col = _pick(tl, ["amount", "amt"], df_name="token_ledger")
tl[amt_col] = pd.to_numeric(tl[amt_col], errors="coerce").fillna(0.0)
tl["abs_amount"] = tl[amt_col].abs()

# Price oracle
po = None
if price_oracle is not None:
    status.write("Parsing price_oracle…")
    po = price_oracle.copy()
    po_day = _pick(po, ["day", "date"], df_name="price_oracle")
    po["day"] = _floor_day(po[po_day])
    po_res = _pick(po, ["resource", "asset_base"], df_name="price_oracle")
    po["resource"] = po[po_res].astype(str).str.upper()

    if "volatility" in po.columns:
        po["volatility"] = pd.to_numeric(po["volatility"], errors="coerce").fillna(0.0)
    if "price_token" in po.columns:
        po["price_token"] = pd.to_numeric(po["price_token"], errors="coerce").fillna(0.0)

# Date bounds
date_min = min(
    d for d in [
        mt["day"].min(),
        tl["day"].min()
    ] if pd.notna(d)
)
date_max = max(
    d for d in [
        mt["day"].max(),
        tl["day"].max()
    ] if pd.notna(d)
)

if pd.isna(date_min) or pd.isna(date_max):
    st.error("Could not infer date range from data.")
    st.stop()

date_from, date_to = st.sidebar.date_input(
    "Date range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date(),
)

date_from = pd.Timestamp(date_from)
date_to   = pd.Timestamp(date_to) + pd.Timedelta(days=1)  # inclusive -> exclusive

resources_all = sorted(mt["resource"].dropna().unique().tolist())
resource_pick = st.sidebar.multiselect("Resources", options=resources_all, default=resources_all[: min(6, len(resources_all))])

# Apply filters
mt_f = mt[(mt["day"] >= date_from) & (mt["day"] < date_to)].copy()
tl_f = tl[(tl["day"] >= date_from) & (tl["day"] < date_to)].copy()
if resource_pick:
    mt_f = mt_f[mt_f["resource"].isin(resource_pick)].copy()
    if po is not None:
        po_f = po[(po["day"] >= date_from) & (po["day"] < date_to) & (po["resource"].isin(resource_pick))].copy()
    else:
        po_f = None
else:
    po_f = po[(po["day"] >= date_from) & (po["day"] < date_to)].copy() if po is not None else None

status.write("Building aggregates…")


# =========================
# Step 2: Economy Desk aggregates
# =========================
# Ledger daily pivot
ledger_daily = (
    tl_f.groupby(["day", tx_type])["abs_amount"]
      .sum()
      .reset_index()
      .pivot_table(index="day", columns=tx_type, values="abs_amount", aggfunc="sum", fill_value=0.0)
      .reset_index()
)
ledger_daily.columns = [str(c) for c in ledger_daily.columns]

# Normalize expected types
for c in ["reward", "tax", "fee", "market_fee", "stake", "unstake"]:
    if c not in ledger_daily.columns:
        ledger_daily[c] = 0.0

ledger_daily["emission"] = ledger_daily["reward"]
ledger_daily["sinks"] = ledger_daily["tax"] + ledger_daily["fee"] + ledger_daily["market_fee"]
ledger_daily["net_issuance"] = ledger_daily["emission"] - ledger_daily["sinks"]

# Locked proxy: stake increases locked, unstake decreases locked
ledger_daily["locked_delta"] = ledger_daily["stake"] - ledger_daily["unstake"]
ledger_daily = ledger_daily.sort_values("day")
ledger_daily["locked_proxy"] = ledger_daily["locked_delta"].cumsum()
ledger_daily["locked_proxy"] = ledger_daily["locked_proxy"].clip(lower=0.0)

ledger_daily["supply_proxy"] = ledger_daily["net_issuance"].cumsum()
ledger_daily["circulating_proxy"] = (ledger_daily["supply_proxy"] - ledger_daily["locked_proxy"]).clip(lower=1e-6)

# Market daily
market_daily = mt_f.groupby("day").agg(
    trades=("trade_id", "count"),
    volume_token=("gross_token", "sum"),
    volume_base=(amt_base, "sum"),
).reset_index()

if fee_col and fee_col in mt_f.columns:
    market_daily["fees_token_trades"] = mt_f.groupby("day")[fee_col].sum().reindex(market_daily["day"]).values
else:
    market_daily["fees_token_trades"] = 0.0

if tax_col and tax_col in mt_f.columns:
    market_daily["taxes_token_trades"] = mt_f.groupby("day")[tax_col].sum().reindex(market_daily["day"]).values
else:
    market_daily["taxes_token_trades"] = 0.0

if maker:
    traders_daily = mt_f.groupby("day")[maker].nunique().reset_index(name="active_traders")
    market_daily = market_daily.merge(traders_daily, on="day", how="left")
else:
    market_daily["active_traders"] = np.nan

# Economy daily merged
econ_daily = market_daily.merge(
    ledger_daily[["day", "emission", "sinks", "net_issuance", "supply_proxy", "locked_proxy", "circulating_proxy"]],
    on="day",
    how="outer",
).sort_values("day")

econ_daily[["trades","volume_token","volume_base","fees_token_trades","taxes_token_trades","active_traders"]] = (
    econ_daily[["trades","volume_token","volume_base","fees_token_trades","taxes_token_trades","active_traders"]].fillna(0.0)
)
econ_daily[["emission","sinks","net_issuance","supply_proxy","locked_proxy","circulating_proxy"]] = (
    econ_daily[["emission","sinks","net_issuance","supply_proxy","locked_proxy","circulating_proxy"]].fillna(0.0)
)

econ_daily["token_velocity"] = econ_daily["volume_token"] / econ_daily["circulating_proxy"].replace(0, np.nan)

# Some smoothings
for c in ["volume_token","token_velocity","net_issuance","sinks","emission","active_traders"]:
    econ_daily[f"{c}_7d_ma"] = econ_daily[c].rolling(7, min_periods=1).mean()


# =========================
# Step 3: Market Quality aggregates
# =========================
# Day×resource aggregates (volume-weighted where appropriate)
dr = mt_f.groupby(["day", "resource"]).agg(
    trades=("trade_id", "count"),
    volume_token=("gross_token", "sum"),
    volume_base=(amt_base, "sum"),
).reset_index()

if fee_col and fee_col in mt_f.columns:
    dr["fees_token"] = mt_f.groupby(["day","resource"])[fee_col].sum().reset_index(name="fees_token")["fees_token"]
else:
    dr["fees_token"] = 0.0

if tax_col and tax_col in mt_f.columns:
    dr["taxes_token"] = mt_f.groupby(["day","resource"])[tax_col].sum().reset_index(name="taxes_token")["taxes_token"]
else:
    dr["taxes_token"] = 0.0

# Volume-weighted slippage/spread
if slip_col and slip_col in mt_f.columns:
    tmp = mt_f.copy()
    tmp["w"] = tmp["gross_token"].clip(lower=0.0)
    slip_w = tmp.groupby(["day","resource"]).apply(
        lambda g: (g[slip_col].fillna(0.0) * g["w"]).sum() / g["w"].sum() if g["w"].sum() > 0 else np.nan
    ).reset_index(name="slippage_bps_vw")
    dr = dr.merge(slip_w, on=["day","resource"], how="left")
else:
    dr["slippage_bps_vw"] = np.nan

if spr_col and spr_col in mt_f.columns:
    tmp = mt_f.copy()
    tmp["w"] = tmp["gross_token"].clip(lower=0.0)
    spr_w = tmp.groupby(["day","resource"]).apply(
        lambda g: (g[spr_col].fillna(0.0) * g["w"]).sum() / g["w"].sum() if g["w"].sum() > 0 else np.nan
    ).reset_index(name="spread_bps_vw")
    dr = dr.merge(spr_w, on=["day","resource"], how="left")
else:
    dr["spread_bps_vw"] = np.nan

# Merge oracle (volatility, price)
if po_f is not None:
    keep_cols = ["day", "resource"]
    if "volatility" in po_f.columns: keep_cols.append("volatility")
    if "price_token" in po_f.columns: keep_cols.append("price_token")
    dr = dr.merge(po_f[keep_cols], on=["day","resource"], how="left")

# Build daily market quality summary (weighted by volume_token)
mq = dr.groupby("day").apply(
    lambda g: pd.Series({
        "resources_traded": g["resource"].nunique(),
        "trades": g["trades"].sum(),
        "volume_token": g["volume_token"].sum(),
        "slippage_bps_vw": np.nan if g["volume_token"].sum() <= 0 else np.nansum(g["slippage_bps_vw"] * g["volume_token"]) / g["volume_token"].sum(),
        "spread_bps_vw": np.nan if g["volume_token"].sum() <= 0 else np.nansum(g["spread_bps_vw"] * g["volume_token"]) / g["volume_token"].sum(),
        "volatility_wavg": (np.nan if ("volatility" not in g.columns or g["volume_token"].sum() <= 0)
                            else np.nansum(pd.to_numeric(g["volatility"], errors="coerce").fillna(0.0) * g["volume_token"]) / g["volume_token"].sum())
    })
).reset_index()

# Liquidity stress index (simple z-score blend)
mq["stress_index"] = (
    zscore(mq["slippage_bps_vw"].fillna(0.0))
    + zscore(mq["spread_bps_vw"].fillna(0.0))
    + zscore(mq["volatility_wavg"].fillna(0.0))
)

# Concentration: top-1% daily share of volume (if maker exists)
conc = None
if maker:
    ud = mt_f.groupby(["day", maker])["gross_token"].sum().reset_index(name="u_volume")
    def top_share_day(g: pd.DataFrame, p=0.01) -> float:
        x = g["u_volume"].sort_values(ascending=False)
        if len(x) == 0:
            return np.nan
        k = max(1, int(len(x) * p))
        denom = x.sum()
        if denom <= 0:
            return np.nan
        return float(x.head(k).sum() / denom)

    conc = ud.groupby("day").apply(lambda g: top_share_day(g, 0.01)).reset_index(name="top1_share")
    mq = mq.merge(conc, on="day", how="left")
else:
    mq["top1_share"] = np.nan

status.update(label="Data prepared.", state="complete", expanded=False)


# =========================
# Tabs: Overview / Economy / Market Quality / Raw tables
# =========================
tab_overview, tab_econ, tab_mq, tab_raw = st.tabs(["Overview", "Economy Desk", "Market Quality", "Raw tables"])


# -------- Overview --------
with tab_overview:
    st.subheader("Economy health snapshot")

    last = econ_daily.dropna(subset=["day"]).sort_values("day").tail(1)
    if len(last) == 1:
        r = last.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Volume (token)", f"{r['volume_token']:.2f}")
        c2.metric("Velocity", f"{r['token_velocity']:.4f}" if pd.notna(r["token_velocity"]) else "n/a")
        c3.metric("Net issuance", f"{r['net_issuance']:.2f}")
        c4.metric("Sinks", f"{r['sinks']:.2f}")
        c5.metric("Active traders", f"{int(r['active_traders'])}")

    st.divider()

    st.subheader("Quick alerts (simple heuristics)")
    # Compute alert thresholds using recent window
    window = econ_daily.tail(30).copy()
    if len(window) >= 7:
        iss_p95 = safe_pct(window["net_issuance"], 0.95)
        vel_p05 = safe_pct(window["token_velocity"], 0.05)
        stress_p95 = safe_pct(mq.tail(30)["stress_index"], 0.95) if len(mq) else np.nan
        top1_p95 = safe_pct(mq.tail(30)["top1_share"], 0.95) if "top1_share" in mq.columns else np.nan

        alerts = []
        if pd.notna(iss_p95) and r["net_issuance"] >= iss_p95:
            alerts.append("⚠️ Net issuance spike (>= 95th percentile of last 30d)")
        if pd.notna(vel_p05) and pd.notna(r["token_velocity"]) and r["token_velocity"] <= vel_p05:
            alerts.append("⚠️ Velocity drop (<= 5th percentile of last 30d)")
        if pd.notna(stress_p95) and len(mq):
            last_stress = mq.sort_values("day").tail(1)["stress_index"].iloc[0]
            if last_stress >= stress_p95:
                alerts.append("⚠️ Liquidity stress spike (>= 95th percentile of last 30d)")
        if pd.notna(top1_p95) and "top1_share" in mq.columns:
            last_top1 = mq.sort_values("day").tail(1)["top1_share"].iloc[0]
            if pd.notna(last_top1) and last_top1 >= top1_p95:
                alerts.append("⚠️ Concentration spike (Top 1% volume share high)")

        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.success("No major anomalies detected by basic heuristics.")
    else:
        st.info("Not enough data for alert heuristics.")

    st.divider()

    st.subheader("High-level trends")
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["volume_token"], name="volume_token"))
        fig.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["token_velocity"], name="velocity", yaxis="y2"))
        fig.update_layout(
            height=420,
            xaxis_title="day",
            yaxis=dict(title="volume_token"),
            yaxis2=dict(title="velocity", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(econ_daily.set_index("day")[["volume_token", "token_velocity"]])


# -------- Economy Desk --------
with tab_econ:
    st.subheader("Sinks / Sources / Issuance / Velocity")

    c1, c2 = st.columns([2, 1])

    with c1:
        if PLOTLY_OK:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["emission"], name="emission"))
            fig.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["sinks"], name="sinks"))
            fig.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["net_issuance"], name="net_issuance"))
            fig.update_layout(height=420, xaxis_title="day", yaxis_title="token amount")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(econ_daily.set_index("day")[["emission", "sinks", "net_issuance"]])

        if PLOTLY_OK:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["supply_proxy"], name="supply_proxy"))
            fig2.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["locked_proxy"], name="locked_proxy"))
            fig2.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["circulating_proxy"], name="circulating_proxy"))
            fig2.update_layout(height=420, xaxis_title="day", yaxis_title="proxy balances")
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["token_velocity"], name="token_velocity"))
            fig3.add_trace(go.Scatter(x=econ_daily["day"], y=econ_daily["token_velocity_7d_ma"], name="velocity_7d_ma"))
            fig3.update_layout(height=360, xaxis_title="day", yaxis_title="velocity")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.line_chart(econ_daily.set_index("day")[["supply_proxy", "locked_proxy", "circulating_proxy", "token_velocity"]])

    with c2:
        st.markdown("### Notes")
        st.write(
            "- **Emission** is proxied by ledger `reward`.\n"
            "- **Sinks** are proxied by ledger `tax + fee + market_fee`.\n"
            "- **Locked proxy** is `cumsum(stake - unstake)` (clipped at 0).\n"
            "- **Supply proxy** is `cumsum(net issuance)`.\n"
            "- **Circulating proxy** is `supply - locked`.\n"
            "- **Velocity** is `market volume / circulating proxy`."
        )

        st.markdown("### Flow reconciliation (trades vs ledger)")
        # Compare trade taxes/fees vs ledger sinks (rough)
        tmp = econ_daily.copy()
        tmp["trade_taxes_fees"] = tmp["taxes_token_trades"] + tmp["fees_token_trades"]
        tmp["sinks_minus_trade"] = tmp["sinks"] - tmp["trade_taxes_fees"]
        st.dataframe(tmp[["day","sinks","trade_taxes_fees","sinks_minus_trade"]].tail(14), use_container_width=True)

        st.markdown("### Basic correlations (daily)")
        cols_corr = ["volume_token","token_velocity","net_issuance","sinks","emission","active_traders"]
        cc = econ_daily[cols_corr].copy()
        corr = cc.corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)

    st.divider()
    st.subheader("Resource-level economy (day × resource)")
    st.dataframe(dr.sort_values(["day","volume_token"], ascending=[False, False]).head(50), use_container_width=True)


# -------- Market Quality --------
with tab_mq:
    st.subheader("Slippage / Spread / Volatility / Liquidity stress")

    c1, c2 = st.columns([2, 1])

    with c1:
        if PLOTLY_OK:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mq["day"], y=mq["slippage_bps_vw"], name="slippage_bps_vw"))
            fig.add_trace(go.Scatter(x=mq["day"], y=mq["spread_bps_vw"], name="spread_bps_vw"))
            if "volatility_wavg" in mq.columns:
                fig.add_trace(go.Scatter(x=mq["day"], y=mq["volatility_wavg"], name="volatility_wavg", yaxis="y2"))
            fig.update_layout(
                height=420,
                xaxis_title="day",
                yaxis=dict(title="bps (slippage/spread)"),
                yaxis2=dict(title="volatility", overlaying="y", side="right"),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=mq["day"], y=mq["stress_index"], name="stress_index"))
            fig2.update_layout(height=320, xaxis_title="day", yaxis_title="stress (z-score blend)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.line_chart(mq.set_index("day")[["slippage_bps_vw","spread_bps_vw","stress_index"]])

        st.markdown("### Volatility vs microstructure (scatter)")
        if PLOTLY_OK and ("volatility_wavg" in mq.columns):
            # Use resource-level scatter if available
            if "volatility" in dr.columns:
                sample = dr.dropna(subset=["volatility","slippage_bps_vw","spread_bps_vw"]).copy()
                if len(sample) > 50000:
                    sample = sample.sample(50000, random_state=42)

                fig_sc1 = px.scatter(sample, x="volatility", y="slippage_bps_vw", hover_data=["resource","day","volume_token"])
                fig_sc1.update_layout(height=380, title="Resource-day: slippage vs volatility")
                st.plotly_chart(fig_sc1, use_container_width=True)

                fig_sc2 = px.scatter(sample, x="volatility", y="spread_bps_vw", hover_data=["resource","day","volume_token"])
                fig_sc2.update_layout(height=380, title="Resource-day: spread vs volatility")
                st.plotly_chart(fig_sc2, use_container_width=True)
            else:
                st.info("price_oracle is missing resource-level volatility, scatter skipped.")
        else:
            st.info("Plotly not available or volatility is missing.")

    with c2:
        st.markdown("### Liquidity stress interpretation")
        st.write(
            "Stress index is a simple blended z-score of:\n"
            "- volume-weighted slippage\n"
            "- volume-weighted spread\n"
            "- volume-weighted volatility (if oracle exists)\n\n"
            "Use it as a **trigger**, not as a final diagnosis."
        )

        if "top1_share" in mq.columns:
            st.markdown("### Concentration (Top 1% volume share)")
            if PLOTLY_OK:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=mq["day"], y=mq["top1_share"], name="top1_share"))
                fig_c.update_layout(height=260, xaxis_title="day", yaxis_title="share")
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.line_chart(mq.set_index("day")[["top1_share"]])

        st.markdown("### Correlation matrix (daily)")
        corr_cols = ["volume_token","slippage_bps_vw","spread_bps_vw","volatility_wavg","stress_index","top1_share"]
        corr_cols = [c for c in corr_cols if c in mq.columns]
        if len(corr_cols) >= 2:
            st.dataframe(mq[corr_cols].corr(numeric_only=True), use_container_width=True)
        else:
            st.info("Not enough columns for correlation table.")

    st.divider()
    st.subheader("Resource-level Market Quality (table)")
    show_n = st.slider("Rows to show", 50, 500, 150, step=50)
    st.dataframe(
        dr.sort_values(["day","volume_token"], ascending=[False, False]).head(show_n),
        use_container_width=True
    )


# -------- Raw tables --------
with tab_raw:
    st.subheader("Raw inputs (after filtering)")

    with st.expander("market_trades (filtered)", expanded=False):
        st.dataframe(mt_f.head(300), use_container_width=True)

    with st.expander("token_ledger (filtered)", expanded=False):
        st.dataframe(tl_f.head(300), use_container_width=True)

    if po_f is not None:
        with st.expander("price_oracle (filtered)", expanded=False):
            st.dataframe(po_f.head(300), use_container_width=True)

    st.divider()
    st.subheader("Aggregates")
    with st.expander("econ_daily", expanded=False):
        st.dataframe(econ_daily.tail(200), use_container_width=True)
    with st.expander("mq (daily market quality)", expanded=False):
        st.dataframe(mq.tail(200), use_container_width=True)
    with st.expander("dr (day×resource)", expanded=False):
        st.dataframe(dr.tail(200), use_container_width=True)
