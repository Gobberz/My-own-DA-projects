
#05_StressTesting.py
# pages/05_StressTesting.py
# Anti-fragility / Stress Testing
# Stress test economy robustness against shocks, policy changes, and adversarial conditions.
#
# UI:
# - Shock builder: volatility +X, production -Y, tax +Z, fee +Z2, adversarial pressure
# - Batch scenarios & comparison
# - Recovery curves
#
# Metrics:
# - worst drawdown (liquidity/volume)
# - time-to-recover
# - systemic risk score (concentration + volatility + traders drop + volume drop)
#
# All outputs/comments in English (as requested).

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

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
# Page config
# =========================
st.set_page_config(page_title="Anti-fragility / Stress Testing", layout="wide")
st.title("Anti-fragility / Stress Testing")
st.caption("Stress test economy robustness against shocks, policy changes, and adversarial conditions.")


# =========================
# Defaults (your folders)
# =========================
DEFAULT_MAIN_DIR = r"C:\Users\Shaim\defi game\synthetic_defi_game_data"
DEFAULT_EXTRA_DIR = r"C:\Users\Shaim\defi game\synthetic_data"


# =========================
# Helpers
# =========================
def dt(x):
    return pd.to_datetime(x, errors="coerce")

def floor_day(x):
    return dt(x).dt.floor("D")

def ensure_col(df: pd.DataFrame, candidates: List[str], required: bool = True, name: str = "df") -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"{name}: missing any of {candidates}. Available={list(df.columns)}")
    return None

def safe_num(x):
    return pd.to_numeric(x, errors="coerce")
def num_col(df: pd.DataFrame, candidates: list[str], required: bool = True, name: str = "df") -> pd.Series:
    col = ensure_col(df, candidates, required=required, name=name)
    if col is None:
        return pd.Series([0.0] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

def to_series(x, index=None, name: str = None) -> pd.Series:
    """Force any vector-like input into a pandas Series so downstream .fillna/.replace works."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.Index):
        return pd.Series(x, name=name)
    if isinstance(x, (np.ndarray, list, tuple)):
        return pd.Series(x, index=index, name=name)
    # scalar
    if index is None:
        return pd.Series([x], name=name)
    return pd.Series([x] * len(index), index=index, name=name)

def safe_div(num, den) -> pd.Series:
    """
    SAFE division that always returns a pandas Series (so .fillna works).
    If inputs are arrays, we wrap them into Series.
    """
    num_s = to_series(num)
    den_s = to_series(den, index=num_s.index)
    num_s = safe_num(num_s).astype(float)
    den_s = safe_num(den_s).astype(float)
    out = num_s / den_s.replace(0.0, np.nan)
    return out

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def plot_line(df: pd.DataFrame, x: str, ys: List[str], title: str):
    if df is None or df.empty:
        st.info("No data.")
        return
    for y in ys:
        if y not in df.columns:
            st.warning(f"Missing column: {y}")
            return
    if PLOTLY_OK:
        fig = go.Figure()
        for y in ys:
            fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y))
        fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df.set_index(x)[ys])

def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    if df is None or df.empty:
        return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")

def find_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# =========================
# Load tables
# =========================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=True)
def load_tables(main_dir: str, extra_dir: str) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    # Main required
    for key, fname in [
        ("users", "users.csv"),
        ("sessions", "sessions.csv"),
        ("market_trades", "market_trades.csv"),
        ("token_ledger", "token_ledger.csv"),
    ]:
        p = find_existing([os.path.join(main_dir, fname)])
        if p:
            tables[key] = load_csv(p)

    # Extra / enriched (optional but recommended)
    # You said these exist now as CSVs in synthetic_data:
    optional_files = {
        "price_oracle": ["price_oracle.csv", "price_oracle_daily.csv", "price_oracle_enriched.csv"],
        "econ_events": ["econ_events.csv", "economic_events.csv"],
        "resource_production_daily": ["resource_production_daily.csv"],
    }
    for k, candidates in optional_files.items():
        p = find_existing([os.path.join(extra_dir, c) for c in candidates])
        if p:
            tables[k] = load_csv(p)

    return tables


# =========================
# Baseline panels builder
# =========================
@st.cache_data(show_spinner=True)
def build_baseline_panels(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Builds baseline day×resource and global daily panels:
    - volume, traders, trades, avg slippage/spread, revenue, volatility
    - concentration (top1 share global)
    - velocity proxies from ledger
    """
    # ---- Trades normalize ----
    mt = tables["market_trades"].copy()
    tl = tables["token_ledger"].copy()

    ts_col = ensure_col(mt, ["ts"], True, "market_trades")
    mt[ts_col] = pd.to_datetime(mt[ts_col], errors="coerce")
    mt["day"] = mt[ts_col].dt.floor("D")

    mt["amount_base"] = num_col(mt, ["amount_base"], True, "market_trades").fillna(0.0)
    mt["price"] = num_col(mt, ["price"], True, "market_trades").fillna(0.0)

    mt["fee_amount"] = num_col(mt, ["fee_amount"], False, "market_trades").fillna(0.0)
    mt["tax_amount"] = num_col(mt, ["tax_amount"], False, "market_trades").fillna(0.0)

    mt["slippage_bps"] = num_col(mt, ["slippage_bps"], False, "market_trades").fillna(0.0)
    mt["spread_bps"] = num_col(mt, ["spread_bps"], False, "market_trades").fillna(0.0)

    mt["volume_token"] = (mt["amount_base"] * mt["price"]).fillna(0.0)
    mt["revenue_token"] = (mt["fee_amount"] + mt["tax_amount"]).clip(lower=0.0)

    for c in ["fee_amount", "tax_amount", "slippage_bps", "spread_bps"]:
        if c in mt.columns:
            mt[c] = safe_num(mt[c]).fillna(0.0)
        else:
            mt[c] = 0.0

    mt["revenue_token"] = (mt["fee_amount"] + mt["tax_amount"]).clip(lower=0.0)
    if "resource" in mt.columns:
        mt["resource_key"] = mt["resource"].astype(str).str.upper()
    elif "asset_base" in mt.columns:
        mt["resource_key"] = mt["asset_base"].astype(str).str.upper()
    elif "instrument_exch" in mt.columns:
        # fallback: parse base from instrument like ABCUSDT
        mt["resource_key"] = mt["instrument_exch"].astype(str).str.upper().str.replace(
            r"(USDT|USDC|BUSD|USD|BTC|ETH|EUR|TRY|BRL|GBP)$", "", regex=True)
    else:
        raise KeyError("market_trades: need one of ['resource','asset_base','instrument_exch'] to build resource_key")
    assert "resource_key" in mt.columns and mt["resource_key"].notna().any()

    # --- day×resource ---
    dr = mt.groupby(["day", "resource_key"]).agg(
        trades=("trade_id", "count"),
        traders=("maker_user_id", "nunique"),
        volume_token=("volume_token", "sum"),
        revenue_token=("revenue_token", "sum"),
        avg_slippage=("slippage_bps", "mean"),
        avg_spread=("spread_bps", "mean"),
    ).reset_index()

    # Convert tax_rate to actual ratio tax/volume
    # We recompute using merge with summed tax:
    tax_sum = mt.groupby(["day", "resource_key"])["tax_amount"].sum().reset_index(name="tax_sum")
    fee_sum = mt.groupby(["day", "resource_key"])["fee_amount"].sum().reset_index(name="fee_sum")
    if "tax_rate" in dr.columns:
        dr = dr.drop(columns=["tax_rate"])
    dr = dr.merge(tax_sum, on=["day", "resource_key"], how="left").merge(fee_sum, on=["day","resource_key"], how="left")
    dr["tax_sum"] = dr["tax_sum"].fillna(0.0)
    dr["fee_sum"] = dr["fee_sum"].fillna(0.0)
    dr["tax_rate"] = safe_div(dr["tax_sum"], dr["volume_token"]).fillna(0.0)
    dr["fee_rate"] = safe_div(dr["fee_sum"], dr["volume_token"]).fillna(0.0)

    # --- price oracle merge (optional) ---
    if "price_oracle" in tables and not tables["price_oracle"].empty:
        po = tables["price_oracle"].copy()
        po["day"] = pd.to_datetime(po["day"], errors="coerce").dt.floor("D")

        if "resource" in po.columns:
            po["resource_key"] = po["resource"].astype(str).str.upper()
        elif "asset_base" in po.columns:
            po["resource_key"] = po["asset_base"].astype(str).str.upper()
        else:
            raise KeyError("price_oracle: need 'resource' (or 'asset_base')")

        po = po.rename(columns={"volatility": "day_volatility", "price_token": "price_token"})
        po["day_volatility"] = pd.to_numeric(po.get("day_volatility"), errors="coerce").fillna(0.0)
        po["price_token"] = pd.to_numeric(po.get("price_token"), errors="coerce").fillna(0.0)

        dr = dr.merge(
            po[["day", "resource_key", "day_volatility", "price_token"]],
            on=["day", "resource_key"],
            how="left"
        )
        dr["day_volatility"] = dr["day_volatility"].fillna(0.0)
        dr["price_token"] = dr["price_token"].fillna(0.0)
    else:
        # If price_oracle is missing, fill with defaults
        dr["day_volatility"] = 0.0
        dr["price_token"] = 0.0

    # --- global daily aggregation ---
    g = dr.groupby("day").agg(
        volume_token_global=("volume_token", "sum"),
        revenue_token_global=("revenue_token", "sum"),
        traders_global=("traders", "sum"),
        trades_global=("trades", "sum"),
        slippage_bps_global=("avg_slippage", "mean"),
        spread_bps_global=("avg_spread", "mean"),
        volatility_global=("day_volatility", "mean"),
    ).reset_index()

    # --- concentration proxy: top1 share by user volume (global) ---
    maker = ensure_col(mt, ["maker_user_id", "user_id"], True, "market_trades")
    uv = mt.groupby(["day", maker])["volume_token"].sum().reset_index()
    top1 = uv.sort_values(["day", "volume_token"], ascending=[True, False]).groupby("day").head(1)
    top1 = top1.groupby("day")["volume_token"].sum().reset_index(name="top1_volume_global")
    tot = mt.groupby("day")["volume_token"].sum().reset_index(name="volume_token_global_check")
    conc_g = tot.merge(top1, on="day", how="left")
    conc_g["top1_volume_global"] = conc_g["top1_volume_global"].fillna(0.0)
    conc_g["top1_share_global"] = safe_div(conc_g["top1_volume_global"], conc_g["volume_token_global_check"]).fillna(0.0)

    g = g.merge(conc_g[["day", "top1_share_global"]], on="day", how="left")
    g["top1_share_global"] = g["top1_share_global"].fillna(0.0)

    # --- velocity / issuance proxies from ledger ---
    tl_ts = ensure_col(tl, ["ts"], True, "token_ledger")
    tl[tl_ts] = dt(tl[tl_ts])
    tl["day"] = tl[tl_ts].dt.floor("D")
    tx_type = ensure_col(tl, ["tx_type", "type"], True, "token_ledger")
    amt = ensure_col(tl, ["amount"], True, "token_ledger")
    tl["amount"] = safe_num(tl[amt]).fillna(0.0)
    tl["tx_type_norm"] = tl[tx_type].astype(str).str.lower()

    piv = tl.groupby(["day", "tx_type_norm"])["amount"].sum().reset_index()
    piv = piv.pivot_table(index="day", columns="tx_type_norm", values="amount", aggfunc="sum", fill_value=0.0).reset_index()
    piv.columns = [str(c) for c in piv.columns]

    for c in ["reward", "tax", "fee", "market_fee", "stake", "unstake"]:
        if c not in piv.columns:
            piv[c] = 0.0

    piv["emission"] = piv["reward"].clip(lower=0.0)
    piv["sinks"] = (piv["tax"].abs() + piv["fee"].abs() + piv["market_fee"].abs()).clip(lower=0.0)
    piv["net_issuance"] = piv["emission"] - piv["sinks"]

    piv["supply_proxy"] = piv["net_issuance"].cumsum()
    piv["locked_proxy"] = (piv["stake"] + piv["unstake"]).cumsum()
    piv["circulating_proxy"] = (piv["supply_proxy"] - piv["locked_proxy"]).clip(lower=1e-6)

    g = g.merge(piv[["day", "emission", "sinks", "net_issuance", "circulating_proxy"]], on="day", how="left")
    for c in ["emission", "sinks", "net_issuance", "circulating_proxy"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)

    # Trading velocity = trading volume / circulating proxy
    g["velocity"] = safe_div(g["volume_token_global"], g["circulating_proxy"]).fillna(0.0)

    # Liquidity proxy (simple): volume / (1 + volatility) and penalize spread/slippage
    g["liquidity_proxy"] = g["volume_token_global"] / (1.0 + g["volatility_global"]).replace(0, np.nan)
    g["liquidity_proxy"] = g["liquidity_proxy"].fillna(0.0) / (1.0 + 0.01 * g["spread_bps_global"].fillna(0.0) + 0.01 * g["slippage_bps_global"].fillna(0.0))

    # Clean types
    for c in ["volume_token_global","revenue_token_global","traders_global","trades_global",
              "slippage_bps_global","spread_bps_global","volatility_global",
              "velocity","liquidity_proxy","top1_share_global"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)

    return {"dr": dr, "global": g}


# =========================
# Shock simulation
# =========================
@dataclass
class ShockConfig:
    name: str
    shock_start: pd.Timestamp
    shock_days: int

    vol_mult: float          # e.g. +20% => 1.2
    prod_mult: float         # e.g. -10% => 0.9 (used as volume factor proxy)
    tax_add: float           # add to tax_rate (absolute)
    fee_add: float           # add to fee_rate (absolute)

    adversarial: float       # 0..1 increases concentration + reduces traders
    elasticity_tax: float    # volume response to tax delta
    elasticity_fee: float    # volume response to fee delta

    recovery_halflife: float # days (mean reversion)
    recover_tol: float       # fraction to consider "recovered" (e.g. 0.02 => within 2% of baseline)


def apply_shock_and_recover(
    baseline_dr: pd.DataFrame,
    baseline_g: pd.DataFrame,
    cfg: ShockConfig,
    resources: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Apply a shock to day×resource and global series and then simulate recovery
    via exponential mean reversion to baseline after shock.
    """
    dr = baseline_dr.copy()
    g = baseline_g.copy()
    dr["day"] = dt(dr["day"]).dt.floor("D")
    g["day"] = dt(g["day"]).dt.floor("D")

    if resources is not None and len(resources) > 0:
        dr["is_target"] = dr["resource_key"].isin([r.upper() for r in resources]).astype(int)
    else:
        dr["is_target"] = 1

    start = pd.to_datetime(cfg.shock_start).floor("D")
    end = start + pd.Timedelta(days=int(cfg.shock_days))

    # Shock mask
    dr["in_shock"] = ((dr["day"] >= start) & (dr["day"] < end) & (dr["is_target"] == 1)).astype(int)

    # Apply shock at dr level:
    # - volatility multiplier
    # - production multiplier -> volume proxy
    # - taxes/fees up -> reduce volume via elasticity
    dr["volatility_shocked"] = dr["day_volatility"] * np.where(dr["in_shock"] == 1, cfg.vol_mult, 1.0)

    dr["tax_rate_shocked"] = dr["tax_rate"] + np.where(dr["in_shock"] == 1, cfg.tax_add, 0.0)
    dr["fee_rate_shocked"] = dr["fee_rate"] + np.where(dr["in_shock"] == 1, cfg.fee_add, 0.0)

    d_tax = (dr["tax_rate_shocked"] - dr["tax_rate"]).clip(lower=0.0)
    d_fee = (dr["fee_rate_shocked"] - dr["fee_rate"]).clip(lower=0.0)

    # Volume response (simple demand model):
    vol_factor_tax = np.exp(-cfg.elasticity_tax * d_tax)
    vol_factor_fee = np.exp(-cfg.elasticity_fee * d_fee)

    prod_factor = np.where(dr["in_shock"] == 1, cfg.prod_mult, 1.0)

    dr["volume_shocked_raw"] = dr["volume_token"] * prod_factor * vol_factor_tax * vol_factor_fee

    # Microstructure worsens with volatility
    dr["slippage_shocked_raw"] = dr["avg_slippage"] * (1.0 + 0.8 * (dr["volatility_shocked"] - dr["day_volatility"]).clip(lower=0.0))
    dr["spread_shocked_raw"] = dr["avg_spread"] * (1.0 + 0.8 * (dr["volatility_shocked"] - dr["day_volatility"]).clip(lower=0.0))

    # Revenue scales with volume and new rates
    dr["revenue_shocked_raw"] = dr["volume_shocked_raw"] * (dr["tax_rate_shocked"] + dr["fee_rate_shocked"]).clip(lower=0.0)

    # Traders drop with adversarial pressure + cost shock
    # (pure proxy; better later with real behavior model)
    cost_shock = (d_tax + d_fee).clip(lower=0.0)
    trader_factor = 1.0 - cfg.adversarial * 0.25 - 0.5 * cost_shock
    trader_factor = trader_factor.clip(lower=0.3, upper=1.0)
    dr["traders_shocked_raw"] = (dr["traders"] * trader_factor).round().astype(int)

    # Aggregate to global daily shocked raw
    g_sh = dr.groupby("day").agg(
        volume_token=("volume_shocked_raw", "sum"),
        revenue_token=("revenue_shocked_raw", "sum"),
        traders=("traders_shocked_raw", "sum"),
        slippage=("slippage_shocked_raw", "mean"),
        spread=("spread_shocked_raw", "mean"),
        volatility=("volatility_shocked", "mean"),
    ).reset_index()

    g2 = g[["day","volume_token_global","revenue_token_global","traders_global","slippage_bps_global","spread_bps_global","volatility_global",
            "velocity","liquidity_proxy","top1_share_global","circulating_proxy","emission","sinks","net_issuance"]].copy()

    g2 = g2.merge(g_sh, on="day", how="left")

    # Fill missing (days outside dr coverage)
    for c in ["volume_token","revenue_token","traders","slippage","spread","volatility"]:
        g2[c] = pd.to_numeric(g2[c], errors="coerce").fillna(0.0)

    # Concentration increases under adversarial
    # top1_share_shocked = baseline + adv * bump * (in shock window)
    g2["in_shock"] = ((g2["day"] >= start) & (g2["day"] < end)).astype(int)
    g2["top1_share_shocked_raw"] = (g2["top1_share_global"] + g2["in_shock"] * cfg.adversarial * 0.10).clip(0, 1)

    # Issuance/velocity/liquidity recompute using shocked volume + baseline circulating
    g2["velocity_shocked_raw"] = safe_div(g2["volume_token"], g2["circulating_proxy"]).fillna(0.0)
    g2["liquidity_shocked_raw"] = (g2["volume_token"] / (1.0 + g2["volatility"]).replace(0, np.nan)).fillna(0.0)
    g2["liquidity_shocked_raw"] = g2["liquidity_shocked_raw"] / (1.0 + 0.01*g2["spread"].fillna(0.0) + 0.01*g2["slippage"].fillna(0.0))

    # Recovery: exponential mean reversion after shock end
    # value(t) = baseline(t) + (shock_raw(t) - baseline(t)) * exp(-dt/tau)
    # tau from halflife: halflife = tau * ln(2) => tau = halflife/ln(2)
    tau = max(cfg.recovery_halflife, 0.01) / np.log(2.0)

    def recover_series(day_series: pd.Series, base: pd.Series, shocked: pd.Series) -> pd.Series:
        day_series = dt(day_series).dt.floor("D")
        dt_days = (day_series - end).dt.total_seconds() / 86400.0
        dt_days = dt_days.clip(lower=0.0)
        decay = np.exp(-dt_days / tau)
        return base + (shocked - base) * decay

    # Build final scenario
    out = pd.DataFrame({"day": g2["day"]})

    out["volume_baseline"] = g2["volume_token_global"]
    out["volume_shocked"] = recover_series(g2["day"], g2["volume_token_global"], g2["volume_token"])

    out["liquidity_baseline"] = g2["liquidity_proxy"]
    out["liquidity_shocked"] = recover_series(g2["day"], g2["liquidity_proxy"], g2["liquidity_shocked_raw"])

    out["velocity_baseline"] = g2["velocity"]
    out["velocity_shocked"] = recover_series(g2["day"], g2["velocity"], g2["velocity_shocked_raw"])

    out["traders_baseline"] = g2["traders_global"]
    out["traders_shocked"] = recover_series(g2["day"], g2["traders_global"], g2["traders"])

    out["volatility_baseline"] = g2["volatility_global"]
    out["volatility_shocked"] = recover_series(g2["day"], g2["volatility_global"], g2["volatility"])

    out["slippage_baseline"] = g2["slippage_bps_global"]
    out["slippage_shocked"] = recover_series(g2["day"], g2["slippage_bps_global"], g2["slippage"])

    out["spread_baseline"] = g2["spread_bps_global"]
    out["spread_shocked"] = recover_series(g2["day"], g2["spread_bps_global"], g2["spread"])

    out["top1_share_baseline"] = g2["top1_share_global"]
    out["top1_share_shocked"] = recover_series(g2["day"], g2["top1_share_global"], g2["top1_share_shocked_raw"]).clip(0, 1)

    # Systemic risk score (0..100): combine z-scores (higher = riskier)
    # Drivers:
    # - concentration up
    # - volatility up
    # - traders down
    # - volume down
    vol_drop = safe_div(out["volume_baseline"] - out["volume_shocked"], out["volume_baseline"]).fillna(0.0).clip(lower=0.0)
    tr_drop = safe_div(out["traders_baseline"] - out["traders_shocked"], out["traders_baseline"]).fillna(0.0).clip(lower=0.0)
    conc_up = (out["top1_share_shocked"] - out["top1_share_baseline"]).clip(lower=0.0)
    vola_up = (out["volatility_shocked"] - out["volatility_baseline"]).clip(lower=0.0)

    risk_raw = (
        0.35 * zscore(conc_up) +
        0.25 * zscore(vola_up) +
        0.25 * zscore(tr_drop) +
        0.15 * zscore(vol_drop)
    )
    risk_score = 50 + 15 * risk_raw  # center around 50
    out["systemic_risk_score"] = risk_score.clip(0, 100)

    # Summary metrics
    # Worst drawdown (min of shocked/baseline - 1)
    out["volume_rel"] = safe_div(out["volume_shocked"], out["volume_baseline"]).fillna(0.0)
    out["liquidity_rel"] = safe_div(out["liquidity_shocked"], out["liquidity_baseline"]).fillna(0.0)

    return {"series": out}


def compute_recovery_metrics(series: pd.DataFrame, shock_end: pd.Timestamp, tol: float) -> Dict[str, float]:
    """
    Worst drawdown & time-to-recover after shock end.
    tol: recovery tolerance (within tol of baseline, relative)
    """
    df = series.copy()
    df = df.sort_values("day")
    df["day"] = dt(df["day"]).dt.floor("D")

    after = df[df["day"] >= shock_end].copy()
    if after.empty:
        return {"worst_dd_volume": np.nan, "worst_dd_liquidity": np.nan, "ttr_volume_days": np.nan, "ttr_liquidity_days": np.nan}

    # drawdowns
    rel_v = safe_div(df["volume_shocked"], df["volume_baseline"]).fillna(0.0)
    rel_l = safe_div(df["liquidity_shocked"], df["liquidity_baseline"]).fillna(0.0)

    worst_dd_volume = float(1.0 - rel_v.min())  # e.g. 0.3 = -30%
    worst_dd_liquidity = float(1.0 - rel_l.min())

    # time to recover: first day after shock_end when shocked >= baseline*(1 - tol)
    target_v = df["volume_baseline"] * (1.0 - tol)
    target_l = df["liquidity_baseline"] * (1.0 - tol)

    rec_v = after[after["volume_shocked"] >= target_v.loc[after.index].values]
    rec_l = after[after["liquidity_shocked"] >= target_l.loc[after.index].values]

    ttr_v = np.nan
    ttr_l = np.nan
    if not rec_v.empty:
        ttr_v = float((rec_v["day"].iloc[0] - shock_end).days)
    if not rec_l.empty:
        ttr_l = float((rec_l["day"].iloc[0] - shock_end).days)

    return {
        "worst_dd_volume": worst_dd_volume,
        "worst_dd_liquidity": worst_dd_liquidity,
        "ttr_volume_days": ttr_v,
        "ttr_liquidity_days": ttr_l,
    }


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Data folders")
    main_dir = st.text_input("Main data folder", value=DEFAULT_MAIN_DIR)
    extra_dir = st.text_input("Extra data folder", value=DEFAULT_EXTRA_DIR)

    st.divider()
    st.header("Shock window")
    shock_start = st.date_input("Shock start day", value=None)
    shock_days = st.slider("Shock duration (days)", 1, 60, 7, 1)

    st.divider()
    st.header("Shock builder")
    vol_plus = st.slider("Volatility shock +X%", 0, 200, 20, 5)
    prod_minus = st.slider("Production shock -Y% (volume proxy)", 0, 90, 10, 5)
    tax_add_bps = st.slider("Tax +Z bps (absolute)", 0, 500, 50, 10)
    fee_add_bps = st.slider("Fee +Z2 bps (absolute)", 0, 500, 20, 10)
    adversarial = st.slider("Adversarial pressure (0..1)", 0.0, 1.0, 0.3, 0.05)

    st.divider()
    st.header("Behavior model")
    elasticity_tax = st.slider("Tax elasticity", 0.0, 200.0, 60.0, 5.0)
    elasticity_fee = st.slider("Fee elasticity", 0.0, 200.0, 30.0, 5.0)

    st.divider()
    st.header("Recovery")
    recovery_halflife = st.slider("Recovery half-life (days)", 1.0, 60.0, 14.0, 1.0)
    recover_tol = st.slider("Recovery tolerance (within X%)", 0.5, 10.0, 2.0, 0.5) / 100.0

    st.divider()
    st.header("Scenarios")
    run_batch = st.checkbox("Run batch scenarios", value=True)

    st.divider()
    show_debug = st.checkbox("Show debug tables", value=False)


# =========================
# Load & baseline build with progress
# =========================
tables = load_tables(main_dir, extra_dir)

need = ["market_trades", "token_ledger"]
missing = [k for k in need if k not in tables]
if missing:
    st.error(f"Missing required tables: {missing}. Check main folder: {main_dir}")
    st.stop()

# Determine shock_start default from baseline if not provided
with st.status("Building baseline panels...", expanded=True) as status:
    prog = st.progress(0)
    prog.progress(5, text="Loading & normalizing inputs...")

    panels = build_baseline_panels(tables)

    prog.progress(75, text="Baseline panels created. Finalizing...")
    status.update(state="complete", label="Baseline panels ready")
    prog.progress(100)

dr = panels["dr"]
g = panels["global"]

if show_debug:
    st.subheader("Baseline preview")
    st.write("day×resource (head)", dr.head(10))
    st.write("global daily (head)", g.head(10))

# pick shock_start default
g_min = dt(g["day"]).min()
g_max = dt(g["day"]).max()
if shock_start is None:
    # default: middle-ish
    default_day = (g_min + (g_max - g_min) / 2) if pd.notna(g_min) and pd.notna(g_max) else pd.Timestamp.today().floor("D")
    shock_start_ts = pd.to_datetime(default_day).floor("D")
else:
    shock_start_ts = pd.to_datetime(shock_start).floor("D")


# =========================
# Resources selector
# =========================
st.subheader("Shock scope (resource-level)")
resources_all = sorted(dr["resource_key"].astype(str).unique().tolist())
col1, col2 = st.columns([2, 3])
with col1:
    target_mode = st.radio("Scope", ["All resources", "Selected resources"], horizontal=True)
with col2:
    if target_mode == "Selected resources":
        target_resources = st.multiselect("Resources", resources_all, default=resources_all[: min(5, len(resources_all))])
    else:
        target_resources = []

# =========================
# Run single scenario
# =========================
cfg = ShockConfig(
    name="Custom",
    shock_start=shock_start_ts,
    shock_days=int(shock_days),
    vol_mult=1.0 + vol_plus / 100.0,
    prod_mult=1.0 - prod_minus / 100.0,
    tax_add=tax_add_bps / 10000.0,
    fee_add=fee_add_bps / 10000.0,
    adversarial=float(adversarial),
    elasticity_tax=float(elasticity_tax),
    elasticity_fee=float(elasticity_fee),
    recovery_halflife=float(recovery_halflife),
    recover_tol=float(recover_tol),
)

run = st.button("Run stress test", type="primary")

if run:
    with st.status("Simulating shock + recovery...", expanded=False) as status:
        prog = st.progress(0)
        prog.progress(10, text="Applying shock to baseline day×resource...")
        sim = apply_shock_and_recover(dr, g, cfg, resources=target_resources if target_mode == "Selected resources" else None)
        prog.progress(70, text="Computing metrics (drawdown, time-to-recover, systemic risk)...")

        series = sim["series"].copy()
        shock_end = cfg.shock_start + pd.Timedelta(days=cfg.shock_days)
        metrics = compute_recovery_metrics(series, shock_end, cfg.recover_tol)

        prog.progress(100, text="Done")
        status.update(state="complete", label="Simulation complete")

    st.subheader("Key metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Worst DD (Volume)", f"{metrics['worst_dd_volume']*100:.1f}%")
    c2.metric("Worst DD (Liquidity)", f"{metrics['worst_dd_liquidity']*100:.1f}%")
    c3.metric("Time-to-recover Volume", f"{metrics['ttr_volume_days']:.0f}d" if np.isfinite(metrics["ttr_volume_days"]) else "NA")
    c4.metric("Time-to-recover Liquidity", f"{metrics['ttr_liquidity_days']:.0f}d" if np.isfinite(metrics["ttr_liquidity_days"]) else "NA")
    c5.metric("Peak Systemic Risk", f"{series['systemic_risk_score'].max():.0f}/100")

    st.subheader("Recovery curves (baseline vs shocked)")
    plot_line(series, "day", ["volume_baseline", "volume_shocked"], "Volume recovery")
    plot_line(series, "day", ["liquidity_baseline", "liquidity_shocked"], "Liquidity recovery")
    plot_line(series, "day", ["velocity_baseline", "velocity_shocked"], "Velocity recovery")
    plot_line(series, "day", ["traders_baseline", "traders_shocked"], "Traders recovery")
    plot_line(series, "day", ["top1_share_baseline", "top1_share_shocked"], "Concentration (top1 share) recovery")
    plot_line(series, "day", ["systemic_risk_score"], "Systemic Risk Score")

    st.markdown("**Download:**")
    download_df(series, "stress_test_series.csv", "Download scenario series CSV")


# =========================
# Batch scenarios
# =========================
if run_batch:
    st.subheader("Batch scenarios (compare)")
    st.caption("Run multiple presets and compare worst drawdown / recovery time / peak systemic risk.")

    presets = [
        ("Mild shock", dict(vol_plus=10, prod_minus=5, tax_bps=20, fee_bps=10, adv=0.1, et=40, ef=20, hl=10)),
        ("Severe shock", dict(vol_plus=50, prod_minus=20, tax_bps=80, fee_bps=40, adv=0.4, et=60, ef=30, hl=18)),
        ("Adversarial attack", dict(vol_plus=30, prod_minus=10, tax_bps=30, fee_bps=20, adv=0.9, et=40, ef=20, hl=25)),
        ("Policy-heavy (tax)", dict(vol_plus=10, prod_minus=5, tax_bps=150, fee_bps=20, adv=0.2, et=80, ef=20, hl=20)),
    ]

    if st.button("Run batch"):
        rows = []
        with st.status("Running batch scenarios...", expanded=False) as status:
            prog = st.progress(0)
            for i, (name, p) in enumerate(presets, start=1):
                prog.progress(int(100 * (i - 1) / len(presets)), text=f"Running: {name}")
                cfg_i = ShockConfig(
                    name=name,
                    shock_start=shock_start_ts,
                    shock_days=int(shock_days),
                    vol_mult=1.0 + p["vol_plus"]/100.0,
                    prod_mult=1.0 - p["prod_minus"]/100.0,
                    tax_add=p["tax_bps"]/10000.0,
                    fee_add=p["fee_bps"]/10000.0,
                    adversarial=float(p["adv"]),
                    elasticity_tax=float(p["et"]),
                    elasticity_fee=float(p["ef"]),
                    recovery_halflife=float(p["hl"]),
                    recover_tol=float(recover_tol),
                )
                sim = apply_shock_and_recover(dr, g, cfg_i, resources=target_resources if target_mode == "Selected resources" else None)
                series = sim["series"]
                shock_end = cfg_i.shock_start + pd.Timedelta(days=cfg_i.shock_days)
                m = compute_recovery_metrics(series, shock_end, cfg_i.recover_tol)

                rows.append({
                    "scenario": name,
                    "worst_dd_volume_pct": 100*m["worst_dd_volume"],
                    "worst_dd_liquidity_pct": 100*m["worst_dd_liquidity"],
                    "ttr_volume_days": m["ttr_volume_days"],
                    "ttr_liquidity_days": m["ttr_liquidity_days"],
                    "peak_systemic_risk": float(series["systemic_risk_score"].max()),
                })

            prog.progress(100, text="Done")
            status.update(state="complete", label="Batch done")

        res = pd.DataFrame(rows).sort_values("peak_systemic_risk", ascending=False)
        st.dataframe(res, use_container_width=True)
        download_df(res, "stress_test_batch_summary.csv", "Download batch summary CSV")

        # Simple bar chart
        if PLOTLY_OK and not res.empty:
            fig = px.bar(res, x="scenario", y="peak_systemic_risk", title="Peak systemic risk by scenario")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
