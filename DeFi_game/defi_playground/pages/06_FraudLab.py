# pages/06_FraudLab.py
# Fraud Lab: wash trading, bots, multi-account scoring
#
# This page is heuristic / "risk scoring" oriented:
# - Wash trading signals (self-trades, ping-pong pairs, concentrated counterparties)
# - Bot signals (high frequency, regularity, 24/7 activity, short sessions, fast trading)
# - Multi-account signals (shared fingerprint groups, signup proximity, shared referrers, behavior similarity proxies)
#
# Notes:
# - These are NOT proofs of fraud. Treat as triage / investigation queue.
# - Replace heuristics with real labels / model over time.
#
# Data expected (based on your synthetic generator):
# - users.csv: user_id, created_at, country, timezone, acq_channel, campaign_id, ad_group, device_os, client, app_version,
#             wallet_type, chain_pref, referrer_user_id, is_kyc, player_segment, risk_profile, whale_flag, initial_deposit_usd
# - sessions.csv: session_id, user_id, start_ts, end_ts, session_len_sec, entry_point, net_latency_ms, crash_flag
# - market_trades.csv: trade_id, ts, order_id, maker_user_id, taker_user_id, asset_base, asset_quote,
#                      amount_base, price, fee_amount, tax_amount, slippage_bps, spread_bps, bull_bear_regime, user_7d_volume_bucket
# - token_ledger.csv: ledger_id, ts, user_id, tx_type, token_symbol, amount, chain, gas_fee_token, tx_hash

from __future__ import annotations

import os
import math
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


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Fraud Lab", layout="wide")
st.title("Fraud Lab")
st.caption("Heuristic detection & scoring for wash trading, bots, and multi-account clusters. Use as investigation queue.")


# -------------------------
# Defaults (your folders)
# -------------------------
DEFAULT_MAIN_DIR = r"C:\Users\Shaim\defi game\synthetic_defi_game_data"
DEFAULT_EXTRA_DIR = r"C:\Users\Shaim\defi game\synthetic_data"


# =========================
# Helpers
# =========================
def dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def floor_day(s) -> pd.Series:
    return dt(s).dt.floor("D")

def ensure_col(df: pd.DataFrame, candidates: List[str], required: bool = True, name: str = "df") -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"{name}: missing any of {candidates}. Available={list(df.columns)}")
    return None

def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def rank_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Percentile rank in [0..1]. If ascending=True then larger values get larger ranks.
    """
    x = pd.to_numeric(s, errors="coerce")
    if x.isna().all():
        return pd.Series(np.zeros(len(x)), index=x.index)
    return x.rank(pct=True, ascending=ascending).fillna(0.0)

def inv_rank_pct(s: pd.Series) -> pd.Series:
    """
    Inverted percentile rank: small values => large score.
    """
    return 1.0 - rank_pct(s, ascending=True)

def human_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def human_float(x, k=4) -> str:
    try:
        return f"{float(x):,.{k}f}"
    except Exception:
        return str(x)

def plot_hist(df: pd.DataFrame, col: str, title: str):
    if df is None or df.empty or col not in df.columns:
        st.info("No data to plot.")
        return
    if PLOTLY_OK:
        fig = px.histogram(df, x=col, nbins=50, title=title)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df[col].value_counts().head(50))

def plot_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str], title: str):
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        st.info("No data to plot.")
        return
    if PLOTLY_OK:
        fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None, title=title, opacity=0.5)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(df[[x, y]].head(200))

def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    if df is None or df.empty:
        return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


# =========================
# Loading tables
# =========================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def find_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=True)
def load_core_tables(main_dir: str) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    p_users = find_existing([os.path.join(main_dir, "users.csv")])
    p_sessions = find_existing([os.path.join(main_dir, "sessions.csv")])
    p_trades = find_existing([os.path.join(main_dir, "market_trades.csv")])
    p_ledger = find_existing([os.path.join(main_dir, "token_ledger.csv")])

    if p_users: tables["users"] = load_csv(p_users)
    if p_sessions: tables["sessions"] = load_csv(p_sessions)
    if p_trades: tables["market_trades"] = load_csv(p_trades)
    if p_ledger: tables["token_ledger"] = load_csv(p_ledger)

    return tables


# =========================
# Feature Engineering
# =========================
@st.cache_data(show_spinner=True)
def build_user_features(
    users: pd.DataFrame,
    sessions: pd.DataFrame,
    trades: pd.DataFrame,
    ledger: pd.DataFrame,
    pingpong_seconds: int = 300,
    fast_trade_seconds: int = 60
) -> pd.DataFrame:
    """
    Produce user-level features used for risk scoring.
    Avoid heavy Python loops; rely on groupby+diff aggregations.

    Output columns include:
    - trade features: trades_cnt, volume_token, self_trade_share, pingpong_share, cp_hhi, unique_cps, unique_assets,
      median_inter_trade_sec, frac_fast_trades
    - session features: sessions_cnt, active_days, mean_sessions_per_day, avg_session_len, cv_inter_session, hour_coverage, night_share
    - ledger features: tx_cnt, reward_sum, tax_sum, fee_sum, claim_cnt, tx_per_day_mean
    - multiacc metadata: fingerprint, fingerprint_size, referrer_group_size, signup_span_days_in_fp
    """
    u = users.copy()
    s = sessions.copy()
    mt = trades.copy()
    tl = ledger.copy()

    # --- users ---
    uid = ensure_col(u, ["user_id", "id"], True, "users")
    created = ensure_col(u, ["created_at", "signup_ts", "registered_at"], True, "users")
    u[created] = dt(u[created])
    u["created_day"] = u[created].dt.floor("D")

    # Build a "device fingerprint" (string) as a proxy for shared identity.
    fp_fields = [
        ensure_col(u, ["device_os"], required=False, name="users"),
        ensure_col(u, ["client"], required=False, name="users"),
        ensure_col(u, ["app_version"], required=False, name="users"),
        ensure_col(u, ["timezone"], required=False, name="users"),
        ensure_col(u, ["country"], required=False, name="users"),
        ensure_col(u, ["wallet_type"], required=False, name="users"),
        ensure_col(u, ["chain_pref"], required=False, name="users"),
        ensure_col(u, ["ad_group"], required=False, name="users"),
    ]
    fp_fields = [c for c in fp_fields if c is not None]

    def _norm_str(x):
        return x.astype(str).fillna("NA").str.strip().str.lower()

    if fp_fields:
        fp = _norm_str(u[fp_fields[0]])
        for c in fp_fields[1:]:
            fp = fp + "|" + _norm_str(u[c])
        u["fingerprint"] = fp
    else:
        u["fingerprint"] = "na"

    # fingerprint size
    fp_size = u.groupby("fingerprint")[uid].transform("count")
    u["fingerprint_size"] = fp_size.astype(int)

    # referrer group size (multiacc / farming rings often share referrers)
    ref = ensure_col(u, ["referrer_user_id"], required=False, name="users")
    if ref is not None:
        u["referrer_group_size"] = u.groupby(ref)[uid].transform("count").fillna(1).astype(int)
    else:
        u["referrer_group_size"] = 1

    # signup span within fingerprint group (smaller span => more suspicious)
    fp_min = u.groupby("fingerprint")[created].transform("min")
    fp_max = u.groupby("fingerprint")[created].transform("max")
    u["signup_span_days_in_fp"] = (fp_max - fp_min).dt.total_seconds() / 86400.0
    u["signup_span_days_in_fp"] = u["signup_span_days_in_fp"].fillna(0.0)

    # kyc
    is_kyc = ensure_col(u, ["is_kyc"], required=False, name="users")
    if is_kyc is not None:
        u["is_kyc"] = u[is_kyc].astype(int)
    else:
        u["is_kyc"] = 0

    # --- sessions ---
    s_uid = ensure_col(s, ["user_id"], True, "sessions")
    s_start = ensure_col(s, ["start_ts", "ts"], True, "sessions")
    s_end = ensure_col(s, ["end_ts"], required=False, name="sessions")
    s_len = ensure_col(s, ["session_len_sec", "duration_sec", "len_sec"], required=False, name="sessions")

    s[s_start] = dt(s[s_start])
    s["day"] = s[s_start].dt.floor("D")
    s["hour"] = s[s_start].dt.hour

    if s_len is not None:
        s["session_len_sec"] = safe_num(s[s_len]).fillna(0.0)
    elif s_end is not None:
        s[s_end] = dt(s[s_end])
        s["session_len_sec"] = (s[s_end] - s[s_start]).dt.total_seconds().clip(lower=0).fillna(0.0)
    else:
        s["session_len_sec"] = 0.0

    # sessions per user-day
    ud_sess = s.groupby([s_uid, "day"]).size().rename("sessions_per_day").reset_index()
    sess_stats = ud_sess.groupby(s_uid).agg(
        active_days=("day", "nunique"),
        sessions_cnt=("sessions_per_day", "sum"),
        mean_sessions_per_day=("sessions_per_day", "mean"),
        p95_sessions_per_day=("sessions_per_day", lambda x: float(np.nanpercentile(x, 95))),
        max_sessions_per_day=("sessions_per_day", "max"),
    ).reset_index().rename(columns={s_uid: uid})

    # session length stats
    sess_len_stats = s.groupby(s_uid)["session_len_sec"].agg(
        avg_session_len="mean",
        med_session_len="median",
        p90_session_len=lambda x: float(np.nanpercentile(x, 90)),
        std_session_len="std",
    ).reset_index().rename(columns={s_uid: uid})

    # inter-session regularity (bots tend to be too regular)
    s_sorted = s.sort_values([s_uid, s_start]).copy()
    s_sorted["dt_prev_sec"] = s_sorted.groupby(s_uid)[s_start].diff().dt.total_seconds()
    inter = s_sorted.groupby(s_uid)["dt_prev_sec"].agg(
        med_inter_session_sec="median",
        mean_inter_session_sec="mean",
        std_inter_session_sec="std",
    ).reset_index().rename(columns={s_uid: uid})
    inter["cv_inter_session"] = inter["std_inter_session_sec"] / inter["mean_inter_session_sec"].replace(0, np.nan)
    inter["cv_inter_session"] = inter["cv_inter_session"].fillna(np.nan)

    # hour coverage and night share
    hour_cov = s.groupby(s_uid)["hour"].nunique().reset_index().rename(columns={s_uid: uid, "hour": "hours_active"})
    hour_cov["hour_coverage"] = hour_cov["hours_active"] / 24.0

    night = s.assign(is_night=(s["hour"].between(0, 5)).astype(int)) \
             .groupby(s_uid)["is_night"].mean().reset_index().rename(columns={s_uid: uid, "is_night": "night_share"})

    # crash/latency
    crash = ensure_col(s, ["crash_flag"], required=False, name="sessions")
    lat = ensure_col(s, ["net_latency_ms"], required=False, name="sessions")
    if crash is not None:
        s["crash_flag"] = safe_num(s[crash]).fillna(0.0)
        crash_stats = s.groupby(s_uid)["crash_flag"].mean().reset_index().rename(columns={s_uid: uid, "crash_flag": "crash_rate"})
    else:
        crash_stats = pd.DataFrame({uid: u[uid], "crash_rate": 0.0}).drop_duplicates()

    if lat is not None:
        s["net_latency_ms"] = safe_num(s[lat]).fillna(np.nan)
        lat_stats = s.groupby(s_uid)["net_latency_ms"].agg(
            latency_mean="mean",
            latency_std="std"
        ).reset_index().rename(columns={s_uid: uid})
    else:
        lat_stats = pd.DataFrame({uid: u[uid], "latency_mean": np.nan, "latency_std": np.nan}).drop_duplicates()

    # --- trades ---
    mt_ts = ensure_col(mt, ["ts"], True, "market_trades")
    mt_maker = ensure_col(mt, ["maker_user_id"], True, "market_trades")
    mt_taker = ensure_col(mt, ["taker_user_id"], True, "market_trades")
    mt_asset = ensure_col(mt, ["asset_base", "resource"], True, "market_trades")
    mt_amt = ensure_col(mt, ["amount_base"], True, "market_trades")
    mt_px = ensure_col(mt, ["price"], True, "market_trades")

    mt[mt_ts] = dt(mt[mt_ts])
    mt["day"] = mt[mt_ts].dt.floor("D")
    mt["asset_base"] = mt[mt_asset].astype(str).str.upper()
    mt["amount_base"] = safe_num(mt[mt_amt]).fillna(0.0)
    mt["price"] = safe_num(mt[mt_px]).fillna(0.0)
    mt["volume_token"] = mt["amount_base"] * mt["price"]

    for c in ["fee_amount", "tax_amount", "slippage_bps", "spread_bps"]:
        if c in mt.columns:
            mt[c] = safe_num(mt[c]).fillna(0.0)

    # self-trades
    mt["is_self_trade"] = (mt[mt_maker] == mt[mt_taker]).astype(int)

    # pair_key for ping-pong detection (order-independent)
    a = mt[mt_maker].astype(str)
    b = mt[mt_taker].astype(str)
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    mt["pair_key"] = pd.Series(lo, index=mt.index) + "|" + pd.Series(hi, index=mt.index)

    # ping-pong trades: repeated trades within pingpong_seconds for same pair_key + asset
    mt_sorted = mt.sort_values(["pair_key", "asset_base", mt_ts]).copy()
    mt_sorted["dt_pair_sec"] = mt_sorted.groupby(["pair_key", "asset_base"])[mt_ts].diff().dt.total_seconds()
    mt_sorted["is_pingpong"] = ((mt_sorted["dt_pair_sec"] <= pingpong_seconds).fillna(False)).astype(int)

    # long format: attribute pair/pingpong stats to BOTH maker and taker
    mt_long = pd.concat([
        mt_sorted[[mt_ts, "day", "asset_base", "volume_token", "is_self_trade", "is_pingpong", mt_maker, mt_taker,
                   "tax_amount" if "tax_amount" in mt_sorted.columns else None,
                   "fee_amount" if "fee_amount" in mt_sorted.columns else None,
                   "slippage_bps" if "slippage_bps" in mt_sorted.columns else None,
                   "spread_bps" if "spread_bps" in mt_sorted.columns else None,
                   ]].rename(columns={mt_maker: "user_id", mt_taker: "counterparty"}),
        mt_sorted[[mt_ts, "day", "asset_base", "volume_token", "is_self_trade", "is_pingpong", mt_taker, mt_maker,
                   "tax_amount" if "tax_amount" in mt_sorted.columns else None,
                   "fee_amount" if "fee_amount" in mt_sorted.columns else None,
                   "slippage_bps" if "slippage_bps" in mt_sorted.columns else None,
                   "spread_bps" if "spread_bps" in mt_sorted.columns else None,
                   ]].rename(columns={mt_taker: "user_id", mt_maker: "counterparty"})
    ], ignore_index=True)

    # drop None columns introduced above
    mt_long = mt_long.loc[:, [c for c in mt_long.columns if c is not None]].copy()

    # per user trade stats
    t_stats = mt_long.groupby("user_id").agg(
        trades_cnt=("volume_token", "size"),
        volume_token=("volume_token", "sum"),
        unique_assets=("asset_base", "nunique"),
        unique_cps=("counterparty", "nunique"),
        self_trade_cnt=("is_self_trade", "sum"),
        pingpong_cnt=("is_pingpong", "sum"),
        avg_slippage_bps=("slippage_bps", "mean") if "slippage_bps" in mt_long.columns else ("volume_token", "mean"),
        avg_spread_bps=("spread_bps", "mean") if "spread_bps" in mt_long.columns else ("volume_token", "mean"),
        tax_total=("tax_amount", "sum") if "tax_amount" in mt_long.columns else ("volume_token", "sum"),
        fee_total=("fee_amount", "sum") if "fee_amount" in mt_long.columns else ("volume_token", "sum"),
    ).reset_index().rename(columns={"user_id": uid})

    if "tax_amount" not in mt_long.columns:
        t_stats["tax_total"] = 0.0
    if "fee_amount" not in mt_long.columns:
        t_stats["fee_total"] = 0.0
    if "slippage_bps" not in mt_long.columns:
        t_stats["avg_slippage_bps"] = 0.0
    if "spread_bps" not in mt_long.columns:
        t_stats["avg_spread_bps"] = 0.0

    t_stats["self_trade_share"] = t_stats["self_trade_cnt"] / t_stats["trades_cnt"].replace(0, np.nan)
    t_stats["pingpong_share"] = t_stats["pingpong_cnt"] / t_stats["trades_cnt"].replace(0, np.nan)
    t_stats["tax_rate_eff"] = t_stats["tax_total"] / t_stats["volume_token"].replace(0, np.nan)
    t_stats["fee_rate_eff"] = t_stats["fee_total"] / t_stats["volume_token"].replace(0, np.nan)
    t_stats = t_stats.fillna({"self_trade_share": 0.0, "pingpong_share": 0.0, "tax_rate_eff": 0.0, "fee_rate_eff": 0.0})

    # counterparty concentration (HHI) on volume shares per user
    uc = mt_long.groupby(["user_id", "counterparty"], as_index=False)["volume_token"].sum()
    tot = uc.groupby("user_id", as_index=False)["volume_token"].sum().rename(columns={"volume_token": "tot"})
    uc = uc.merge(tot, on="user_id", how="left")
    uc["share"] = uc["volume_token"] / uc["tot"].replace(0, np.nan)
    hhi = uc.groupby("user_id")["share"].apply(lambda x: float(np.nansum((x.fillna(0.0).to_numpy()) ** 2))).reset_index()
    hhi = hhi.rename(columns={"user_id": uid, "share": "cp_hhi"})
    t_stats = t_stats.merge(hhi, on=uid, how="left")
    t_stats["cp_hhi"] = t_stats["cp_hhi"].fillna(0.0)

    # inter-trade stats (per user)
    mt_long_sorted = mt_long.sort_values(["user_id", mt_ts]).copy()
    mt_long_sorted["dt_prev_trade_sec"] = mt_long_sorted.groupby("user_id")[mt_ts].diff().dt.total_seconds()
    itd = mt_long_sorted.groupby("user_id")["dt_prev_trade_sec"].agg(
        med_inter_trade_sec="median",
        p10_inter_trade_sec=lambda x: float(np.nanpercentile(x.dropna(), 10)) if x.dropna().size else np.nan,
    ).reset_index().rename(columns={"user_id": uid})
    mt_long_sorted["is_fast_trade"] = ((mt_long_sorted["dt_prev_trade_sec"] <= fast_trade_seconds).fillna(False)).astype(int)
    frac_fast = mt_long_sorted.groupby("user_id")["is_fast_trade"].mean().reset_index().rename(columns={"user_id": uid, "is_fast_trade": "frac_fast_trades"})
    t_stats = t_stats.merge(itd, on=uid, how="left").merge(frac_fast, on=uid, how="left")
    t_stats["frac_fast_trades"] = t_stats["frac_fast_trades"].fillna(0.0)

    # top-2 assets per user (behavior similarity proxy for multi-acc)
    top_assets = mt_long.groupby(["user_id", "asset_base"], as_index=False)["volume_token"].sum()
    top_assets = top_assets.sort_values(["user_id", "volume_token"], ascending=[True, False])
    top2 = top_assets.groupby("user_id").head(2).copy()
    top2["rank"] = top2.groupby("user_id").cumcount() + 1
    top2_wide = top2.pivot_table(index="user_id", columns="rank", values="asset_base", aggfunc="first").reset_index()
    top2_wide.columns = ["user_id", "top1_asset", "top2_asset"]
    top2_wide = top2_wide.rename(columns={"user_id": uid})
    t_stats = t_stats.merge(top2_wide, on=uid, how="left")
    t_stats["top1_asset"] = t_stats["top1_asset"].fillna("NA")
    t_stats["top2_asset"] = t_stats["top2_asset"].fillna("NA")

    # --- ledger ---
    tl_uid = ensure_col(tl, ["user_id"], True, "token_ledger")
    tl_ts = ensure_col(tl, ["ts"], True, "token_ledger")
    tl_type = ensure_col(tl, ["tx_type", "type"], True, "token_ledger")
    tl_amt = ensure_col(tl, ["amount"], True, "token_ledger")

    tl[tl_ts] = dt(tl[tl_ts])
    tl["day"] = tl[tl_ts].dt.floor("D")
    tl["amount"] = safe_num(tl[tl_amt]).fillna(0.0)

    # Define sinks/sources on TOKEN ledger:
    # - reward is emission (+)
    # - tax / fee / market_fee are sinks (usually negative for user in many ledgers)
    # Here, we robustly treat:
    #   reward_sum = sum(amount where tx_type=='reward' and amount>0)
    #   tax_sum = sum(abs(amount)) where tx_type=='tax'
    #   fee_sum = sum(abs(amount)) where tx_type in ('fee','market_fee')
    tl["tx_type_norm"] = tl[tl_type].astype(str).str.lower()

    def sum_abs(mask):
        return tl.loc[mask, "amount"].abs().sum()

    l_stats = tl.groupby(tl_uid).agg(
        tx_cnt=("amount", "size"),
        net_amount=("amount", "sum"),
    ).reset_index().rename(columns={tl_uid: uid})

    reward_sum = tl[tl["tx_type_norm"] == "reward"].groupby(tl_uid)["amount"].sum().reset_index().rename(columns={tl_uid: uid, "amount": "reward_sum"})
    claim_cnt = tl[tl["tx_type_norm"] == "reward"].groupby(tl_uid).size().reset_index(name="claim_cnt").rename(columns={tl_uid: uid})

    tax_sum = tl[tl["tx_type_norm"] == "tax"].groupby(tl_uid)["amount"].apply(lambda x: float(np.abs(x).sum())).reset_index().rename(columns={tl_uid: uid, "amount": "tax_sum"})
    fee_sum = tl[tl["tx_type_norm"].isin(["fee", "market_fee"])].groupby(tl_uid)["amount"].apply(lambda x: float(np.abs(x).sum())).reset_index().rename(columns={tl_uid: uid, "amount": "fee_sum"})

    l_stats = l_stats.merge(reward_sum, on=uid, how="left") \
                     .merge(claim_cnt, on=uid, how="left") \
                     .merge(tax_sum, on=uid, how="left") \
                     .merge(fee_sum, on=uid, how="left")

    for c in ["reward_sum", "claim_cnt", "tax_sum", "fee_sum"]:
        if c not in l_stats.columns:
            l_stats[c] = 0.0
        l_stats[c] = l_stats[c].fillna(0.0)

    # tx per day mean
    tx_ud = tl.groupby([tl_uid, "day"]).size().reset_index(name="tx_per_day")
    tx_rate = tx_ud.groupby(tl_uid)["tx_per_day"].mean().reset_index().rename(columns={tl_uid: uid, "tx_per_day": "tx_per_day_mean"})
    l_stats = l_stats.merge(tx_rate, on=uid, how="left")
    l_stats["tx_per_day_mean"] = l_stats["tx_per_day_mean"].fillna(0.0)

    # --- merge all features ---
    out = u[[uid, created, "created_day", "fingerprint", "fingerprint_size", "referrer_group_size", "signup_span_days_in_fp", "is_kyc"]].copy()
    out = out.merge(sess_stats, on=uid, how="left") \
             .merge(sess_len_stats, on=uid, how="left") \
             .merge(inter[[uid, "cv_inter_session", "med_inter_session_sec"]], on=uid, how="left") \
             .merge(hour_cov[[uid, "hour_coverage", "hours_active"]], on=uid, how="left") \
             .merge(night[[uid, "night_share"]], on=uid, how="left") \
             .merge(crash_stats, on=uid, how="left") \
             .merge(lat_stats, on=uid, how="left") \
             .merge(t_stats, on=uid, how="left") \
             .merge(l_stats, on=uid, how="left")

    # fill NAs with safe defaults
    fill0 = [
        "active_days", "sessions_cnt", "mean_sessions_per_day", "p95_sessions_per_day", "max_sessions_per_day",
        "avg_session_len", "med_session_len", "p90_session_len", "std_session_len",
        "cv_inter_session", "med_inter_session_sec", "hour_coverage", "hours_active", "night_share",
        "crash_rate", "latency_mean", "latency_std",
        "trades_cnt", "volume_token", "unique_assets", "unique_cps", "self_trade_cnt", "pingpong_cnt",
        "self_trade_share", "pingpong_share", "avg_slippage_bps", "avg_spread_bps", "tax_total", "fee_total",
        "tax_rate_eff", "fee_rate_eff", "cp_hhi", "med_inter_trade_sec", "p10_inter_trade_sec", "frac_fast_trades",
        "tx_cnt", "net_amount", "reward_sum", "tax_sum", "fee_sum", "claim_cnt", "tx_per_day_mean"
    ]
    for c in fill0:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # Top assets default
    for c in ["top1_asset", "top2_asset"]:
        if c in out.columns:
            out[c] = out[c].fillna("NA")

    return out


# =========================
# Scoring
# =========================
def score_users(
    feats: pd.DataFrame,
    w_wash: float,
    w_bot: float,
    w_multi: float,
    wash_weights: Dict[str, float],
    bot_weights: Dict[str, float],
    multi_weights: Dict[str, float],
) -> pd.DataFrame:
    """
    Convert features into three risk scores (0..100) + combined score.
    """
    df = feats.copy()

    # ----------------
    # Wash trading score components (all in [0..1])
    # ----------------
    c_self = rank_pct(df["self_trade_share"])
    c_ping = rank_pct(df["pingpong_share"])
    c_hhi = rank_pct(df["cp_hhi"])
    c_cps_low = inv_rank_pct(df["unique_cps"])  # fewer counterparties => suspicious
    c_assets_low = inv_rank_pct(df["unique_assets"])  # fewer assets => suspicious
    c_trades_hi = rank_pct(df["trades_cnt"])
    c_tax_eff_low = inv_rank_pct(df["tax_rate_eff"])  # if taxes are abnormally low (could indicate exploit routes)

    wash_raw = (
        wash_weights["self"] * c_self +
        wash_weights["pingpong"] * c_ping +
        wash_weights["hhi"] * c_hhi +
        wash_weights["cps_low"] * c_cps_low +
        wash_weights["assets_low"] * c_assets_low +
        wash_weights["trades_hi"] * c_trades_hi +
        wash_weights["tax_eff_low"] * c_tax_eff_low
    )
    wash_raw = wash_raw / max(1e-9, sum(wash_weights.values()))
    df["wash_score"] = 100.0 * clip01(wash_raw)

    # ----------------
    # Bot score components
    # ----------------
    c_sessions_hi = rank_pct(df["mean_sessions_per_day"])
    c_short_sessions = inv_rank_pct(df["avg_session_len"])
    c_regular = inv_rank_pct(df["cv_inter_session"].replace(0, np.nan).fillna(df["cv_inter_session"].median() if len(df) else 0.0))
    c_always_on = rank_pct(df["hour_coverage"])
    c_night = rank_pct(df["night_share"])
    c_fast_trades = rank_pct(df["frac_fast_trades"])
    c_fast_it = inv_rank_pct(df["med_inter_trade_sec"].replace(0, np.nan).fillna(df["med_inter_trade_sec"].median() if len(df) else 0.0))
    c_claim_hi = rank_pct(df["claim_cnt"])
    c_tx_hi = rank_pct(df["tx_per_day_mean"])

    bot_raw = (
        bot_weights["sessions_hi"] * c_sessions_hi +
        bot_weights["short_sessions"] * c_short_sessions +
        bot_weights["regularity"] * c_regular +
        bot_weights["always_on"] * c_always_on +
        bot_weights["night"] * c_night +
        bot_weights["fast_trades"] * c_fast_trades +
        bot_weights["fast_intertrade"] * c_fast_it +
        bot_weights["claims_hi"] * c_claim_hi +
        bot_weights["tx_hi"] * c_tx_hi
    )
    bot_raw = bot_raw / max(1e-9, sum(bot_weights.values()))
    df["bot_score"] = 100.0 * clip01(bot_raw)

    # ----------------
    # Multi-account score components
    # ----------------
    # fingerprint size big => suspicious
    c_fp_size = rank_pct(np.log1p(df["fingerprint_size"]))
    # signup span small => suspicious (invert)
    c_span_small = inv_rank_pct(df["signup_span_days_in_fp"])
    # referrer group big => suspicious
    c_ref_big = rank_pct(np.log1p(df["referrer_group_size"]))
    # non-KYC => more suspicious
    c_no_kyc = 1.0 - clip01(df["is_kyc"])
    # same behavior proxy: top assets are "too common" inside the population
    # (we measure how frequent user's (top1,top2) combo is)
    combo = df["top1_asset"].astype(str) + "|" + df["top2_asset"].astype(str)
    combo_freq = combo.map(combo.value_counts(dropna=False)).fillna(1).astype(float)
    c_combo = rank_pct(np.log1p(combo_freq))

    multi_raw = (
        multi_weights["fp_size"] * c_fp_size +
        multi_weights["span_small"] * c_span_small +
        multi_weights["ref_big"] * c_ref_big +
        multi_weights["no_kyc"] * c_no_kyc +
        multi_weights["behavior_combo"] * c_combo
    )
    multi_raw = multi_raw / max(1e-9, sum(multi_weights.values()))
    df["multiacc_score"] = 100.0 * clip01(multi_raw)

    # Combined score
    comb = (w_wash * df["wash_score"] + w_bot * df["bot_score"] + w_multi * df["multiacc_score"]) / max(1e-9, (w_wash + w_bot + w_multi))
    df["risk_score"] = comb

    return df


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Data")
    main_dir = st.text_input("Main data folder", value=DEFAULT_MAIN_DIR)

    st.divider()
    st.header("Heuristic thresholds")
    thr_wash = st.slider("Wash score threshold", 0, 100, 75, 1)
    thr_bot = st.slider("Bot score threshold", 0, 100, 75, 1)
    thr_multi = st.slider("Multi-account score threshold", 0, 100, 75, 1)
    thr_risk = st.slider("Combined risk threshold", 0, 100, 75, 1)

    st.divider()
    st.header("Scoring weights (global)")
    w_wash = st.slider("Weight: Wash", 0.0, 3.0, 1.0, 0.1)
    w_bot = st.slider("Weight: Bot", 0.0, 3.0, 1.0, 0.1)
    w_multi = st.slider("Weight: Multi-account", 0.0, 3.0, 1.0, 0.1)

    st.divider()
    st.header("Wash weights")
    wash_weights = {
        "self": st.slider("self-trades", 0.0, 3.0, 1.2, 0.1),
        "pingpong": st.slider("ping-pong pairs", 0.0, 3.0, 1.2, 0.1),
        "hhi": st.slider("counterparty concentration (HHI)", 0.0, 3.0, 1.0, 0.1),
        "cps_low": st.slider("few counterparties", 0.0, 3.0, 0.8, 0.1),
        "assets_low": st.slider("few assets", 0.0, 3.0, 0.3, 0.1),
        "trades_hi": st.slider("high trade count", 0.0, 3.0, 0.8, 0.1),
        "tax_eff_low": st.slider("low effective tax rate", 0.0, 3.0, 0.2, 0.1),
    }

    st.divider()
    st.header("Bot weights")
    bot_weights = {
        "sessions_hi": st.slider("high sessions/day", 0.0, 3.0, 1.0, 0.1),
        "short_sessions": st.slider("short sessions", 0.0, 3.0, 0.8, 0.1),
        "regularity": st.slider("high regularity (low CV)", 0.0, 3.0, 1.0, 0.1),
        "always_on": st.slider("24/7 activity (hour coverage)", 0.0, 3.0, 0.8, 0.1),
        "night": st.slider("night share", 0.0, 3.0, 0.4, 0.1),
        "fast_trades": st.slider("high fraction of fast trades", 0.0, 3.0, 0.8, 0.1),
        "fast_intertrade": st.slider("low median inter-trade time", 0.0, 3.0, 0.8, 0.1),
        "claims_hi": st.slider("many reward claims", 0.0, 3.0, 0.6, 0.1),
        "tx_hi": st.slider("high ledger tx/day", 0.0, 3.0, 0.4, 0.1),
    }

    st.divider()
    st.header("Multi-account weights")
    multi_weights = {
        "fp_size": st.slider("shared fingerprint size", 0.0, 3.0, 1.2, 0.1),
        "span_small": st.slider("signup span small", 0.0, 3.0, 1.0, 0.1),
        "ref_big": st.slider("shared referrer group size", 0.0, 3.0, 0.6, 0.1),
        "no_kyc": st.slider("no KYC", 0.0, 3.0, 0.6, 0.1),
        "behavior_combo": st.slider("behavior similarity (top assets)", 0.0, 3.0, 0.6, 0.1),
    }

    st.divider()
    st.header("Computation")
    pingpong_seconds = st.slider("Ping-pong window (sec)", 30, 3600, 300, 30)
    fast_trade_seconds = st.slider("Fast trade threshold (sec)", 10, 600, 60, 10)


# =========================
# Load and compute
# =========================
tables = load_core_tables(main_dir)

need = ["users", "sessions", "market_trades", "token_ledger"]
missing = [k for k in need if k not in tables]
if missing:
    st.error(f"Missing required table(s): {missing}. Check {main_dir}")
    st.stop()

users = tables["users"]
sessions = tables["sessions"]
trades = tables["market_trades"]
ledger = tables["token_ledger"]

with st.status("Building user features (may take a bit)...", expanded=False) as status:
    feats = build_user_features(
        users=users,
        sessions=sessions,
        trades=trades,
        ledger=ledger,
        pingpong_seconds=int(pingpong_seconds),
        fast_trade_seconds=int(fast_trade_seconds),
    )
    status.update(label="Features ready", state="complete")

with st.status("Scoring users...", expanded=False) as status:
    scored = score_users(
        feats=feats,
        w_wash=float(w_wash),
        w_bot=float(w_bot),
        w_multi=float(w_multi),
        wash_weights=wash_weights,
        bot_weights=bot_weights,
        multi_weights=multi_weights,
    )
    status.update(label="Scoring ready", state="complete")


# =========================
# Top-level KPIs
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Users", human_int(len(scored)))
col2.metric("High wash", human_int((scored["wash_score"] >= thr_wash).sum()))
col3.metric("High bot", human_int((scored["bot_score"] >= thr_bot).sum()))
col4.metric("High multi-acc", human_int((scored["multiacc_score"] >= thr_multi).sum()))

st.divider()

# =========================
# Tabs
# =========================
tab_overview, tab_wash, tab_bots, tab_multi, tab_drill = st.tabs([
    "Overview",
    "Wash Trading",
    "Bots",
    "Multi-account",
    "User Drilldown",
])


# =========================
# Overview
# =========================
with tab_overview:
    st.subheader("Risk score distribution")
    cA, cB = st.columns(2)
    with cA:
        plot_hist(scored, "risk_score", "Combined risk_score distribution")
    with cB:
        plot_hist(scored, "wash_score", "wash_score distribution")

    cC, cD = st.columns(2)
    with cC:
        plot_hist(scored, "bot_score", "bot_score distribution")
    with cD:
        plot_hist(scored, "multiacc_score", "multiacc_score distribution")

    st.subheader("Top suspects (combined)")
    topn = st.slider("Top N", 10, 500, 50, 10)
    view_cols = [
        "user_id", "risk_score", "wash_score", "bot_score", "multiacc_score",
        "fingerprint_size", "signup_span_days_in_fp", "referrer_group_size", "is_kyc",
        "trades_cnt", "volume_token", "self_trade_share", "pingpong_share", "cp_hhi",
        "mean_sessions_per_day", "avg_session_len", "cv_inter_session", "hour_coverage",
        "claim_cnt", "tx_per_day_mean",
        "top1_asset", "top2_asset"
    ]
    view_cols = [c for c in view_cols if c in scored.columns]
    top = scored.sort_values("risk_score", ascending=False).head(topn)[view_cols]
    st.dataframe(top, use_container_width=True)
    download_df(top, "fraudlab_top_suspects.csv", "Download top suspects CSV")

    st.markdown(
        """
**How to read these scores (quick):**
- **wash_score** spikes with self-trades, ping-pong between a pair, and concentrated counterparties.
- **bot_score** spikes with regular activity, 24/7 presence, short sessions, fast trading cadence.
- **multiacc_score** spikes with shared device fingerprint groups, tight signup clusters, shared referrers, no KYC.
"""
    )


# =========================
# Wash Trading tab
# =========================
with tab_wash:
    st.subheader("Wash trading signals (heuristic)")
    flagged = scored[scored["wash_score"] >= thr_wash].sort_values("wash_score", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        plot_hist(scored, "self_trade_share", "Self-trade share distribution")
    with c2:
        plot_hist(scored, "pingpong_share", "Ping-pong share distribution")

    c3, c4 = st.columns(2)
    with c3:
        plot_hist(scored, "cp_hhi", "Counterparty concentration (HHI) distribution")
    with c4:
        plot_scatter(
            scored.sample(min(3000, len(scored)), random_state=42) if len(scored) > 0 else scored,
            x="pingpong_share", y="cp_hhi", color=None,
            title="Ping-pong share vs counterparty concentration (sample)"
        )

    st.subheader("Flagged users")
    st.write(f"Users with wash_score ≥ {thr_wash}: **{human_int(len(flagged))}**")
    cols = [
        "user_id", "wash_score", "risk_score",
        "trades_cnt", "volume_token",
        "self_trade_share", "pingpong_share", "cp_hhi",
        "unique_cps", "unique_assets",
        "tax_rate_eff", "fee_rate_eff",
        "avg_slippage_bps", "avg_spread_bps",
        "fingerprint_size", "is_kyc",
        "top1_asset", "top2_asset"
    ]
    cols = [c for c in cols if c in flagged.columns]
    st.dataframe(flagged[cols].head(300), use_container_width=True)
    download_df(flagged[cols], "fraudlab_wash_flagged.csv", "Download wash-flagged CSV")


# =========================
# Bots tab
# =========================
with tab_bots:
    st.subheader("Bot-like behavior signals (heuristic)")
    flagged = scored[scored["bot_score"] >= thr_bot].sort_values("bot_score", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        plot_hist(scored, "mean_sessions_per_day", "Mean sessions/day distribution")
    with c2:
        plot_hist(scored, "cv_inter_session", "Inter-session regularity (CV) distribution")

    c3, c4 = st.columns(2)
    with c3:
        plot_hist(scored, "hour_coverage", "Hour coverage (0..1) distribution")
    with c4:
        plot_scatter(
            scored.sample(min(3000, len(scored)), random_state=7) if len(scored) > 0 else scored,
            x="mean_sessions_per_day", y="avg_session_len", color=None,
            title="Sessions/day vs avg session length (sample)"
        )

    c5, c6 = st.columns(2)
    with c5:
        plot_hist(scored, "frac_fast_trades", "Fraction of fast trades distribution")
    with c6:
        plot_hist(scored, "med_inter_trade_sec", "Median inter-trade time (sec) distribution")

    st.subheader("Flagged users")
    st.write(f"Users with bot_score ≥ {thr_bot}: **{human_int(len(flagged))}**")
    cols = [
        "user_id", "bot_score", "risk_score",
        "mean_sessions_per_day", "avg_session_len", "cv_inter_session", "hour_coverage", "night_share",
        "trades_cnt", "frac_fast_trades", "med_inter_trade_sec",
        "claim_cnt", "tx_per_day_mean",
        "crash_rate", "latency_mean",
        "fingerprint_size", "is_kyc",
    ]
    cols = [c for c in cols if c in flagged.columns]
    st.dataframe(flagged[cols].head(300), use_container_width=True)
    download_df(flagged[cols], "fraudlab_bot_flagged.csv", "Download bot-flagged CSV")


# =========================
# Multi-account tab
# =========================
with tab_multi:
    st.subheader("Multi-account / Sybil clusters (heuristic)")
    flagged = scored[scored["multiacc_score"] >= thr_multi].sort_values("multiacc_score", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        plot_hist(scored, "fingerprint_size", "Fingerprint group size distribution")
    with c2:
        plot_hist(scored, "signup_span_days_in_fp", "Signup span (days) within fingerprint distribution")

    st.subheader("Top fingerprint clusters")
    # Cluster summary
    cluster = scored.groupby("fingerprint", as_index=False).agg(
        users=("user_id", "count"),
        avg_multi=("multiacc_score", "mean"),
        avg_risk=("risk_score", "mean"),
        kyc_rate=("is_kyc", "mean"),
        span_days=("signup_span_days_in_fp", "max"),
    )
    cluster = cluster.sort_values(["users", "avg_multi"], ascending=[False, False])

    st.dataframe(cluster.head(100), use_container_width=True)
    download_df(cluster, "fraudlab_fingerprint_clusters.csv", "Download fingerprint clusters CSV")

    st.subheader("Flagged users")
    st.write(f"Users with multiacc_score ≥ {thr_multi}: **{human_int(len(flagged))}**")
    cols = [
        "user_id", "multiacc_score", "risk_score",
        "fingerprint_size", "signup_span_days_in_fp", "referrer_group_size", "is_kyc",
        "top1_asset", "top2_asset",
        "mean_sessions_per_day", "trades_cnt", "volume_token"
    ]
    cols = [c for c in cols if c in flagged.columns]
    st.dataframe(flagged[cols].head(300), use_container_width=True)
    download_df(flagged[cols], "fraudlab_multiacc_flagged.csv", "Download multiacc-flagged CSV")

    st.subheader("Cluster drill (fingerprint)")
    fp_pick = st.selectbox("Pick a fingerprint cluster", options=cluster["fingerprint"].head(200).tolist())
    members = scored[scored["fingerprint"] == fp_pick].sort_values("risk_score", ascending=False)
    st.write(f"Cluster size: **{human_int(len(members))}** | avg risk: **{human_float(members['risk_score'].mean(),2)}**")
    st.dataframe(members.head(300), use_container_width=True)


# =========================
# User Drilldown
# =========================
with tab_drill:
    st.subheader("User drilldown (timeline + features)")
    # pick user from top risk by default
    default_user = scored.sort_values("risk_score", ascending=False)["user_id"].iloc[0]
    user_pick = st.selectbox("Pick a user_id", options=scored.sort_values("risk_score", ascending=False)["user_id"].head(2000).tolist(), index=0)

    row = scored[scored["user_id"] == user_pick].iloc[0].to_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("risk_score", human_float(row.get("risk_score", 0), 2))
    c2.metric("wash_score", human_float(row.get("wash_score", 0), 2))
    c3.metric("bot_score", human_float(row.get("bot_score", 0), 2))
    c4.metric("multiacc_score", human_float(row.get("multiacc_score", 0), 2))

    with st.expander("User features (raw)"):
        st.json({k: row[k] for k in row.keys() if k not in ["fingerprint"]})
        st.write("fingerprint:", row.get("fingerprint"))

    # Slice raw tables for timelines
    # sessions
    s = sessions.copy()
    s["start_ts"] = dt(s["start_ts"])
    us = s[s["user_id"] == user_pick].sort_values("start_ts")

    # trades: user as maker or taker
    mt = trades.copy()
    mt["ts"] = dt(mt["ts"])
    umt = mt[(mt["maker_user_id"] == user_pick) | (mt["taker_user_id"] == user_pick)].sort_values("ts").copy()
    if not umt.empty:
        umt["volume_token"] = safe_num(umt["amount_base"]).fillna(0.0) * safe_num(umt["price"]).fillna(0.0)
        umt["role"] = np.where(umt["maker_user_id"] == user_pick, "maker", "taker")
        umt["counterparty"] = np.where(umt["role"] == "maker", umt["taker_user_id"], umt["maker_user_id"])
        umt["is_self_trade"] = (umt["maker_user_id"] == umt["taker_user_id"]).astype(int)

    # ledger
    tl = ledger.copy()
    tl["ts"] = dt(tl["ts"])
    utl = tl[tl["user_id"] == user_pick].sort_values("ts")

    st.markdown("### Activity timelines")

    colA, colB = st.columns(2)
    with colA:
        st.write("Sessions (latest 300)")
        st.dataframe(us.tail(300), use_container_width=True)
    with colB:
        st.write("Token ledger (latest 300)")
        st.dataframe(utl.tail(300), use_container_width=True)

    st.write("Market trades (latest 300)")
    st.dataframe(umt.tail(300), use_container_width=True)

    if not us.empty:
        us2 = us.copy()
        us2["day"] = us2["start_ts"].dt.floor("D")
        daily_sess = us2.groupby("day").size().reset_index(name="sessions")
        if PLOTLY_OK:
            fig = px.line(daily_sess, x="day", y="sessions", title="Sessions per day")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(daily_sess.set_index("day")["sessions"])

    if not umt.empty:
        umt2 = umt.copy()
        umt2["day"] = umt2["ts"].dt.floor("D")
        daily_vol = umt2.groupby("day")["volume_token"].sum().reset_index(name="volume_token")
        daily_self = umt2.groupby("day")["is_self_trade"].mean().reset_index(name="self_trade_share")

        c1, c2 = st.columns(2)
        with c1:
            if PLOTLY_OK:
                fig = px.line(daily_vol, x="day", y="volume_token", title="Trade volume per day")
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(daily_vol.set_index("day")["volume_token"])
        with c2:
            if PLOTLY_OK:
                fig = px.line(daily_self, x="day", y="self_trade_share", title="Self-trade share per day")
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(daily_self.set_index("day")["self_trade_share"])

    # Cluster context
    st.markdown("### Cluster context (fingerprint group)")
    fp = row.get("fingerprint", None)
    if fp is not None:
        members = scored[scored["fingerprint"] == fp].sort_values("risk_score", ascending=False)
        st.write(f"Fingerprint group size: **{human_int(len(members))}**")
        st.dataframe(members.head(200), use_container_width=True)


# =========================
# Combined export
# =========================
st.divider()
st.subheader("Export")
export_cols = [c for c in scored.columns]
download_df(scored[export_cols], "fraudlab_scored_users.csv", "Download full scored users CSV")
