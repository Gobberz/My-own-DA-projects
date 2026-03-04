# pages/07_Alliances.py
# Alliances
# Alliance analytics: ASI, Shapley payouts, retention, and deeper behavioral/network diagnostics.
#
# Includes:
# - ASI (Alliance Strength Index) over time
# - Retention in alliances (member activity retention)
# - Shapley payouts (approx; Monte Carlo) for a selected alliance and window
# + 5 extra analyses:
#   A1) Cohesion: internal vs external trading share
#   A2) Contribution inequality: Gini/top shares per alliance
#   A3) Member migration flows (guild switching)
#   A4) Event impact (guild events -> activity lift)
#   A5) Churn risk model (predict next-7d drop in active members)
#
# All comments/labels in English.

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

try:
    import networkx as nx
    NX_OK = True
except Exception:
    NX_OK = False

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    SK_OK = True
except Exception:
    SK_OK = False


# =========================
# Page config
# =========================
st.set_page_config(page_title="Alliances", layout="wide")
st.title("Alliances")
st.caption("Alliance analytics: ASI, Shapley payouts, retention, and deeper behavioral/network diagnostics.")


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
    """Always returns a pandas Series if input is Series/arraylike; for scalar returns scalar."""
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    if isinstance(x, (np.ndarray, list, tuple, pd.Index)):
        return pd.to_numeric(pd.Series(x), errors="coerce")
    # scalar
    return pd.to_numeric(x, errors="coerce")

def to_series(x, index=None, name: str = None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (np.ndarray, list, tuple, pd.Index)):
        return pd.Series(x, index=index, name=name)
    if index is None:
        return pd.Series([x], name=name)
    return pd.Series([x] * len(index), index=index, name=name)

def safe_div(num, den) -> pd.Series:
    num_s = to_series(num)
    den_s = to_series(den, index=num_s.index)
    num_s = pd.to_numeric(num_s, errors="coerce").astype(float)
    den_s = pd.to_numeric(den_s, errors="coerce").astype(float)
    return num_s / den_s.replace(0.0, np.nan)

def gini(x: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy()
    a = a[a >= 0]
    if len(a) == 0:
        return np.nan
    s = a.sum()
    if s == 0:
        return 0.0
    a = np.sort(a)
    n = len(a)
    cum = np.cumsum(a)
    return float((n + 1 - 2 * np.sum(cum) / s) / n)

def top_share(x: pd.Series, p: float = 0.01) -> float:
    s = pd.to_numeric(x, errors="coerce").dropna()
    s = s[s >= 0].sort_values(ascending=False)
    if len(s) == 0:
        return np.nan
    k = max(1, int(len(s) * p))
    return float(s.head(k).sum() / s.sum()) if s.sum() > 0 else 0.0

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def plot_lines(df: pd.DataFrame, x: str, ys: List[str], title: str, height: int = 380):
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
        fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50, b=10))
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

    # main required
    for key, fname in [
        ("users", "users.csv"),
        ("sessions", "sessions.csv"),
        ("market_trades", "market_trades.csv"),
    ]:
        p = find_existing([os.path.join(main_dir, fname)])
        if p:
            tables[key] = load_csv(p)

    # extra alliance-related
    opt = {
        "guild_membership": ["guild_membership.csv", "alliance_membership.csv", "aliance.csv", "alliance.csv"],
        "guild_events": ["guild_events.csv", "alliance_events.csv"],
        "funnel_events": ["funnel_events.csv"],
        "marketing_spend": ["marketing_spend.csv"],
    }
    for k, candidates in opt.items():
        p = find_existing([os.path.join(extra_dir, c) for c in candidates])
        if p:
            tables[k] = load_csv(p)

    return tables


# =========================
# Build alliance panels
# =========================
@dataclass
class AlliancePanels:
    user_day: pd.DataFrame
    user_day_guild: pd.DataFrame
    guild_day: pd.DataFrame
    guild_summary: pd.DataFrame
    migration: pd.DataFrame
    event_impact: pd.DataFrame

@st.cache_data(show_spinner=True)
def build_alliance_panels(tables) -> dict:
    """
    Builds:
    - user_day: activity and trading at user-day level
    - user_day_guild: user-day with guild_id attached (expanded membership)
    - guild_day: guild/day metrics + ASI components
    - guild_summary: overall guild stats
    - migration: user guild switches
    - event_impact: event-day lift proxy
    """
    users = tables["users"].copy()
    sessions = tables["sessions"].copy()
    mt = tables["market_trades"].copy()
    guild_day = pd.DataFrame()

    gm = tables.get("guild_membership")
    ge = tables.get("guild_events")

    if gm is not None and ge is not None and not gm.empty and not ge.empty:
        gm = gm.copy()
        ge = ge.copy()

        # columns
        gm_uid = "user_id" if "user_id" in gm.columns else None
        gm_gid = "guild_id" if "guild_id" in gm.columns else None
        if gm_uid is None or gm_gid is None:
            guild_day = pd.DataFrame()
        else:
            ge_ts = "ts" if "ts" in ge.columns else None
            ge_gid = "guild_id" if "guild_id" in ge.columns else None
            ge_ev = "event_type" if "event_type" in ge.columns else (
                "event_name" if "event_name" in ge.columns else None)

            if ge_ts is None or ge_gid is None:
                guild_day = pd.DataFrame()
            else:
                ge[ge_ts] = pd.to_datetime(ge[ge_ts], errors="coerce")
                ge["day"] = ge[ge_ts].dt.floor("D")

                # events per guild-day
                g_events = ge.groupby([ge_gid, "day"]).size().reset_index(name="guild_events_cnt")
                g_events = g_events.rename(columns={ge_gid: "guild_id"})

                # members per guild (static snapshot proxy)
                members = gm.groupby(gm_gid)[gm_uid].nunique().reset_index(name="members")
                members = members.rename(columns={gm_gid: "guild_id"})

                guild_day = g_events.merge(members, on="guild_id", how="left")
                guild_day["members"] = pd.to_numeric(guild_day["members"], errors="coerce").fillna(0).astype(int)

                # basic "activity per member" proxy
                guild_day["events_per_member"] = guild_day["guild_events_cnt"] / guild_day["members"].replace(0, np.nan)
                guild_day["events_per_member"] = guild_day["events_per_member"].fillna(0.0)
    else:
        guild_day = pd.DataFrame()

    # ---------- Normalize users ----------
    u_user = ensure_col(users, ["user_id", "id"], True, "users")
    u_created = ensure_col(users, ["created_at", "signup_ts", "reg_ts"], True, "users")
    users[u_created] = dt(users[u_created])

    # ---------- Normalize sessions ----------
    s_user = ensure_col(sessions, ["user_id"], True, "sessions")
    s_start = ensure_col(sessions, ["start_ts", "ts"], True, "sessions")
    sessions[s_start] = dt(sessions[s_start])
    sessions["day"] = sessions[s_start].dt.floor("D")
    s_len = ensure_col(sessions, ["session_len_sec", "duration_sec"], required=False, name="sessions")
    if s_len:
        sessions[s_len] = pd.to_numeric(sessions[s_len], errors="coerce").fillna(0.0)
    else:
        sessions["session_len_proxy"] = 1.0
        s_len = "session_len_proxy"

    # ---------- Normalize trades ----------
    mt_ts = ensure_col(mt, ["ts"], True, "market_trades")
    mt[mt_ts] = dt(mt[mt_ts])
    mt["day"] = mt[mt_ts].dt.floor("D")

    maker = ensure_col(mt, ["maker_user_id", "user_id"], True, "market_trades")
    taker = ensure_col(mt, ["taker_user_id"], True, "market_trades")
    base_col = ensure_col(mt, ["asset_base", "resource", "resource_key"], True, "market_trades")
    mt["resource"] = mt[base_col].astype(str).str.upper()

    # >>> CRITICAL FIX: safe_num must receive a Series, not a column name string <<<
    amt_col = ensure_col(mt, ["amount_base"], True, "market_trades")
    px_col = ensure_col(mt, ["price"], True, "market_trades")

    mt["amount_base"] = safe_num(mt[amt_col]).fillna(0.0)
    mt["price"] = safe_num(mt[px_col]).fillna(0.0)

    for c in ["fee_amount", "tax_amount", "slippage_bps", "spread_bps"]:
        if c in mt.columns:
            mt[c] = safe_num(mt[c]).fillna(0.0)
        else:
            mt[c] = 0.0

    mt["volume_token"] = (mt["amount_base"] * mt["price"]).fillna(0.0)
    mt["revenue_token"] = (mt["fee_amount"] + mt["tax_amount"]).clip(lower=0.0)

    # ---------- Define global day range ----------
    day_min = min(sessions["day"].min(), mt["day"].min())
    day_max = max(sessions["day"].max(), mt["day"].max())
    all_days = pd.date_range(day_min, day_max, freq="D")

    # ---------- user_day base ----------
    sess_ud = sessions.groupby([s_user, "day"]).agg(
        sessions_cnt=("day", "size"),
        session_time=(s_len, "sum")
    ).reset_index().rename(columns={s_user: "user_id"})

    trade_ud = mt.groupby([maker, "day"]).agg(
        trades_cnt=("day", "size"),
        volume_token=("volume_token", "sum"),
        revenue_token=("revenue_token", "sum")
    ).reset_index().rename(columns={maker: "user_id"})

    user_day = sess_ud.merge(trade_ud, on=["user_id", "day"], how="outer").fillna(0.0)
    user_day["active"] = (user_day["sessions_cnt"] > 0).astype(int)
    user_day["trader"] = (user_day["trades_cnt"] > 0).astype(int)

    # ---------- Expand membership to user-day ----------
    # Check if guild_membership exists
    if gm is None or gm.empty:
        # No guild data - assign all users to NO_GUILD
        user_day_guild = user_day.copy()
        user_day_guild["guild_id"] = "NO_GUILD"
        
        # Create empty aggregations
        gd = pd.DataFrame(columns=["day", "guild_id", "members_total", "active_members", "traders",
                                    "sessions_total", "session_time_total", "trades_total",
                                    "volume_token", "revenue_token", "active_ratio", "volume_per_member",
                                    "revenue_per_member", "internal_share", "gini_volume", 
                                    "top1_share_volume", "top5_share_volume", "ret7", "asi_raw", "ASI"])
        gs = pd.DataFrame(columns=["guild_id", "days", "members_avg", "active_ratio_avg",
                                    "volume_total", "revenue_total", "internal_share_avg",
                                    "gini_avg", "ret7_avg", "ASI_avg", "ASI_p90"])
        migration = pd.DataFrame(columns=["day", "from_guild", "to_guild", "switch_users"])
    else:
        gm = gm.copy()
        gm_user = ensure_col(gm, ["user_id"], True, "guild_membership")
        gm_gid = ensure_col(gm, ["guild_id", "alliance_id"], True, "guild_membership")

        # membership dates can vary; handle robustly
        gm_join = ensure_col(gm, ["joined_at", "join_ts", "start_ts", "joined_day"], required=False, name="guild_membership")
        gm_left = ensure_col(gm, ["left_at", "leave_ts", "end_ts", "left_day"], required=False, name="guild_membership")

        if gm_join:
            gm[gm_join] = dt(gm[gm_join]).dt.floor("D")
        else:
            gm[gm_join or "joined_at_proxy"] = day_min
            gm_join = gm_join or "joined_at_proxy"

        if gm_left:
            gm[gm_left] = dt(gm[gm_left]).dt.floor("D")
        else:
            gm[gm_left or "left_at_proxy"] = day_max
            gm_left = gm_left or "left_at_proxy"

        gm[gm_join] = gm[gm_join].fillna(day_min)
        gm[gm_left] = gm[gm_left].fillna(day_max)
        gm[gm_left] = gm[[gm_left]].min(axis=1)  # defensive
        gm[gm_join] = gm[[gm_join]].max(axis=1)

        # Expand with explode (manageable for synthetic)
        # rows ~ membership_count * days_in_membership
        gm2 = gm[[gm_user, gm_gid, gm_join, gm_left]].copy()
        gm2["day"] = gm2.apply(lambda r: pd.date_range(r[gm_join], r[gm_left], freq="D"), axis=1)
        gm2 = gm2.explode("day").rename(columns={gm_user: "user_id", gm_gid: "guild_id"})
        gm2["guild_id"] = gm2["guild_id"].astype(str)

        # If a user is in multiple guilds same day, pick the first (rare; synthetic)
        gm2 = gm2.sort_values(["user_id", "day", "guild_id"]).drop_duplicates(["user_id", "day"], keep="first")

        user_day_guild = user_day.merge(gm2, on=["user_id", "day"], how="left")
        user_day_guild["guild_id"] = user_day_guild["guild_id"].fillna("NO_GUILD")

        # ---------- Guild-day aggregation ----------
        gd = user_day_guild.groupby(["day", "guild_id"]).agg(
            members_total=("user_id", "nunique"),
            active_members=("active", "sum"),
            traders=("trader", "sum"),
            sessions_total=("sessions_cnt", "sum"),
            session_time_total=("session_time", "sum"),
            trades_total=("trades_cnt", "sum"),
            volume_token=("volume_token", "sum"),
            revenue_token=("revenue_token", "sum"),
        ).reset_index()

        gd["active_ratio"] = safe_div(gd["active_members"], gd["members_total"]).fillna(0.0)
        gd["volume_per_member"] = safe_div(gd["volume_token"], gd["members_total"]).fillna(0.0)
        gd["revenue_per_member"] = safe_div(gd["revenue_token"], gd["members_total"]).fillna(0.0)

        # ---------- A1) Cohesion: internal vs external share ----------
        # Map trades to guilds on that day for maker and taker
        gm_map = gm2.rename(columns={"guild_id": "guild_id_map"})
        maker_map = gm_map.rename(columns={"user_id": maker, "guild_id_map": "guild_maker"})
        taker_map = gm_map.rename(columns={"user_id": taker, "guild_id_map": "guild_taker"})

        mt2 = mt[[ "day", maker, taker, "volume_token" ]].copy()
        mt2 = mt2.merge(maker_map[[maker, "day", "guild_maker"]], on=[maker, "day"], how="left")
        mt2 = mt2.merge(taker_map[[taker, "day", "guild_taker"]], on=[taker, "day"], how="left")
        mt2["guild_maker"] = mt2["guild_maker"].fillna("NO_GUILD")
        mt2["guild_taker"] = mt2["guild_taker"].fillna("NO_GUILD")

        mt2["is_internal"] = (mt2["guild_maker"] == mt2["guild_taker"]) & (mt2["guild_maker"] != "NO_GUILD")
        mt2["guild_id"] = mt2["guild_maker"]

        coh = mt2[mt2["guild_id"] != "NO_GUILD"].groupby(["day", "guild_id"]).agg(
            internal_volume=("volume_token", lambda s: float(s[mt2.loc[s.index, "is_internal"]].sum()) if len(s) else 0.0),
            total_volume=("volume_token", "sum")
        ).reset_index()
        coh["internal_share"] = safe_div(coh["internal_volume"], coh["total_volume"]).fillna(0.0)

        gd = gd.merge(coh[["day", "guild_id", "internal_share"]], on=["day", "guild_id"], how="left")
        gd["internal_share"] = gd["internal_share"].fillna(0.0)

        # ---------- A2) Contribution inequality (Gini/top1) ----------
        # Compute per guild-day user volumes
        u_gd = user_day_guild[user_day_guild["guild_id"] != "NO_GUILD"].groupby(["day", "guild_id", "user_id"]).agg(
            user_volume=("volume_token", "sum"),
            user_revenue=("revenue_token", "sum"),
            user_trades=("trades_cnt", "sum"),
            user_active=("active", "max"),
        ).reset_index()

        ineq = u_gd.groupby(["day", "guild_id"]).apply(
            lambda g: pd.Series({
                "gini_volume": gini(g["user_volume"]),
                "top1_share_volume": top_share(g["user_volume"], 0.01),
                "top5_share_volume": top_share(g["user_volume"], 0.05),
            })
        ).reset_index()

        gd = gd.merge(ineq, on=["day", "guild_id"], how="left")
        for c in ["gini_volume", "top1_share_volume", "top5_share_volume"]:
            gd[c] = pd.to_numeric(gd[c], errors="coerce").fillna(0.0)

        # ---------- Retention in alliances (activity retention) ----------
        # For each guild-day: among active members today, share active again in +7 days (still in same guild-day mapping)
        ud = user_day_guild.copy()
        ud["day_plus7"] = ud["day"] + pd.Timedelta(days=7)
        active_today = ud[ud["active"] == 1][["user_id", "guild_id", "day", "day_plus7"]].copy()

        active_future = ud[ud["active"] == 1][["user_id", "guild_id", "day"]].copy()
        active_future = active_future.rename(columns={"day": "day_plus7"})

        ret7 = active_today.merge(active_future, on=["user_id", "guild_id", "day_plus7"], how="left", indicator=True)
        ret7["ret7_flag"] = (ret7["_merge"] == "both").astype(int)

        ret7_g = ret7.groupby(["day", "guild_id"]).agg(
            ret7=("ret7_flag", "mean"),
            active_members_today=("user_id", "nunique")
        ).reset_index()

        gd = gd.merge(ret7_g[["day", "guild_id", "ret7"]], on=["day", "guild_id"], how="left")
        gd["ret7"] = gd["ret7"].fillna(0.0)

        # ---------- ASI (Alliance Strength Index) ----------
        # ASI combines: active_ratio, volume_per_member, internal_share, ret7, and penalizes gini/top1.
        gd["asi_raw"] = (
            0.28 * zscore(gd["active_ratio"]) +
            0.25 * zscore(np.log1p(gd["volume_per_member"])) +
            0.18 * zscore(gd["internal_share"]) +
            0.19 * zscore(gd["ret7"]) -
            0.10 * zscore(gd["gini_volume"])
        )
        gd["ASI"] = (50 + 15 * gd["asi_raw"]).clip(0, 100)

        # ---------- Guild summary (overall) ----------
        gs = gd[gd["guild_id"] != "NO_GUILD"].groupby("guild_id").agg(
            days=("day", "nunique"),
            members_avg=("members_total", "mean"),
            active_ratio_avg=("active_ratio", "mean"),
            volume_total=("volume_token", "sum"),
            revenue_total=("revenue_token", "sum"),
            internal_share_avg=("internal_share", "mean"),
            gini_avg=("gini_volume", "mean"),
            ret7_avg=("ret7", "mean"),
            ASI_avg=("ASI", "mean"),
            ASI_p90=("ASI", lambda s: float(np.nanpercentile(s, 90)) if len(s) else np.nan),
        ).reset_index()

        # ---------- A3) Migration flows (guild switching) ----------
        # Identify switches: guild_id changes day-to-day for a user
        uday = user_day_guild.sort_values(["user_id", "day"]).copy()
        uday["guild_prev"] = uday.groupby("user_id")["guild_id"].shift(1)
        mig = uday[(uday["guild_id"] != uday["guild_prev"]) & uday["guild_prev"].notna()].copy()
        mig = mig[(mig["guild_prev"] != "NO_GUILD") | (mig["guild_id"] != "NO_GUILD")]
        migration = mig.groupby(["day", "guild_prev", "guild_id"]).size().reset_index(name="switch_users")
        migration = migration.rename(columns={"guild_prev": "from_guild", "guild_id": "to_guild"})

    # ---------- A4) Event impact ----------
    event_impact = pd.DataFrame()
    if ge is not None and not ge.empty:
        ge = ge.copy()
        ge_ts = ensure_col(ge, ["ts", "event_ts"], True, "guild_events")
        ge_gid = ensure_col(ge, ["guild_id", "alliance_id"], True, "guild_events")
        ge_type = ensure_col(ge, ["event_type", "event_name"], True, "guild_events")
        ge[ge_ts] = dt(ge[ge_ts])
        ge["day"] = ge[ge_ts].dt.floor("D")
        ge["guild_id"] = ge[ge_gid].astype(str)
        ge["event_type"] = ge[ge_type].astype(str)

        # join guild_day and measure lift vs prior 7d mean
        gd2 = gd[gd["guild_id"] != "NO_GUILD"].copy()
        gd2 = gd2.sort_values(["guild_id", "day"])
        gd2["vol_7d_mean"] = gd2.groupby("guild_id")["volume_token"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        gd2["act_7d_mean"] = gd2.groupby("guild_id")["active_members"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)

        ev = ge.groupby(["day", "guild_id", "event_type"]).size().reset_index(name="events_cnt")
        evm = ev.merge(gd2[["day","guild_id","volume_token","vol_7d_mean","active_members","act_7d_mean"]], on=["day","guild_id"], how="left")

        evm["volume_lift"] = safe_div(evm["volume_token"] - evm["vol_7d_mean"], evm["vol_7d_mean"]).fillna(0.0)
        evm["active_lift"] = safe_div(evm["active_members"] - evm["act_7d_mean"], evm["act_7d_mean"]).fillna(0.0)
        event_impact = evm.sort_values("volume_lift", ascending=False)

    return {
        "user_day": user_day,
        "user_day_guild": user_day_guild,
        "guild_day": gd,
        "guild_summary": gs,
        "migration": migration,
        "event_impact": event_impact,
    }


# =========================
# Shapley payouts (approx)
# =========================
def shapley_payouts_mc(
    trades: pd.DataFrame,
    user_day_guild: pd.DataFrame,
    guild_id: str,
    start_day: pd.Timestamp,
    end_day_excl: pd.Timestamp,
    n_perm: int = 250,
    alpha_synergy: float = 0.3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Approximate Shapley values for a guild over a time window.

    Value function v(S):
      v(S) = sum(volume by members in S) + alpha * internal_volume_within_S

    internal_volume_within_S computed from maker/taker both in S and both in guild.

    Returns payout weights normalized to 1.
    """
    rng = np.random.default_rng(random_seed)

    # Filter window & guild membership mapping for day-level
    udg = user_day_guild.copy()
    udg["day"] = dt(udg["day"]).dt.floor("D")

    mask_ud = (udg["guild_id"].astype(str) == str(guild_id)) & (udg["day"] >= start_day) & (udg["day"] < end_day_excl)
    members = sorted(udg.loc[mask_ud, "user_id"].unique().tolist())
    if len(members) == 0:
        return pd.DataFrame(columns=["user_id","shapley","payout_share"])

    # Build user set for fast checks
    mem_set = set(members)

    mt = trades.copy()
    mt["day"] = dt(mt["day"]).dt.floor("D")
    maker = ensure_col(mt, ["maker_user_id"], True, "market_trades")
    taker = ensure_col(mt, ["taker_user_id"], True, "market_trades")
    vol_col = "volume_token"
    if vol_col not in mt.columns:
        # try rebuild
        amt_col = ensure_col(mt, ["amount_base"], True, "market_trades")
        px_col = ensure_col(mt, ["price"], True, "market_trades")
        mt["amount_base"] = pd.to_numeric(mt[amt_col], errors="coerce").fillna(0.0)
        mt["price"] = pd.to_numeric(mt[px_col], errors="coerce").fillna(0.0)
        mt["volume_token"] = mt["amount_base"] * mt["price"]

    mt = mt[(mt["day"] >= start_day) & (mt["day"] < end_day_excl)].copy()

    # Keep only trades where maker is in guild members (simplify)
    mt = mt[mt[maker].isin(mem_set) | mt[taker].isin(mem_set)].copy()

    # Precompute per-user solo volume
    solo = mt.groupby(maker)[vol_col].sum()
    solo = solo.reindex(members).fillna(0.0)

    # Precompute pairwise internal volume between members (maker,taker both in members)
    pairs = mt[(mt[maker].isin(mem_set)) & (mt[taker].isin(mem_set))].copy()
    if pairs.empty:
        # no synergy
        alpha_synergy = 0.0

    # store interactions as list of tuples for incremental updates
    inter = []
    if alpha_synergy > 0 and not pairs.empty:
        for r in pairs[[maker, taker, vol_col]].itertuples(index=False):
            inter.append((r[0], r[1], float(r[2])))

    # Shapley MC
    shap = {u: 0.0 for u in members}

    # Helper v(S) incremental:
    # maintain sum_solo, and synergy within S: sum(volume of edges with both ends in S)
    for _ in range(n_perm):
        perm = members.copy()
        rng.shuffle(perm)

        S = set()
        sum_solo = 0.0
        synergy = 0.0

        # For faster synergy update, we recompute synergy naive per step for small n
        # (OK for synthetic; for huge you'd pre-index adjacency)
        for u in perm:
            v_before = sum_solo + alpha_synergy * synergy

            # add u
            S.add(u)
            sum_solo += float(solo.get(u, 0.0))

            if alpha_synergy > 0 and inter:
                # add edges newly internal because u joined
                add_syn = 0.0
                for a, b, w in inter:
                    if (a == u and b in S) or (b == u and a in S):
                        add_syn += w
                synergy += add_syn

            v_after = sum_solo + alpha_synergy * synergy
            shap[u] += (v_after - v_before)

    # average
    for u in shap:
        shap[u] /= float(n_perm)

    out = pd.DataFrame({"user_id": list(shap.keys()), "shapley": list(shap.values())})
    out["shapley"] = pd.to_numeric(out["shapley"], errors="coerce").fillna(0.0)
    total = out["shapley"].sum()
    out["payout_share"] = safe_div(out["shapley"], total).fillna(0.0) if total > 0 else 0.0
    out = out.sort_values("payout_share", ascending=False)
    return out


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Data folders")
    main_dir = st.text_input("Main data folder", value=DEFAULT_MAIN_DIR)
    extra_dir = st.text_input("Extra data folder", value=DEFAULT_EXTRA_DIR)

    st.divider()
    st.header("Options")
    show_debug = st.checkbox("Show debug tables", value=False)
    use_network = st.checkbox("Network diagnostics (requires networkx)", value=True)
    n_perm = st.slider("Shapley permutations (MC)", 50, 800, 250, 50)


# =========================
# Load & build
# =========================
tables = load_tables(main_dir, extra_dir)

need = ["users", "sessions", "market_trades", "guild_membership"]
missing = [k for k in need if k not in tables or tables[k] is None or tables[k].empty]
if missing:
    st.error(f"Missing required tables: {missing}. Check folders:\n- main: {main_dir}\n- extra: {extra_dir}")
    st.stop()

with st.status("Building alliance panels (user-day, guild-day, ASI)...", expanded=True) as status:
    prog = st.progress(0)
    prog.progress(10, text="Loading tables & normalizing columns...")
    pack = build_alliance_panels(tables)
    st.dataframe(pack["user_day"])
    prog.progress(100, text="Done")
    status.update(state="complete", label="Alliance panels ready")

ud = pack["user_day_guild"]
gd = pack["guild_day"]
gs = pack["guild_summary"]

# =========================
# Top overview
# =========================
st.subheader("Alliance overview")

c1, c2, c3, c4, c5 = st.columns(5)
active_guilds = int((gd["guild_id"] != "NO_GUILD").nunique())
c1.metric("Guilds (incl. NO_GUILD)", int(gd["guild_id"].nunique()))
c2.metric("Active guilds", active_guilds)
c3.metric("Avg ASI (top 10 guilds)", f"{gs.head(10)['ASI_avg'].mean():.1f}")
c4.metric("Max ASI (any day)", f"{gd['ASI'].max():.1f}")
c5.metric("Avg internal share (top 10)", f"{gs.head(10)['internal_share_avg'].mean():.2f}")

st.dataframe(gs.head(30), use_container_width=True)
download_df(gs, "alliances_guild_summary.csv", "Download guild summary CSV")

# =========================
# Controls: guild & date range
# =========================
st.subheader("Guild drill-down")

guilds = sorted([g for g in gd["guild_id"].astype(str).unique() if g != "NO_GUILD"])
if not guilds:
    st.warning("No guilds found (all users in NO_GUILD). Check guild_membership.csv.")
    st.stop()

colA, colB, colC = st.columns([2, 2, 3])
with colA:
    guild_id = st.selectbox("Guild", guilds, index=0)
with colB:
    dmin = dt(gd["day"]).min()
    dmax = dt(gd["day"]).max()
    date_from = st.date_input("From", value=dmin.date())
    date_to = st.date_input("To (inclusive)", value=dmax.date())
with colC:
    st.caption("Tip: Shapley is computed for selected window; reduce range for speed.")

start_day = pd.to_datetime(date_from).floor("D")
end_day_excl = (pd.to_datetime(date_to).floor("D") + pd.Timedelta(days=1))

gsel = gd[(gd["guild_id"].astype(str) == str(guild_id)) & (gd["day"] >= start_day) & (gd["day"] < end_day_excl)].copy()
gsel = gsel.sort_values("day")

if gsel.empty:
    st.warning("No data in selected window.")
    st.stop()

# =========================
# Charts
# =========================
left, right = st.columns([2, 1])

with left:
    plot_lines(gsel, "day", ["ASI"], "ASI over time")
    plot_lines(gsel, "day", ["members_total", "active_members", "traders"], "Members / Active / Traders")
    plot_lines(gsel, "day", ["volume_token", "revenue_token"], "Volume & Revenue (token)")
with right:
    plot_lines(gsel, "day", ["internal_share"], "Cohesion: internal trade share", height=300)
    plot_lines(gsel, "day", ["gini_volume", "top1_share_volume"], "Inequality: gini & top1 share", height=300)
    plot_lines(gsel, "day", ["ret7"], "Guild activity retention (ret7)", height=300)

download_df(gsel, f"guild_{guild_id}_day_metrics.csv", "Download selected guild day metrics CSV")

# =========================
# Shapley payouts
# =========================
st.subheader("Shapley payouts (approx)")
st.caption("Approximates marginal contribution to guild value (volume + internal synergy). Use smaller windows for speed.")

if st.button("Compute Shapley payouts"):
    with st.status("Computing Shapley (Monte Carlo)...", expanded=False) as status:
        prog = st.progress(0)
        prog.progress(10, text="Preparing trades subset...")
        trades = tables["market_trades"].copy()
        trades["day"] = floor_day(trades["ts"])
        # rebuild volume if needed
        if "volume_token" not in trades.columns:
            amt_col = ensure_col(trades, ["amount_base"], True, "market_trades")
            px_col = ensure_col(trades, ["price"], True, "market_trades")
            trades["amount_base"] = pd.to_numeric(trades[amt_col], errors="coerce").fillna(0.0)
            trades["price"] = pd.to_numeric(trades[px_col], errors="coerce").fillna(0.0)
            trades["volume_token"] = trades["amount_base"] * trades["price"]

        prog.progress(40, text="Running MC permutations...")
        shp = shapley_payouts_mc(
            trades=trades,
            user_day_guild=ud,
            guild_id=str(guild_id),
            start_day=start_day,
            end_day_excl=end_day_excl,
            n_perm=int(n_perm),
            alpha_synergy=0.3,
            random_seed=42,
        )
        prog.progress(100, text="Done")
        status.update(state="complete", label="Shapley ready")

    st.dataframe(shp.head(30), use_container_width=True)
    download_df(shp, f"guild_{guild_id}_shapley.csv", "Download Shapley CSV")

    if PLOTLY_OK and not shp.empty:
        fig = px.bar(shp.head(25), x="user_id", y="payout_share", title="Top Shapley payout shares (top 25)")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =========================
# Extra analyses
# =========================
st.subheader("Deeper diagnostics (5 extra analyses)")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "A1 Cohesion (internal/external)",
    "A3 Migration flows",
    "A4 Event impact",
    "A5 Churn risk model",
    "Network diagnostics",
])

with tab1:
    st.markdown("**Cohesion** is proxied by internal trading share (maker & taker in same guild).")
    st.write("Selected guild stats:")
    st.metric("Avg internal share", f"{gsel['internal_share'].mean():.2f}")
    st.metric("Avg volume per member", f"{gsel['volume_per_member'].mean():.2f}")
    st.metric("Avg ret7", f"{gsel['ret7'].mean():.2f}")

with tab2:
    mig = pack.migration.copy()
    mig = mig[(mig["day"] >= start_day) & (mig["day"] < end_day_excl)].copy()
    st.caption("User guild switching (from_guild -> to_guild).")
    st.dataframe(mig.sort_values("switch_users", ascending=False).head(50), use_container_width=True)
    download_df(mig, "alliances_migration_flows.csv", "Download migration flows CSV")

with tab3:
    ev = pack.event_impact.copy()
    if ev is None or ev.empty:
        st.info("No guild_events found or insufficient data for event impact.")
    else:
        ev2 = ev[(ev["guild_id"].astype(str) == str(guild_id)) & (ev["day"] >= start_day) & (ev["day"] < end_day_excl)].copy()
        st.caption("Event-day lift vs prior 7d mean (volume_lift, active_lift).")
        st.dataframe(ev2.sort_values("volume_lift", ascending=False).head(40), use_container_width=True)
        download_df(ev2, f"guild_{guild_id}_event_impact.csv", "Download event impact CSV")

with tab4:
    st.caption("Predict risk of *next-7d drop in active_members* (simple logistic model).")
    if not SK_OK:
        st.info("scikit-learn not available in your venv. Install scikit-learn to enable this.")
    else:
        # Build label: drop next 7d active members by > X%
        dfm = gd[gd["guild_id"] != "NO_GUILD"].copy()
        dfm = dfm.sort_values(["guild_id", "day"])
        dfm["active_next7"] = dfm.groupby("guild_id")["active_members"].shift(-7)
        dfm["drop_rate"] = safe_div(dfm["active_members"] - dfm["active_next7"], dfm["active_members"]).fillna(0.0)
        threshold = st.slider("Drop threshold (% in 7d)", 5, 80, 25, 5) / 100.0
        dfm["y"] = (dfm["drop_rate"] >= threshold).astype(int)

        features = ["ASI", "internal_share", "gini_volume", "top1_share_volume", "active_ratio", "volume_per_member", "ret7"]
        for f in features:
            dfm[f] = pd.to_numeric(dfm[f], errors="coerce").fillna(0.0)

        dfm = dfm.dropna(subset=["active_next7"]).copy()
        if dfm["y"].nunique() < 2:
            st.warning("Target has only one class in the current data. Try different threshold or date range.")
        else:
            # Train on all guilds, evaluate quick AUC (in-sample)
            X = dfm[features].to_numpy()
            y = dfm["y"].to_numpy()

            clf = LogisticRegression(max_iter=300)
            clf.fit(X, y)
            p = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
            st.metric("AUC (in-sample, quick)", f"{auc:.3f}")

            coefs = pd.DataFrame({"feature": features, "coef": clf.coef_[0]})
            coefs = coefs.sort_values("coef", ascending=False)
            st.dataframe(coefs, use_container_width=True)

            # Show guild risk timeline for selected guild
            gdf = gd[(gd["guild_id"].astype(str) == str(guild_id))].copy().sort_values("day")
            for f in features:
                gdf[f] = pd.to_numeric(gdf[f], errors="coerce").fillna(0.0)
            pp = clf.predict_proba(gdf[features].to_numpy())[:, 1]
            gdf["risk_drop_next7"] = pp
            gdf = gdf[(gdf["day"] >= start_day) & (gdf["day"] < end_day_excl)]
            plot_lines(gdf, "day", ["risk_drop_next7"], "Predicted risk: active_members drop in next 7 days", height=320)
            download_df(gdf[["day","guild_id","risk_drop_next7"]], f"guild_{guild_id}_risk.csv", "Download risk series CSV")

with tab5:
    st.caption("Network diagnostics: build user-user graph from internal trades (requires networkx).")
    if not use_network:
        st.info("Network diagnostics disabled in sidebar.")
    elif not NX_OK:
        st.info("networkx not available. Install networkx to enable this.")
    else:
        # Build graph for selected guild and window
        trades = tables["market_trades"].copy()
        trades["day"] = floor_day(trades["ts"])
        maker = ensure_col(trades, ["maker_user_id"], True, "market_trades")
        taker = ensure_col(trades, ["taker_user_id"], True, "market_trades")

        # attach guild for maker & taker on day using ud mapping
        map_ud = ud[["user_id", "day", "guild_id"]].copy()
        map_m = map_ud.rename(columns={"user_id": maker, "guild_id": "guild_m"})
        map_t = map_ud.rename(columns={"user_id": taker, "guild_id": "guild_t"})

        t2 = trades[(trades["day"] >= start_day) & (trades["day"] < end_day_excl)].copy()
        t2 = t2.merge(map_m, on=[maker, "day"], how="left").merge(map_t, on=[taker, "day"], how="left")
        t2["guild_m"] = t2["guild_m"].fillna("NO_GUILD")
        t2["guild_t"] = t2["guild_t"].fillna("NO_GUILD")

        # internal edges within selected guild
        t2 = t2[(t2["guild_m"].astype(str) == str(guild_id)) & (t2["guild_t"].astype(str) == str(guild_id))].copy()
        if t2.empty:
            st.info("No internal trades found for this guild in the selected window.")
        else:
            # edge weights = trade count (or volume if present)
            wcol = "amount_base" if "amount_base" in t2.columns else None
            if "price" in t2.columns and "amount_base" in t2.columns:
                t2["vol"] = pd.to_numeric(t2["amount_base"], errors="coerce").fillna(0.0) * pd.to_numeric(t2["price"], errors="coerce").fillna(0.0)
                wcol = "vol"
            else:
                t2["w"] = 1.0
                wcol = "w"

            edges = t2.groupby([maker, taker])[wcol].sum().reset_index()

            G = nx.Graph()
            for r in edges.itertuples(index=False):
                a, b, w = r[0], r[1], float(r[2])
                if a == b:
                    continue
                if G.has_edge(a, b):
                    G[a][b]["weight"] += w
                else:
                    G.add_edge(a, b, weight=w)

            n = G.number_of_nodes()
            m = G.number_of_edges()
            density = nx.density(G) if n > 1 else 0.0
            deg = dict(G.degree())
            avg_deg = float(np.mean(list(deg.values()))) if deg else 0.0

            st.metric("Nodes", n)
            st.metric("Edges", m)
            st.metric("Density", f"{density:.4f}")
            st.metric("Avg degree", f"{avg_deg:.2f}")

            # Centrality
            dc = nx.degree_centrality(G) if n > 1 else {}
            top = pd.DataFrame({"user_id": list(dc.keys()), "degree_centrality": list(dc.values())})
            top = top.sort_values("degree_centrality", ascending=False).head(30)
            st.dataframe(top, use_container_width=True)
            download_df(top, f"guild_{guild_id}_network_centrality.csv", "Download centrality CSV")

# =========================
# Debug
# =========================
if show_debug:
    st.subheader("Debug tables")
    st.write("user_day_guild (head)", ud.head(10))
    st.write("guild_day (head)", gd.head(10))
