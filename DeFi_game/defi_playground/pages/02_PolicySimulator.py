# pages/02_PolicySimulator.py
# Policy Simulator — Resource-level taxes + Progressive scale + Switchback experiments + Optimization (constraints)
# NOTE: This page is intentionally lightweight. Economy/Market dashboards live elsewhere.
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
# Page setup
# =========================
st.set_page_config(page_title="Policy Simulator", layout="wide")
st.title("Policy Simulator — Resource-level taxes + Switchback experiments")
st.caption("What-if policy • Progressive taxes by volume bucket • Constraints optimization • Switchback/time-based experiments")


# =========================
# Helpers
# =========================
def _to_dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _floor_day(s) -> pd.Series:
    return _to_dt(s).dt.floor("D")

def _pick(df: pd.DataFrame, candidates: list[str], required: bool = True, df_name: str = "df") -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"{df_name}: missing one of {candidates}. Available: {list(df.columns)}")
    return None

def _find_csv(roots: list[str], filename: str) -> str | None:
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
    return pd.read_csv(p)

def first_not_none(*items):
    for x in items:
        if x is not None:
            return x
    return None

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd

def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)

def bootstrap_ci(diff_samples: np.ndarray, alpha=0.05):
    diff_samples = diff_samples[np.isfinite(diff_samples)]
    if len(diff_samples) == 0:
        return (np.nan, np.nan)
    lo = np.quantile(diff_samples, alpha/2)
    hi = np.quantile(diff_samples, 1 - alpha/2)
    return float(lo), float(hi)

def ols_beta(X: np.ndarray, y: np.ndarray):
    """
    Lightweight OLS: returns coef, intercept, r2.
    No p-values (keeps dependencies minimal).
    """
    # add intercept
    X = np.asarray(X)
    y = np.asarray(y)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    if len(y) < 10:
        return None
    X1 = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.lstsq(X1, y, rcond=None)[0]
    yhat = X1 @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"intercept": float(coef[0]), "coef": coef[1:].astype(float), "r2": float(r2), "n": int(len(y))}

def make_switchback_assignment(days: pd.Series, block_len_days: int, start_variant: str = "A"):
    """
    Assigns each day to A/B in contiguous blocks.
    """
    days = pd.Series(pd.to_datetime(days)).dropna().sort_values().unique()
    if len(days) == 0:
        return pd.DataFrame(columns=["day", "variant", "block_id"])
    days = pd.to_datetime(days)
    block_id = np.arange(len(days)) // max(1, int(block_len_days))
    if start_variant.upper() not in ("A", "B"):
        start_variant = "A"
    start = 0 if start_variant.upper() == "A" else 1
    variant = np.where((block_id + start) % 2 == 0, "A", "B")
    return pd.DataFrame({"day": days, "variant": variant, "block_id": block_id.astype(int)})

def parse_bucket_order(bucket_series: pd.Series) -> list[str]:
    """
    Try to order bucket labels (string) in a stable way.
    If labels look like '0-100', '100-1k', use a numeric key from the left bound.
    Else: fallback to sorted unique.
    """
    vals = pd.Series(bucket_series).dropna().astype(str).unique().tolist()

    def key(v):
        v = v.strip()
        # extract left bound number if possible
        # examples: "0-100", "100-1000", "1k-5k"
        import re
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)(k|m)?", v.lower())
        if not m:
            return (1e18, v)
        num = float(m.group(1))
        suf = m.group(2)
        if suf == "k":
            num *= 1_000
        elif suf == "m":
            num *= 1_000_000
        return (num, v)

    return [x for _, x in sorted([key(v) for v in vals], key=lambda t: (t[0], t[1]))]


# =========================
# Sidebar: paths & load
# =========================
st.sidebar.header("Data paths")

default_extra = r"C:\Users\Shaim\defi game\synthetic_data"
default_main  = r"C:\Users\Shaim\defi game\synthetic_defi_game_data"

ROOT_EXTRA = st.sidebar.text_input("Extra synthetic data folder", value=default_extra)
ROOT_MAIN  = st.sidebar.text_input("Main synthetic data folder", value=default_main)
ROOTS = (ROOT_MAIN, ROOT_EXTRA)

st.sidebar.caption("Files are auto-loaded from the folders above.")

# Core
users         = load_csv_any(ROOTS, "users.csv")
sessions      = load_csv_any(ROOTS, "sessions.csv")
market_trades = load_csv_any(ROOTS, "market_trades.csv")
token_ledger  = load_csv_any(ROOTS, "token_ledger.csv")

# Enriched (optional but recommended)
price_oracle = first_not_none(
    load_csv_any(ROOTS, "price_oracle.csv"),
    load_csv_any(ROOTS, "price_oracle_daily.csv"),
    load_csv_any(ROOTS, "price_oracle_enriched.csv"),
)
econ_events = first_not_none(
    load_csv_any(ROOTS, "econ_events.csv"),
    load_csv_any(ROOTS, "economic_events.csv"),
)

# Minimal requirement for policy simulation
if market_trades is None or users is None:
    st.error("Missing required CSVs: users.csv and/or market_trades.csv. Please place them in one of the configured folders.")
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


# =========================
# Normalize inputs
# =========================
status = st.status("Preparing data…", expanded=True)
prog = st.progress(0)

# Users
status.write("Parsing users…")
u = users.copy()
u_id = _pick(u, ["user_id", "id"], df_name="users")
u_created = _pick(u, ["created_at", "signup_ts", "registered_at"], df_name="users")
u[u_created] = _to_dt(u[u_created])

# Common segments present in your data
# ['acq_channel','risk_profile','whale_flag','player_segment', ...]
SEGMENT_COLS = [c for c in ["whale_flag", "acq_channel", "risk_profile", "player_segment", "country", "device_os"] if c in u.columns]

prog.progress(10)

# Trades
status.write("Parsing market_trades…")
mt = market_trades.copy()
mt_ts = _pick(mt, ["ts", "timestamp"], df_name="market_trades")
mt["ts"] = _to_dt(mt[mt_ts])
mt["day"] = mt["ts"].dt.floor("D")

# resource
res_col = _pick(mt, ["asset_base", "resource", "base_asset", "asset"], df_name="market_trades")
mt["resource"] = mt[res_col].astype(str).str.upper()

amt_base = _pick(mt, ["amount_base", "qty_base", "base_amount"], df_name="market_trades")
price_col = _pick(mt, ["price", "px"], df_name="market_trades")

mt[amt_base] = pd.to_numeric(mt[amt_base], errors="coerce").fillna(0.0)
mt[price_col] = pd.to_numeric(mt[price_col], errors="coerce").fillna(0.0)

mt["gross_token"] = mt[amt_base] * mt[price_col]

maker = _pick(mt, ["maker_user_id", "user_id"], required=False, df_name="market_trades")
fee_col = _pick(mt, ["fee_amount"], required=False, df_name="market_trades")
tax_col = _pick(mt, ["tax_amount"], required=False, df_name="market_trades")
slip_col = _pick(mt, ["slippage_bps"], required=False, df_name="market_trades")
spr_col  = _pick(mt, ["spread_bps"], required=False, df_name="market_trades")
bucket_col = _pick(mt, ["user_7d_volume_bucket"], required=False, df_name="market_trades")

for c in [fee_col, tax_col, slip_col, spr_col]:
    if c and c in mt.columns:
        mt[c] = pd.to_numeric(mt[c], errors="coerce")

prog.progress(25)

# Sessions (optional, used to estimate "active users" denominator for extensive margin)
if sessions is not None:
    status.write("Parsing sessions (for active user denominator)…")
    s = sessions.copy()
    s_user = _pick(s, ["user_id"], df_name="sessions")
    s_ts = _pick(s, ["start_ts", "ts"], df_name="sessions")
    s[s_ts] = _to_dt(s[s_ts])
    s["day"] = s[s_ts].dt.floor("D")
    # attach segments
    s = s.merge(u[[u_id] + SEGMENT_COLS], left_on=s_user, right_on=u_id, how="left")
else:
    s = None

prog.progress(35)

# Price oracle (optional)
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

prog.progress(45)

# Econ events (optional)
ev = None
if econ_events is not None:
    status.write("Parsing econ_events…")
    ev = econ_events.copy()
    ev_start = _pick(ev, ["start_ts", "start", "start_time"], df_name="econ_events")
    ev_end   = _pick(ev, ["end_ts", "end", "end_time"], df_name="econ_events")
    ev_res   = _pick(ev, ["affected_resource", "resource", "asset_base"], required=False, df_name="econ_events")

    ev[ev_start] = _to_dt(ev[ev_start])
    ev[ev_end]   = _to_dt(ev[ev_end])
    ev["start_day"] = ev[ev_start].dt.floor("D")
    ev["end_day"]   = ev[ev_end].dt.floor("D")
    if ev_res:
        ev["resource"] = ev[ev_res].astype(str).str.upper()
    else:
        ev["resource"] = None

prog.progress(55)

# Date bounds
date_min = mt["day"].min()
date_max = mt["day"].max()
if pd.isna(date_min) or pd.isna(date_max):
    st.error("Could not infer date range from market_trades.")
    st.stop()

status.update(label="Data prepared.", state="complete", expanded=False)
prog.progress(60)


# =========================
# Sidebar filters
# =========================
st.sidebar.header("Simulation scope")
date_from, date_to = st.sidebar.date_input(
    "Date range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date(),
)

date_from = pd.Timestamp(date_from)
date_to_excl = pd.Timestamp(date_to) + pd.Timedelta(days=1)

resources_all = sorted(mt["resource"].dropna().unique().tolist())
resource_pick = st.sidebar.multiselect(
    "Resources",
    options=resources_all,
    default=resources_all[: min(6, len(resources_all))],
)

segment_mode = st.sidebar.selectbox(
    "Segment by",
    options=["None"] + SEGMENT_COLS + (["user_7d_volume_bucket"] if bucket_col else []),
    index=0
)

st.sidebar.divider()
st.sidebar.header("Policy inputs")

tax_rate_A = st.sidebar.slider("Tax rate A (bps)", min_value=0, max_value=500, value=50, step=5) / 10_000.0
tax_rate_B = st.sidebar.slider("Tax rate B (bps)", min_value=0, max_value=500, value=80, step=5) / 10_000.0

use_progressive = st.sidebar.checkbox("Use progressive tax by 7d volume bucket", value=False)
progressive_base_bps = st.sidebar.slider("Progressive base (bps)", 0, 500, 40, 5) / 10_000.0
progressive_slope_bps = st.sidebar.slider("Progressive slope per bucket step (bps)", -50, 100, 5, 5) / 10_000.0

st.sidebar.divider()
st.sidebar.header("Switchback design")
block_len_days = st.sidebar.select_slider("Block length (days)", options=[1,2,3,5,7,10,14,21,28], value=7)
start_variant = st.sidebar.radio("Start variant", options=["A","B"], horizontal=True)

st.sidebar.divider()
st.sidebar.header("Optimization")
objective = st.sidebar.selectbox(
    "Objective",
    options=["Max tax revenue", "Max volume", "Max traders (extensive)", "Max utility (revenue + w*volume)"],
    index=0
)
w_volume = st.sidebar.slider("Utility weight for volume (if used)", 0.0, 5.0, 1.0, 0.1)

constraint_on = st.sidebar.checkbox("Constraint: revenue ≥ X", value=False)
constraint_rev = st.sidebar.number_input("X (token units per day, avg)", min_value=0.0, value=0.0, step=10.0)

grid_min_bps = st.sidebar.slider("Grid min tax (bps)", 0, 500, 10, 5)
grid_max_bps = st.sidebar.slider("Grid max tax (bps)", 0, 500, 150, 5)
grid_step_bps = st.sidebar.select_slider("Grid step (bps)", options=[1,2,5,10,25], value=5)

run_btn = st.sidebar.button("Run simulation", type="primary")


# =========================
# Core panel builder
# =========================
@st.cache_data(show_spinner=False)
def build_day_resource_segment_panel(
    mt: pd.DataFrame,
    users: pd.DataFrame,
    sessions: pd.DataFrame | None,
    po: pd.DataFrame | None,
    ev: pd.DataFrame | None,
    date_from: pd.Timestamp,
    date_to_excl: pd.Timestamp,
    resources: list[str],
    segment_mode: str,
):
    """
    Build a day×resource×segment panel for policy analysis.
    Keeps it compact (no user-level full cross join).
    """
    mt0 = mt[(mt["day"] >= date_from) & (mt["day"] < date_to_excl)].copy()
    if resources:
        mt0 = mt0[mt0["resource"].isin(resources)].copy()

    # attach user segments to trades (maker side)
    # (We only need segments for grouping; this is lightweight)
    seg_cols = [c for c in ["whale_flag","acq_channel","risk_profile","player_segment","country","device_os"] if c in users.columns]
    if maker and maker in mt0.columns:
        mt0 = mt0.merge(users[[u_id] + seg_cols], left_on=maker, right_on=u_id, how="left")
    else:
        # if maker missing, still proceed without segments
        for c in seg_cols:
            mt0[c] = None

    # segment key
    if segment_mode == "None":
        mt0["segment"] = "ALL"
    elif segment_mode == "user_7d_volume_bucket" and bucket_col and bucket_col in mt0.columns:
        mt0["segment"] = mt0[bucket_col].astype(str)
    else:
        mt0["segment"] = mt0[segment_mode].astype(str)

    # Agg: day×resource×segment
    gcols = ["day", "resource", "segment"]
    agg = {
        "gross_token": "sum",
        "trade_id": "count",
    }
    out = mt0.groupby(gcols).agg(volume_token=("gross_token","sum"),
                                 trades=("trade_id","count")).reset_index()

    # taxes/fees (sum + effective rate)
    if tax_col and tax_col in mt0.columns:
        tax_sum = mt0.groupby(gcols)[tax_col].sum().reset_index(name="tax_token")
        out = out.merge(tax_sum, on=gcols, how="left")
    else:
        out["tax_token"] = 0.0

    if fee_col and fee_col in mt0.columns:
        fee_sum = mt0.groupby(gcols)[fee_col].sum().reset_index(name="fee_token")
        out = out.merge(fee_sum, on=gcols, how="left")
    else:
        out["fee_token"] = 0.0

    out["tax_eff_rate"] = safe_div(out["tax_token"], out["volume_token"]).fillna(0.0)
    out["fee_eff_rate"] = safe_div(out["fee_token"], out["volume_token"]).fillna(0.0)

    # Traders (unique makers)
    if maker and maker in mt0.columns:
        traders = mt0.groupby(gcols)[maker].nunique().reset_index(name="traders")
        out = out.merge(traders, on=gcols, how="left")
    else:
        out["traders"] = np.nan

    out["volume_per_trader"] = safe_div(out["volume_token"], out["traders"]).fillna(0.0)

    # Volume-weighted slippage/spread
    if slip_col and slip_col in mt0.columns:
        tmp = mt0.copy()
        tmp["w"] = tmp["gross_token"].clip(lower=0.0)
        slip_vw = tmp.groupby(gcols).apply(
            lambda g: (g[slip_col].fillna(0.0) * g["w"]).sum() / g["w"].sum() if g["w"].sum() > 0 else np.nan
        ).reset_index()
        slip_vw.columns = gcols + ["slippage_bps_vw"]
        out = out.merge(slip_vw, on=gcols, how="left")
    else:
        out["slippage_bps_vw"] = np.nan

    if spr_col and spr_col in mt0.columns:
        tmp = mt0.copy()
        tmp["w"] = tmp["gross_token"].clip(lower=0.0)
        spr_vw = tmp.groupby(gcols).apply(
            lambda g: (g[spr_col].fillna(0.0) * g["w"]).sum() / g["w"].sum() if g["w"].sum() > 0 else np.nan
        ).reset_index()
        spr_vw.columns = gcols + ["spread_bps_vw"]
        out = out.merge(spr_vw, on=gcols, how="left")
    else:
        out["spread_bps_vw"] = np.nan

    # Oracle merge: day×resource
    if po is not None:
        keep = ["day","resource"]
        if "volatility" in po.columns: keep.append("volatility")
        if "price_token" in po.columns: keep.append("price_token")
        po2 = po[keep].copy()
        out = out.merge(po2, on=["day","resource"], how="left")
    if "volatility" not in out.columns:
        out["volatility"] = 0.0
    out["day_volatility"] = pd.to_numeric(out["volatility"], errors="coerce").fillna(0.0)

    # Events/shocks: mark 1 if any event overlaps day for resource (or global event)
    if ev is not None and len(ev) > 0:
        ev2 = ev.copy()
        # expand to day rows
        # lightweight loop over events (few rows)
        out["shock"] = 0
        for _, r in ev2.iterrows():
            sd = r.get("start_day", pd.NaT)
            ed = r.get("end_day", pd.NaT)
            rr = r.get("resource", None)
            if pd.isna(sd) or pd.isna(ed):
                continue
            mask_day = (out["day"] >= sd) & (out["day"] <= ed)
            if rr is None or (isinstance(rr, float) and np.isnan(rr)):
                out.loc[mask_day, "shock"] = 1
            else:
                out.loc[mask_day & (out["resource"] == str(rr).upper()), "shock"] = 1
    else:
        out["shock"] = 0

    # Extensive margin denominator: active users by day×segment (from sessions)
    if sessions is not None and "day" in sessions.columns:
        s0 = sessions[(sessions["day"] >= date_from) & (sessions["day"] < date_to_excl)].copy()

        # segment for sessions
        if segment_mode == "None":
            s0["segment"] = "ALL"
        elif segment_mode == "user_7d_volume_bucket":
            # sessions do not have bucket; fallback to ALL
            s0["segment"] = "ALL"
        else:
            if segment_mode in s0.columns:
                s0["segment"] = s0[segment_mode].astype(str)
            else:
                s0["segment"] = "ALL"

        active = s0.groupby(["day","segment"])["user_id"].nunique().reset_index(name="active_users")
        out = out.merge(active, on=["day","segment"], how="left")
        out["active_users"] = out["active_users"].fillna(0.0)
    else:
        out["active_users"] = np.nan

    # Trade probability proxy
    # If active_users is available: traders/active_users else NaN
    out["trade_prob"] = safe_div(out["traders"], out["active_users"])
    out["trade_prob"] = out["trade_prob"].clip(lower=0, upper=1)

    # log transforms
    out["log_volume"] = np.log1p(out["volume_token"])
    out["log_vpt"] = np.log1p(out["volume_per_trader"])
    out["log_traders"] = np.log1p(out["traders"].fillna(0.0))

    return out


# =========================
# Modeling: elasticities (panel-level)
# =========================
def estimate_elasticities(panel: pd.DataFrame):
    """
    Estimate simple elasticities vs effective tax rate:
    - intensive: log(volume_per_trader+1) ~ a + b*tax_eff_rate + c*volatility + d*shock
    - extensive: log(traders+1) ~ a + b*tax_eff_rate + c*volatility + d*shock
    These are not causal proofs — they are pragmatic knobs for policy what-if.
    """
    df = panel.copy()

    # Ensure numeric
    for c in ["tax_eff_rate","day_volatility","shock","log_vpt","log_traders"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Drop empty rows
    df = df[df["volume_token"].fillna(0.0) >= 0].copy()

    X = df[["tax_eff_rate","day_volatility","shock"]].to_numpy(dtype=float)

    # intensive
    y_int = df["log_vpt"].to_numpy(dtype=float)
    res_int = ols_beta(X, y_int)

    # extensive (traders)
    y_ext = df["log_traders"].to_numpy(dtype=float)
    res_ext = ols_beta(X, y_ext)

    return {"intensive": res_int, "extensive": res_ext}


# =========================
# Policy what-if simulation
# =========================
def apply_policy_tax_rate(panel: pd.DataFrame, use_progressive: bool, base_rate: float, slope_per_bucket: float):
    """
    Create 'policy_tax_rate' for each row.
    If progressive: depends on bucket order if segment_mode == bucket; otherwise base.
    """
    df = panel.copy()
    df["policy_tax_rate"] = base_rate

    if use_progressive:
        # Only meaningful if segment is bucket-like
        # We approximate by mapping segment labels to an ordered index and applying linear slope.
        buckets = parse_bucket_order(df["segment"])
        idx_map = {b: i for i, b in enumerate(buckets)}
        df["bucket_idx"] = df["segment"].map(idx_map).fillna(0).astype(int)
        df["policy_tax_rate"] = base_rate + df["bucket_idx"] * slope_per_bucket
        df["policy_tax_rate"] = df["policy_tax_rate"].clip(lower=0.0)
    else:
        df["bucket_idx"] = 0

    return df


def predict_under_policy(panel: pd.DataFrame, elas: dict, tax_rate: float, use_progressive: bool, base_rate: float, slope: float):
    """
    Predict new (traders, volume_per_trader, volume, revenue) under policy rate(s).
    Uses multiplicative model: y_new = y_old * exp(beta_tax*(tax_new - tax_old))
    """
    df = panel.copy()

    # Fill missing columns
    for c in ["tax_eff_rate","day_volatility","shock","traders","volume_per_trader","active_users"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Policy tax rate per row
    if use_progressive:
        df = apply_policy_tax_rate(df, True, base_rate, slope)
    else:
        df["policy_tax_rate"] = float(tax_rate)

    # Extract elasticity b for tax from models (tax is first feature)
    b_int = 0.0
    b_ext = 0.0
    if elas.get("intensive") and elas["intensive"] is not None:
        b_int = float(elas["intensive"]["coef"][0])
    if elas.get("extensive") and elas["extensive"] is not None:
        b_ext = float(elas["extensive"]["coef"][0])

    d_tax = (df["policy_tax_rate"] - df["tax_eff_rate"]).astype(float)

    # extensive: traders
    traders_old = df["traders"].astype(float)
    traders_new = traders_old * np.exp(b_ext * d_tax)
    traders_new = np.clip(traders_new, 0, None)

    # intensive: volume_per_trader
    vpt_old = df["volume_per_trader"].astype(float)
    vpt_new = vpt_old * np.exp(b_int * d_tax)
    vpt_new = np.clip(vpt_new, 0, None)

    # If we have active_users, cap traders_new <= active_users
    if df["active_users"].notna().any():
        traders_new = np.minimum(traders_new, df["active_users"].fillna(np.inf))

    vol_new = traders_new * vpt_new

    # Revenue approximation: tax_rate * volume
    rev_new = df["policy_tax_rate"].astype(float) * vol_new

    out = df.copy()
    out["pred_traders"] = traders_new
    out["pred_vpt"] = vpt_new
    out["pred_volume"] = vol_new
    out["pred_revenue"] = rev_new

    # Baseline
    out["base_revenue"] = df["tax_eff_rate"] * df["volume_token"].astype(float)
    out["base_volume"] = df["volume_token"].astype(float)
    out["base_traders"] = traders_old

    return out


# =========================
# Optimization (grid over tax)
# =========================
def optimize_tax(panel_pred_base: pd.DataFrame, elas: dict,
                 grid_bps_min: int, grid_bps_max: int, grid_step_bps: int,
                 objective: str, w_volume: float,
                 constraint_on: bool, constraint_rev: float):
    """
    Simple grid search over a single tax rate applied to all rows (or progressive disabled).
    Returns best_rate, results_df.
    """
    rates = np.arange(grid_bps_min, grid_bps_max + 1e-9, grid_step_bps) / 10_000.0
    rows = []
    best = None

    for r in rates:
        pred = predict_under_policy(panel_pred_base, elas, tax_rate=float(r),
                                    use_progressive=False, base_rate=float(r), slope=0.0)

        # aggregate by day (average daily)
        daily = pred.groupby("day").agg(
            revenue=("pred_revenue","sum"),
            volume=("pred_volume","sum"),
            traders=("pred_traders","sum"),
        )
        avg_rev = float(daily["revenue"].mean()) if len(daily) else 0.0
        avg_vol = float(daily["volume"].mean()) if len(daily) else 0.0
        avg_tr = float(daily["traders"].mean()) if len(daily) else 0.0

        # constraint
        feasible = True
        if constraint_on:
            feasible = avg_rev >= float(constraint_rev)

        # objective
        if objective == "Max tax revenue":
            score = avg_rev
        elif objective == "Max volume":
            score = avg_vol
        elif objective == "Max traders (extensive)":
            score = avg_tr
        else:
            score = avg_rev + w_volume * avg_vol

        rows.append({"tax_rate": r, "avg_revenue": avg_rev, "avg_volume": avg_vol, "avg_traders": avg_tr,
                     "feasible": feasible, "score": score if feasible else -np.inf})

        if feasible and (best is None or score > best["score"]):
            best = {"tax_rate": r, "avg_revenue": avg_rev, "avg_volume": avg_vol, "avg_traders": avg_tr, "score": score}

    res = pd.DataFrame(rows)
    return best, res


# =========================
# Main run
# =========================
if run_btn:
    run_status = st.status("Running Policy Simulator…", expanded=True)
    p = st.progress(0)

    run_status.write("Building day×resource×segment panel…")
    panel = build_day_resource_segment_panel(
        mt=mt,
        users=u,
        sessions=s,
        po=po,
        ev=ev,
        date_from=date_from,
        date_to_excl=date_to_excl,
        resources=resource_pick,
        segment_mode=segment_mode,
    )
    p.progress(20)

    if panel.empty:
        st.warning("Panel is empty for the selected filters.")
        run_status.update(state="complete")
        st.stop()

    run_status.write("Estimating elasticities (lightweight OLS)…")
    elas = estimate_elasticities(panel)
    p.progress(35)

    # Show elasticity summary
    st.subheader("Estimated elasticities (lightweight)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Intensive margin model**: log(volume_per_trader+1) ~ tax + volatility + shock")
        st.write(elas["intensive"] if elas["intensive"] is not None else "Not enough data for model.")
    with c2:
        st.markdown("**Extensive margin model**: log(traders+1) ~ tax + volatility + shock")
        st.write(elas["extensive"] if elas["extensive"] is not None else "Not enough data for model.")

    st.divider()

    # Switchback assignment (based on available days in panel)
    run_status.write("Creating switchback assignment (A/B blocks)…")
    assign = make_switchback_assignment(panel["day"], block_len_days=int(block_len_days), start_variant=start_variant)
    p.progress(45)

    # Apply switchback tax rates
    run_status.write("Applying A/B tax rates and simulating outcomes…")
    panel_sw = panel.merge(assign[["day","variant"]], on="day", how="left")
    panel_sw["variant"] = panel_sw["variant"].fillna("A")

    # Use progressive policy if enabled (only meaningful when segment=bucket)
    if use_progressive:
        # A and B are parameterized via base and slope
        # (we keep A/B tax_rate sliders but progressive uses base/slope instead)
        predA = predict_under_policy(panel_sw[panel_sw["variant"]=="A"], elas, tax_rate=tax_rate_A,
                                     use_progressive=True, base_rate=progressive_base_bps, slope=progressive_slope_bps)
        predB = predict_under_policy(panel_sw[panel_sw["variant"]=="B"], elas, tax_rate=tax_rate_B,
                                     use_progressive=True, base_rate=progressive_base_bps, slope=progressive_slope_bps)
        pred = pd.concat([predA, predB], ignore_index=True)
    else:
        predA = predict_under_policy(panel_sw[panel_sw["variant"]=="A"], elas, tax_rate=tax_rate_A,
                                     use_progressive=False, base_rate=tax_rate_A, slope=0.0)
        predB = predict_under_policy(panel_sw[panel_sw["variant"]=="B"], elas, tax_rate=tax_rate_B,
                                     use_progressive=False, base_rate=tax_rate_B, slope=0.0)
        pred = pd.concat([predA, predB], ignore_index=True)

    p.progress(60)

    # Aggregate by day and variant
    run_status.write("Aggregating A/B results and computing deltas…")
    daily_ab = pred.groupby(["day","variant"]).agg(
        pred_revenue=("pred_revenue","sum"),
        pred_volume=("pred_volume","sum"),
        pred_traders=("pred_traders","sum"),
        base_volume=("base_volume","sum"),
        base_revenue=("base_revenue","sum"),
    ).reset_index()

    # pivot for deltas
    piv = daily_ab.pivot(index="day", columns="variant", values=["pred_revenue","pred_volume","pred_traders"]).sort_index()
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    for m in ["pred_revenue","pred_volume","pred_traders"]:
        if f"{m}_A" in piv.columns and f"{m}_B" in piv.columns:
            piv[f"delta_{m}"] = piv[f"{m}_B"] - piv[f"{m}_A"]
            piv[f"pct_{m}"] = safe_div(piv[f"delta_{m}"], piv[f"{m}_A"]).replace([np.inf, -np.inf], np.nan)

    p.progress(75)

    # Bootstrap CI on daily delta (simple)
    run_status.write("Bootstrapping daily deltas (simple CI)…")
    rng = np.random.default_rng(42)
    ci = {}
    for m in ["pred_revenue","pred_volume","pred_traders"]:
        col = f"delta_{m}"
        if col in piv.columns:
            x = piv[col].dropna().to_numpy()
            if len(x) >= 10:
                boots = []
                for _ in range(500):
                    samp = rng.choice(x, size=len(x), replace=True)
                    boots.append(np.mean(samp))
                lo, hi = bootstrap_ci(np.array(boots), alpha=0.05)
                ci[m] = {"mean_delta": float(np.mean(x)), "ci95": (lo, hi), "days": int(len(x))}
            else:
                ci[m] = {"mean_delta": float(np.mean(x)) if len(x) else np.nan, "ci95": (np.nan, np.nan), "days": int(len(x))}
    p.progress(85)

    # Optimization
    run_status.write("Running tax optimization grid (single-rate)…")
    best, grid = optimize_tax(
        panel_pred_base=panel,
        elas=elas,
        grid_bps_min=int(grid_min_bps),
        grid_bps_max=int(grid_max_bps),
        grid_step_bps=int(grid_step_bps),
        objective=objective,
        w_volume=float(w_volume),
        constraint_on=bool(constraint_on),
        constraint_rev=float(constraint_rev),
    )
    p.progress(95)

    run_status.update(label="Done.", state="complete", expanded=False)
    p.progress(100)

    # =========================
    # UI Output
    # =========================
    st.subheader("Switchback A/B (simulated via elasticities)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Tax A (bps)", f"{tax_rate_A*10_000:.0f}")
    c2.metric("Tax B (bps)", f"{tax_rate_B*10_000:.0f}")
    c3.metric("Block length (days)", f"{block_len_days}")

    st.markdown("**Daily delta summary (B − A)**")
    st.write(ci)

    tab1, tab2, tab3 = st.tabs(["Time series", "Daily table", "By resource/segment"])

    with tab1:
        if PLOTLY_OK:
            fig = go.Figure()
            if "delta_pred_volume" in piv.columns:
                fig.add_trace(go.Scatter(x=piv["day"], y=piv["delta_pred_volume"], name="Δ volume (B-A)"))
            if "delta_pred_revenue" in piv.columns:
                fig.add_trace(go.Scatter(x=piv["day"], y=piv["delta_pred_revenue"], name="Δ revenue (B-A)", yaxis="y2"))

            fig.update_layout(
                height=420,
                xaxis_title="day",
                yaxis=dict(title="Δ volume"),
                yaxis2=dict(title="Δ revenue", overlaying="y", side="right"),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            cols = [c for c in ["delta_pred_volume","delta_pred_revenue","delta_pred_traders"] if c in piv.columns]
            st.line_chart(piv.set_index("day")[cols])

    with tab2:
        st.dataframe(piv.tail(60), use_container_width=True)

    with tab3:
        # breakdown by resource/segment (average over all days)
        br = pred.groupby(["variant","resource","segment"]).agg(
            avg_volume=("pred_volume","mean"),
            avg_revenue=("pred_revenue","mean"),
            avg_traders=("pred_traders","mean"),
        ).reset_index()

        # pivot delta
        brp = br.pivot_table(index=["resource","segment"], columns="variant", values=["avg_volume","avg_revenue","avg_traders"], aggfunc="mean")
        brp.columns = [f"{a}_{b}" for a, b in brp.columns]
        brp = brp.reset_index()

        for m in ["avg_volume","avg_revenue","avg_traders"]:
            if f"{m}_A" in brp.columns and f"{m}_B" in brp.columns:
                brp[f"delta_{m}"] = brp[f"{m}_B"] - brp[f"{m}_A"]
                brp[f"pct_{m}"] = safe_div(brp[f"delta_{m}"], brp[f"{m}_A"]).replace([np.inf, -np.inf], np.nan)

        st.dataframe(brp.sort_values("delta_avg_revenue", ascending=False).head(200), use_container_width=True)

        if PLOTLY_OK and "delta_avg_revenue" in brp.columns:
            top = brp.sort_values("delta_avg_revenue", ascending=False).head(30).copy()
            top["label"] = top["resource"].astype(str) + " | " + top["segment"].astype(str)
            figb = px.bar(top, x="label", y="delta_avg_revenue", title="Top 30 resource×segment by Δ revenue (B-A)")
            figb.update_layout(height=420, xaxis_title="", yaxis_title="Δ revenue")
            st.plotly_chart(figb, use_container_width=True)

    st.divider()

    st.subheader("Optimization (single tax rate grid)")
    if best is None:
        st.warning("No feasible solution under the current constraint settings.")
    else:
        st.success(
            f"Best tax rate: {best['tax_rate']*10_000:.0f} bps | "
            f"avg revenue/day={best['avg_revenue']:.2f}, avg volume/day={best['avg_volume']:.2f}, avg traders/day={best['avg_traders']:.2f}"
        )

    st.dataframe(grid.sort_values("score", ascending=False).head(50), use_container_width=True)

    if PLOTLY_OK:
        g2 = grid.copy()
        figg = go.Figure()
        figg.add_trace(go.Scatter(x=g2["tax_rate"]*10_000, y=g2["avg_revenue"], name="avg_revenue"))
        figg.add_trace(go.Scatter(x=g2["tax_rate"]*10_000, y=g2["avg_volume"], name="avg_volume", yaxis="y2"))
        figg.update_layout(
            height=420,
            xaxis_title="tax rate (bps)",
            yaxis=dict(title="avg revenue/day"),
            yaxis2=dict(title="avg volume/day", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(figg, use_container_width=True)

else:
    st.info("Set filters/policy parameters in the sidebar and click **Run simulation**.")
    st.markdown(
        """
**What this page does (by design):**
- Builds a compact **day×resource×segment** panel (no heavy user-level joins).
- Estimates **simple elasticities** vs effective tax rate.
- Runs **switchback/time-based A/B** (blocks) using those elasticities.
- Runs **grid optimization** of a single tax rate with optional constraints.

**What this page does NOT do:**
- Full economy accounting, sinks/sources dashboards, or deep market microstructure dashboards — those belong to Economy+Market Desk.
        """
    )
