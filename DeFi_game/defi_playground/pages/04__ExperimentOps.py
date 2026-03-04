# pages/03_ExperimentOps.py
# Experiment Ops — A/B launch + analysis + guardrails
# Lightweight, practical tooling for synthetic (and later real) experiments.
# All UI labels and comments are in English.

from __future__ import annotations

import os
import math
import hashlib
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
st.set_page_config(page_title="Experiment Ops", layout="wide")
st.title("Experiment Ops — A/B launch + analysis + guardrails")
st.caption("Deterministic user split • Metric extraction • SRM check • Bootstrapped CI • Segment drilldown • CSV exports")


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

def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)

def bootstrap_ci(x: np.ndarray, alpha=0.05, n_boot=2000, seed=42):
    """Bootstrap CI for mean(x)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        boots[i] = np.mean(samp)
    lo = float(np.quantile(boots, alpha/2))
    hi = float(np.quantile(boots, 1-alpha/2))
    return float(np.mean(x)), lo, hi

def bootstrap_diff_ci(xA: np.ndarray, xB: np.ndarray, alpha=0.05, n_boot=2000, seed=42):
    """Bootstrap CI for mean(B)-mean(A)."""
    xA = np.asarray(xA, dtype=float); xB = np.asarray(xB, dtype=float)
    xA = xA[np.isfinite(xA)]; xB = xB[np.isfinite(xB)]
    if len(xA) < 5 or len(xB) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    nA, nB = len(xA), len(xB)
    for i in range(n_boot):
        a = rng.choice(xA, size=nA, replace=True)
        b = rng.choice(xB, size=nB, replace=True)
        boots[i] = np.mean(b) - np.mean(a)
    lo = float(np.quantile(boots, alpha/2))
    hi = float(np.quantile(boots, 1-alpha/2))
    return float(np.mean(xB) - np.mean(xA)), lo, hi

def hash_to_uniform_0_1(x: str) -> float:
    """Stable hash -> [0,1)."""
    h = hashlib.sha256(x.encode("utf-8")).hexdigest()
    # take first 16 hex chars -> 64-bit
    v = int(h[:16], 16)
    return (v % (10**12)) / (10**12)

def assign_variants(users: pd.DataFrame, user_id_col: str, exp_key: str, alloc: dict[str, float]):
    """
    Deterministic assignment by hashing user_id + exp_key.
    alloc = {"A":0.5,"B":0.5,...} should sum to 1.
    """
    keys = users[user_id_col].astype(str).fillna("")
    u = keys.map(lambda uid: hash_to_uniform_0_1(f"{exp_key}::{uid}")).astype(float)

    # cumulative thresholds
    variants = list(alloc.keys())
    probs = np.array([alloc[v] for v in variants], dtype=float)
    probs = probs / probs.sum()
    cum = np.cumsum(probs)

    # vectorized assign
    r = u.to_numpy()
    idx = np.searchsorted(cum, r, side="right")
    idx = np.clip(idx, 0, len(variants)-1)
    return pd.Series([variants[i] for i in idx], index=users.index, name="variant")

def chi_square_srm(counts: dict[str, int], expected: dict[str, float]):
    """
    SRM chi-square test (approx).
    Returns chi2 statistic and p-value (approx via survival function for df=k-1).
    To keep dependencies minimal, we implement a simple approximation for p-value using scipy-like logic is avoided.
    We'll output only chi2 and a rough "flag" threshold.
    """
    vars_ = list(expected.keys())
    obs = np.array([counts.get(v, 0) for v in vars_], dtype=float)
    n = obs.sum()
    exp = np.array([expected[v] * n for v in vars_], dtype=float)
    chi2 = np.sum((obs - exp) ** 2 / np.clip(exp, 1e-9, None))
    df = max(1, len(vars_) - 1)
    # Very rough rule-of-thumb thresholds for chi2:
    # df=1: 3.84 ~ p=0.05 ; df=2: 5.99 ; df=3: 7.81 ; df=4: 9.49
    crit_map = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07}
    crit = crit_map.get(df, 3.84 + (df-1)*2.0)
    flag = chi2 > crit
    return float(chi2), int(df), float(crit), bool(flag)

@st.cache_data(show_spinner=False)
def build_user_metrics(
    users: pd.DataFrame,
    sessions: pd.DataFrame | None,
    market_trades: pd.DataFrame | None,
    token_ledger: pd.DataFrame | None,
    date_from: pd.Timestamp,
    date_to_excl: pd.Timestamp,
):
    """
    Build user-level metrics within the analysis window.
    Uses what we have (sessions, trades, ledger).
    """

    u = users.copy()
    u_id = _pick(u, ["user_id", "id"], df_name="users")
    u_created = _pick(u, ["created_at", "signup_ts", "registered_at"], df_name="users")
    u[u_created] = _to_dt(u[u_created])

    seg_cols = [c for c in ["whale_flag","acq_channel","risk_profile","player_segment","country","device_os"] if c in u.columns]
    u_keep = [u_id, u_created] + seg_cols
    u0 = u[u_keep].copy()

    # --- Sessions metrics ---
    if sessions is not None:
        s = sessions.copy()
        s_user = _pick(s, ["user_id"], df_name="sessions")
        s_start = _pick(s, ["start_ts", "ts"], df_name="sessions")
        s_end = _pick(s, ["end_ts"], required=False, df_name="sessions")
        s_len = _pick(s, ["session_len_sec"], required=False, df_name="sessions")

        s[s_start] = _to_dt(s[s_start])
        s["day"] = s[s_start].dt.floor("D")
        s = s[(s[s_start] >= date_from) & (s[s_start] < date_to_excl)].copy()

        # basic session stats
        if s_len and s_len in s.columns:
            s[s_len] = pd.to_numeric(s[s_len], errors="coerce").fillna(0.0)
            sess_agg = s.groupby(s_user).agg(
                sessions_cnt=("session_id", "count"),
                active_days=("day", "nunique"),
                session_time_sec=(s_len, "sum"),
            ).reset_index()
        else:
            sess_agg = s.groupby(s_user).agg(
                sessions_cnt=("session_id", "count"),
                active_days=("day", "nunique"),
            ).reset_index()
            sess_agg["session_time_sec"] = np.nan

        # guardrails: crash rate, latency
        if "crash_flag" in s.columns:
            s["crash_flag"] = pd.to_numeric(s["crash_flag"], errors="coerce").fillna(0.0)
            crash = s.groupby(s_user)["crash_flag"].mean().reset_index(name="crash_rate")
        else:
            crash = None

        if "net_latency_ms" in s.columns:
            s["net_latency_ms"] = pd.to_numeric(s["net_latency_ms"], errors="coerce")
            lat = s.groupby(s_user)["net_latency_ms"].quantile(0.95).reset_index(name="latency_p95_ms")
        else:
            lat = None

        # retention proxy: had activity on day+7 from first active day in window
        first_day = s.groupby(s_user)["day"].min().reset_index(name="first_active_day")
        active_ud = s.groupby([s_user, "day"]).size().reset_index(name="active_flag")
        active_ud["active_flag"] = 1

        tmp = first_day.copy()
        tmp["target_day"] = tmp["first_active_day"] + pd.Timedelta(days=7)
        tmp = tmp.merge(active_ud[[s_user, "day", "active_flag"]],
                        left_on=[s_user, "target_day"],
                        right_on=[s_user, "day"],
                        how="left")
        tmp["ret_d7"] = tmp["active_flag"].fillna(0).astype(int)
        ret = tmp[[s_user, "ret_d7"]].copy()

    else:
        sess_agg = pd.DataFrame(columns=["user_id","sessions_cnt","active_days","session_time_sec"])
        crash = None
        lat = None
        ret = pd.DataFrame(columns=["user_id","ret_d7"])

    # --- Trades metrics ---
    if market_trades is not None:
        mt = market_trades.copy()
        mt_ts = _pick(mt, ["ts", "timestamp"], df_name="market_trades")
        mt["ts"] = _to_dt(mt[mt_ts])
        mt = mt[(mt["ts"] >= date_from) & (mt["ts"] < date_to_excl)].copy()

        maker = _pick(mt, ["maker_user_id", "user_id"], required=False, df_name="market_trades")
        if maker is None:
            mt["maker_user_id"] = np.nan
            maker = "maker_user_id"

        amt_base = _pick(mt, ["amount_base"], df_name="market_trades")
        price = _pick(mt, ["price"], df_name="market_trades")
        mt[amt_base] = pd.to_numeric(mt[amt_base], errors="coerce").fillna(0.0)
        mt[price] = pd.to_numeric(mt[price], errors="coerce").fillna(0.0)
        mt["gross_token"] = mt[amt_base] * mt[price]

        # revenue proxy: taxes+fees on trade
        tax_col = _pick(mt, ["tax_amount"], required=False, df_name="market_trades")
        fee_col = _pick(mt, ["fee_amount"], required=False, df_name="market_trades")
        if tax_col and tax_col in mt.columns:
            mt[tax_col] = pd.to_numeric(mt[tax_col], errors="coerce").fillna(0.0)
        else:
            mt["tax_amount"] = 0.0
            tax_col = "tax_amount"
        if fee_col and fee_col in mt.columns:
            mt[fee_col] = pd.to_numeric(mt[fee_col], errors="coerce").fillna(0.0)
        else:
            mt["fee_amount"] = 0.0
            fee_col = "fee_amount"

        # microstructure guardrails
        slip = _pick(mt, ["slippage_bps"], required=False, df_name="market_trades")
        spr = _pick(mt, ["spread_bps"], required=False, df_name="market_trades")
        if slip and slip in mt.columns:
            mt[slip] = pd.to_numeric(mt[slip], errors="coerce")
        if spr and spr in mt.columns:
            mt[spr] = pd.to_numeric(mt[spr], errors="coerce")

        trade_agg = mt.groupby(maker).agg(
            made_trade=("trade_id", "count"),
            trade_volume_token=("gross_token","sum"),
            tax_paid_token=(tax_col,"sum"),
            fee_paid_token=(fee_col,"sum"),
        ).reset_index().rename(columns={maker: "user_id"})

        trade_agg["made_trade"] = (trade_agg["made_trade"] > 0).astype(int)
        trade_agg["revenue_token"] = trade_agg["tax_paid_token"] + trade_agg["fee_paid_token"]

        if slip and slip in mt.columns:
            slip_agg = mt.groupby(maker)[slip].mean().reset_index(name="slippage_bps_avg").rename(columns={maker:"user_id"})
            trade_agg = trade_agg.merge(slip_agg, on="user_id", how="left")
        else:
            trade_agg["slippage_bps_avg"] = np.nan

        if spr and spr in mt.columns:
            spr_agg = mt.groupby(maker)[spr].mean().reset_index(name="spread_bps_avg").rename(columns={maker:"user_id"})
            trade_agg = trade_agg.merge(spr_agg, on="user_id", how="left")
        else:
            trade_agg["spread_bps_avg"] = np.nan

    else:
        trade_agg = pd.DataFrame(columns=[
            "user_id","made_trade","trade_volume_token","tax_paid_token","fee_paid_token",
            "revenue_token","slippage_bps_avg","spread_bps_avg"
        ])

    # --- Ledger optional metrics (if you later add spend, staking, etc.) ---
    # For now we keep it minimal: trade-based revenue is already enough for experiments.
    # You can extend here with tx_type-level outcomes.

    # Merge all
    out = u0.rename(columns={u_id:"user_id"}).copy()
    out = out.merge(sess_agg.rename(columns={_pick(sess_agg, ["user_id"], required=False) or "user_id":"user_id"}),
                    on="user_id", how="left")
    if crash is not None:
        out = out.merge(crash.rename(columns={_pick(crash, ["user_id"], required=False) or "user_id":"user_id"}),
                        on="user_id", how="left")
    else:
        out["crash_rate"] = np.nan

    if lat is not None:
        out = out.merge(lat.rename(columns={_pick(lat, ["user_id"], required=False) or "user_id":"user_id"}),
                        on="user_id", how="left")
    else:
        out["latency_p95_ms"] = np.nan

    if not ret.empty:
        out = out.merge(ret.rename(columns={_pick(ret, ["user_id"], required=False) or "user_id":"user_id"}),
                        on="user_id", how="left")
    else:
        out["ret_d7"] = np.nan

    out = out.merge(trade_agg, on="user_id", how="left")

    # Fill numeric NaNs with 0 where appropriate
    for c in ["sessions_cnt","active_days","session_time_sec","made_trade","trade_volume_token","tax_paid_token","fee_paid_token","revenue_token"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # guardrails numeric
    for c in ["crash_rate","latency_p95_ms","slippage_bps_avg","spread_bps_avg","ret_d7"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # convenience
    out["revenue_per_user"] = out["revenue_token"]
    out["volume_per_user"] = out["trade_volume_token"]
    out["is_active"] = (out["sessions_cnt"] > 0).astype(int)

    return out


def summarize_ab(user_metrics: pd.DataFrame, variant_col: str, metrics: list[tuple[str, str]]):
    """
    metrics: list of (metric_name, metric_type) where type in {"mean","rate"}
    Returns summary dataframe with lift and bootstrap CI.
    """
    df = user_metrics.copy()
    vars_ = df[variant_col].dropna().unique().tolist()
    if len(vars_) < 2:
        return None

    # enforce A as baseline if exists
    if "A" in vars_:
        base = "A"
    else:
        base = vars_[0]
    others = [v for v in vars_ if v != base]

    rows = []
    for m, mtype in metrics:
        if m not in df.columns:
            continue
        A = df.loc[df[variant_col] == base, m].to_numpy(dtype=float)

        # baseline mean/rate
        meanA = float(np.nanmean(A)) if len(A) else np.nan

        for v in others:
            B = df.loc[df[variant_col] == v, m].to_numpy(dtype=float)
            meanB = float(np.nanmean(B)) if len(B) else np.nan

            diff, lo, hi = bootstrap_diff_ci(A, B, n_boot=2000, seed=42)
            lift = diff
            lift_pct = float(lift / meanA) if (meanA not in (0, np.nan) and np.isfinite(meanA) and meanA != 0) else np.nan

            rows.append({
                "metric": m,
                "type": mtype,
                "baseline": base,
                "variant": v,
                "mean_baseline": meanA,
                "mean_variant": meanB,
                "lift_abs": lift,
                "lift_pct": lift_pct,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_baseline": int(np.isfinite(A).sum()),
                "n_variant": int(np.isfinite(B).sum()),
            })

    return pd.DataFrame(rows)


# =========================
# Sidebar: paths & load
# =========================
st.sidebar.header("Data paths")

default_extra = r"C:\Users\Shaim\defi game\synthetic_data"
default_main  = r"C:\Users\Shaim\defi game\synthetic_defi_game_data"

ROOT_EXTRA = st.sidebar.text_input("Extra synthetic data folder", value=default_extra)
ROOT_MAIN  = st.sidebar.text_input("Main synthetic data folder", value=default_main)
ROOTS = (ROOT_MAIN, ROOT_EXTRA)

users         = load_csv_any(ROOTS, "users.csv")
sessions      = load_csv_any(ROOTS, "sessions.csv")
market_trades = load_csv_any(ROOTS, "market_trades.csv")
token_ledger  = load_csv_any(ROOTS, "token_ledger.csv")

if users is None:
    st.error("Missing users.csv. Please place it in one of the configured folders.")
    st.stop()

with st.sidebar.expander("Loaded tables (shapes)", expanded=True):
    st.write({
        "users": None if users is None else users.shape,
        "sessions": None if sessions is None else sessions.shape,
        "market_trades": None if market_trades is None else market_trades.shape,
        "token_ledger": None if token_ledger is None else token_ledger.shape,
    })


# =========================
# Experiment config
# =========================
st.sidebar.header("Experiment setup")

exp_key = st.sidebar.text_input("Experiment key", value="exp_tax_onboarding_v1")
unit = st.sidebar.selectbox("Randomization unit", options=["user_id"], index=0)

# Variants and allocation
variant_mode = st.sidebar.selectbox("Variants", options=["A/B (50/50)", "A/B (70/30)", "A/B/C (34/33/33)", "Custom"], index=0)
if variant_mode == "A/B (50/50)":
    alloc = {"A": 0.5, "B": 0.5}
elif variant_mode == "A/B (70/30)":
    alloc = {"A": 0.7, "B": 0.3}
elif variant_mode == "A/B/C (34/33/33)":
    alloc = {"A": 0.34, "B": 0.33, "C": 0.33}
else:
    st.sidebar.caption("Custom allocation must sum to 1.0")
    aA = st.sidebar.number_input("A share", 0.0, 1.0, 0.5, 0.01)
    aB = st.sidebar.number_input("B share", 0.0, 1.0, 0.5, 0.01)
    aC = st.sidebar.number_input("C share (optional)", 0.0, 1.0, 0.0, 0.01)
    alloc = {"A": float(aA), "B": float(aB)}
    if aC > 0:
        alloc["C"] = float(aC)

# Analysis window
# Use trades range as a default if available, else users created_at range
u_created = _pick(users, ["created_at","signup_ts","registered_at"], df_name="users")
users_dt = _to_dt(users[u_created])
dmin = users_dt.min()
dmax = users_dt.max()
if market_trades is not None:
    mt_ts = _pick(market_trades, ["ts","timestamp"], df_name="market_trades")
    mt_dt = _to_dt(market_trades[mt_ts])
    if mt_dt.notna().any():
        dmin = min(dmin, mt_dt.min())
        dmax = max(dmax, mt_dt.max())

if pd.isna(dmin) or pd.isna(dmax):
    dmin = pd.Timestamp("2025-01-01")
    dmax = pd.Timestamp("2025-12-31")

date_from, date_to = st.sidebar.date_input(
    "Analysis date range",
    value=(dmin.date(), dmax.date()),
    min_value=dmin.date(),
    max_value=dmax.date(),
)
date_from = pd.Timestamp(date_from)
date_to_excl = pd.Timestamp(date_to) + pd.Timedelta(days=1)

# Who is eligible?
eligibility = st.sidebar.selectbox("Eligibility population", options=[
    "All users",
    "Users created before experiment start",
    "Users created within window (new users cohort)"
], index=0)

segment_cols = [c for c in ["whale_flag","acq_channel","risk_profile","player_segment","country","device_os"] if c in users.columns]
segment_pick = st.sidebar.selectbox("Segment breakdown (optional)", options=["None"] + segment_cols, index=0)

st.sidebar.divider()
st.sidebar.header("Guardrails")
guard_crash = st.sidebar.checkbox("Crash rate", value=True)
guard_latency = st.sidebar.checkbox("Latency p95", value=True)
guard_micro = st.sidebar.checkbox("Market quality (slippage/spread)", value=True)

st.sidebar.divider()
export_dir = st.sidebar.text_input("Export folder (optional)", value=str(Path(ROOT_MAIN)))
run_btn = st.sidebar.button("Run experiment analysis", type="primary")


# =========================
# Run
# =========================
if not run_btn:
    st.info("Configure the experiment in the sidebar and click **Run experiment analysis**.")
    st.markdown(
        """
**This page supports:**
- Deterministic user assignment (stable hash, reproducible)
- A/B summary (lift + bootstrap CI)
- SRM check (sample ratio mismatch)
- Guardrails (crash/latency/microstructure)
- Segment drilldown
- CSV exports (assignments + per-user metrics + summary)

**Tip:** For real experiments later, you can replace deterministic assignment with an assignment log (events table).
        """
    )
    st.stop()

status = st.status("Running Experiment Ops…", expanded=True)
p = st.progress(0)

status.write("Preparing eligibility population…")
u = users.copy()
u_id = _pick(u, ["user_id","id"], df_name="users")
u_created = _pick(u, ["created_at","signup_ts","registered_at"], df_name="users")
u[u_created] = _to_dt(u[u_created])

if eligibility == "All users":
    u_elig = u.copy()
elif eligibility == "Users created before experiment start":
    u_elig = u[u[u_created] < date_from].copy()
else:
    u_elig = u[(u[u_created] >= date_from) & (u[u_created] < date_to_excl)].copy()

if u_elig.empty:
    st.warning("Eligibility set is empty. Adjust the date range or eligibility rule.")
    status.update(state="complete")
    st.stop()

p.progress(10)

status.write("Assigning users to variants (deterministic hash)…")
# normalize alloc
total_alloc = sum(alloc.values())
if total_alloc <= 0:
    st.error("Allocation shares sum to 0. Fix allocation in sidebar.")
    status.update(state="complete")
    st.stop()
alloc = {k: float(v)/total_alloc for k, v in alloc.items()}

u_elig = u_elig.copy()
u_elig["variant"] = assign_variants(u_elig, user_id_col=u_id, exp_key=exp_key, alloc=alloc)

# SRM check
counts = u_elig["variant"].value_counts().to_dict()
chi2, df, crit, srm_flag = chi_square_srm(counts, alloc)

p.progress(20)

status.write("Building user-level metrics within analysis window…")
user_metrics = build_user_metrics(
    users=u_elig,  # only eligible users
    sessions=sessions,
    market_trades=market_trades,
    token_ledger=token_ledger,
    date_from=date_from,
    date_to_excl=date_to_excl,
)

# attach variant
user_metrics = user_metrics.merge(u_elig[[u_id, "variant"]].rename(columns={u_id:"user_id"}), on="user_id", how="left")
p.progress(35)

# Metric set
# Primary metrics (typical): conversion to trade, revenue per user, volume per user, retention d7
metrics = [
    ("is_active", "rate"),
    ("made_trade", "rate"),
    ("ret_d7", "rate"),
    ("revenue_per_user", "mean"),
    ("volume_per_user", "mean"),
    ("sessions_cnt", "mean"),
    ("active_days", "mean"),
]

# Guardrails
guardrails = []
if guard_crash and "crash_rate" in user_metrics.columns:
    guardrails.append(("crash_rate", "mean"))
if guard_latency and "latency_p95_ms" in user_metrics.columns:
    guardrails.append(("latency_p95_ms", "mean"))
if guard_micro:
    if "slippage_bps_avg" in user_metrics.columns:
        guardrails.append(("slippage_bps_avg", "mean"))
    if "spread_bps_avg" in user_metrics.columns:
        guardrails.append(("spread_bps_avg", "mean"))

status.write("Computing A/B summary with bootstrap CIs…")
summary = summarize_ab(user_metrics, variant_col="variant", metrics=metrics + guardrails)
p.progress(55)

status.update(label="Done.", state="complete", expanded=False)
p.progress(100)


# =========================
# UI outputs
# =========================
st.subheader("1) Launch / Assignment")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Eligible users", f"{len(u_elig):,}")
c2.metric("Variants", ", ".join(sorted(alloc.keys())))
c3.metric("SRM chi²", f"{chi2:.2f} (df={df})")
c4.metric("SRM flag", "YES" if srm_flag else "NO")

st.caption(f"SRM rule-of-thumb: chi² > {crit:.2f} (≈ p<0.05). Allocation: {alloc}")

with st.expander("Show assignment counts", expanded=False):
    st.write(pd.DataFrame({"variant": list(counts.keys()), "users": list(counts.values())}).sort_values("variant"))

# Optional export assignments
with st.expander("Export assignments (CSV)", expanded=False):
    if export_dir and Path(export_dir).exists():
        out_path = Path(export_dir) / f"assignments_{exp_key}.csv"
        if st.button("Save assignments CSV"):
            u_elig[[u_id, "variant"]].to_csv(out_path, index=False)
            st.success(f"Saved: {out_path}")
    else:
        st.info("Provide a valid export folder in the sidebar to enable exports.")

st.divider()


st.subheader("2) A/B results (primary + guardrails)")

if summary is None or summary.empty:
    st.warning("Not enough variants or metrics to summarize.")
else:
    # Friendly formatting
    view = summary.copy()
    view["lift_pct"] = (view["lift_pct"] * 100.0).round(2)
    view["mean_baseline"] = view["mean_baseline"].round(6)
    view["mean_variant"] = view["mean_variant"].round(6)
    view["lift_abs"] = view["lift_abs"].round(6)
    view["ci95_low"] = view["ci95_low"].round(6)
    view["ci95_high"] = view["ci95_high"].round(6)

    st.dataframe(view, use_container_width=True)

    # Quick “decision-ish” view
    st.caption("Interpretation tip: guardrails should not degrade even if primary metric improves.")

st.divider()


st.subheader("3) Visuals")

tab_ts, tab_dist, tab_seg = st.tabs(["Time series", "Distributions", "Segments"])

# --- Time series ---
with tab_ts:
    st.markdown("Daily time series by variant (based on trades/sessions within window).")

    # Build daily variant panel for a few key metrics
    if sessions is not None:
        s = sessions.copy()
        s_user = _pick(s, ["user_id"], df_name="sessions")
        s_ts = _pick(s, ["start_ts","ts"], df_name="sessions")
        s[s_ts] = _to_dt(s[s_ts])
        s = s[(s[s_ts] >= date_from) & (s[s_ts] < date_to_excl)].copy()
        s["day"] = s[s_ts].dt.floor("D")
        s = s.merge(u_elig[[u_id,"variant"]], left_on=s_user, right_on=u_id, how="inner")
        dau = s.groupby(["day","variant"])[s_user].nunique().reset_index(name="DAU_variant")
    else:
        dau = None

    if market_trades is not None:
        mt = market_trades.copy()
        mt_ts = _pick(mt, ["ts","timestamp"], df_name="market_trades")
        mt["ts"] = _to_dt(mt[mt_ts])
        mt = mt[(mt["ts"] >= date_from) & (mt["ts"] < date_to_excl)].copy()
        maker = _pick(mt, ["maker_user_id","user_id"], required=False, df_name="market_trades") or "maker_user_id"
        mt = mt.merge(u_elig[[u_id,"variant"]], left_on=maker, right_on=u_id, how="inner")
        mt["day"] = mt["ts"].dt.floor("D")
        mt["amount_base"] = pd.to_numeric(mt["amount_base"], errors="coerce").fillna(0.0)
        mt["price"] = pd.to_numeric(mt["price"], errors="coerce").fillna(0.0)
        mt["gross_token"] = mt["amount_base"] * mt["price"]
        daily_vol = mt.groupby(["day","variant"])["gross_token"].sum().reset_index(name="volume_token")
        daily_traders = mt.groupby(["day","variant"])[maker].nunique().reset_index(name="traders")
        daily_trade_cnt = mt.groupby(["day","variant"])["trade_id"].count().reset_index(name="trades")
    else:
        daily_vol = daily_traders = daily_trade_cnt = None

    # Merge daily panel
    daily = None
    for d in [dau, daily_vol, daily_traders, daily_trade_cnt]:
        if d is None:
            continue
        daily = d if daily is None else daily.merge(d, on=["day","variant"], how="outer")
    if daily is None or daily.empty:
        st.info("Not enough event data to build daily plots.")
    else:
        daily = daily.sort_values(["day","variant"])
        if PLOTLY_OK:
            mopt = st.selectbox("Daily metric", options=[c for c in daily.columns if c not in ("day","variant")], index=0)
            fig = px.line(daily, x="day", y=mopt, color="variant", markers=False, title=f"{mopt} by day")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            mopt = st.selectbox("Daily metric", options=[c for c in daily.columns if c not in ("day","variant")], index=0)
            pivot = daily.pivot(index="day", columns="variant", values=mopt)
            st.line_chart(pivot)

# --- Distributions ---
with tab_dist:
    st.markdown("User-level distributions (helps sanity-check heavy tails, whales, etc.).")
    metric_pick = st.selectbox(
        "Metric",
        options=[m[0] for m in metrics + guardrails if m[0] in user_metrics.columns],
        index=0
    )

    dfp = user_metrics[["variant", metric_pick]].copy()
    dfp[metric_pick] = pd.to_numeric(dfp[metric_pick], errors="coerce")

    if PLOTLY_OK:
        fig = px.histogram(dfp, x=metric_pick, color="variant", barmode="overlay", nbins=60, title=f"Distribution: {metric_pick}")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(dfp, x="variant", y=metric_pick, points="outliers", title=f"Box: {metric_pick} by variant")
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write(dfp.groupby("variant")[metric_pick].describe())

# --- Segments ---
with tab_seg:
    st.markdown("Segment drilldown (guardrail + primary metrics).")
    if segment_pick == "None":
        st.info("Select a segment column in the sidebar to enable segment breakdown.")
    else:
        if segment_pick not in user_metrics.columns:
            st.warning(f"Segment column {segment_pick} not found in users.")
        else:
            seg = user_metrics.copy()
            seg[segment_pick] = seg[segment_pick].astype(str)

            pick_metric = st.selectbox(
                "Metric for segment view",
                options=["made_trade","revenue_per_user","volume_per_user","ret_d7"] + ([ "crash_rate","latency_p95_ms" ] if guard_crash or guard_latency else []),
                index=0
            )
            if pick_metric not in seg.columns:
                st.warning("Metric not available in user_metrics.")
            else:
                g = seg.groupby(["variant", segment_pick]).agg(
                    users=("user_id","nunique"),
                    value=(pick_metric, "mean"),
                ).reset_index().sort_values(["variant","users"], ascending=[True, False])

                st.dataframe(g, use_container_width=True)

                if PLOTLY_OK:
                    fig = px.bar(g, x=segment_pick, y="value", color="variant", barmode="group",
                                 title=f"{pick_metric} by {segment_pick}")
                    fig.update_layout(height=420)
                    st.plotly_chart(fig, use_container_width=True)

st.divider()


st.subheader("4) Exports")
colA, colB, colC = st.columns(3)
with colA:
    st.caption("Per-user metrics (eligibility cohort)")
    if export_dir and Path(export_dir).exists():
        if st.button("Save user_metrics CSV"):
            out_path = Path(export_dir) / f"user_metrics_{exp_key}.csv"
            user_metrics.to_csv(out_path, index=False)
            st.success(f"Saved: {out_path}")
    else:
        st.info("Set a valid export folder in sidebar.")

with colB:
    st.caption("Summary table")
    if export_dir and Path(export_dir).exists():
        if st.button("Save summary CSV"):
            out_path = Path(export_dir) / f"ab_summary_{exp_key}.csv"
            if summary is not None:
                summary.to_csv(out_path, index=False)
                st.success(f"Saved: {out_path}")
            else:
                st.warning("No summary to export.")
    else:
        st.info("Set a valid export folder in sidebar.")

with colC:
    st.caption("Experiment config snapshot")
    if export_dir and Path(export_dir).exists():
        if st.button("Save config CSV"):
            out_path = Path(export_dir) / f"exp_config_{exp_key}.csv"
            cfg = pd.DataFrame([{
                "exp_key": exp_key,
                "date_from": str(date_from),
                "date_to_excl": str(date_to_excl),
                "eligibility": eligibility,
                "alloc": str(alloc),
                "srm_chi2": chi2,
                "srm_df": df,
                "srm_crit": crit,
                "srm_flag": srm_flag,
            }])
            cfg.to_csv(out_path, index=False)
            st.success(f"Saved: {out_path}")
    else:
        st.info("Set a valid export folder in sidebar.")
