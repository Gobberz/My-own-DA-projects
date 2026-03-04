import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _dt(s):
    return pd.to_datetime(s, errors="coerce")


def _pick(df, cols, required=True, name="df"):
    for c in cols:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"{name}:didn't find {cols}. We have: {list(df.columns)}")
    return None


def _safe_mean(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else np.nan


def build_user_day_panel(
    users: pd.DataFrame,
    sessions: pd.DataFrame,
    market_trades: pd.DataFrame,
    price_oracle_daily: pd.DataFrame = None,
    max_days: int = 365
) -> dict:
    """
    Build user-day panel:
    - volume_token, trades_cnt, tax, fee, avg_spread, avg_slippage
    - session_time, sessions_cnt
    - tax_rate_eff, fee_rate_eff
    - vol_7d (rolling), bucket_id (по quantiles)
    - day_volatility (если есть price_oracle_daily)
    + join user сегментов (whale_flag, risk_profile, channel)

    Back dict:
      panel (df), bucket_edges, bucket_labels, date_range (min/max day)
    """

    u = users.copy()
    s = sessions.copy()
    mt = market_trades.copy()

    uid_u = _pick(u, ["user_id", "id"], name="users")
    created = _pick(u, ["created_at", "signup_ts", "registered_at"], name="users")

    uid_s = _pick(s, ["user_id", "id_user"], name="sessions")
    s_start = _pick(s, ["start_ts", "ts", "session_start_ts"], name="sessions")
    s_len = _pick(s, ["session_len_sec", "duration_sec", "len_sec"], required=False, name="sessions")

    uid_mt = _pick(mt, ["maker_user_id", "user_id"], name="market_trades")
    ts_mt = _pick(mt, ["ts", "timestamp"], name="market_trades")
    amt_base = _pick(mt, ["amount_base", "qty_base", "base_amount"], name="market_trades")
    price = _pick(mt, ["price", "px"], name="market_trades")

    # to datetime
    u[created] = _dt(u[created])
    s[s_start] = _dt(s[s_start])
    mt[ts_mt] = _dt(mt[ts_mt])

    # date range
    min_day = pd.concat([
        u[created].dt.floor("D"),
        s[s_start].dt.floor("D"),
        mt[ts_mt].dt.floor("D"),
    ]).min()

    max_day = pd.concat([
        s[s_start].dt.floor("D"),
        mt[ts_mt].dt.floor("D"),
    ]).max()

    if pd.isna(min_day) or pd.isna(max_day):
        raise ValueError("Не удалось определить диапазон дат. Проверь timestamps.")

    # limit to max_days (safety)
    day_index = pd.date_range(min_day, max_day, freq="D")
    if len(day_index) > max_days:
        day_index = day_index[-max_days:]

    # base grid users x days
    user_ids = u[uid_u].dropna().unique()
    idx = pd.MultiIndex.from_product([user_ids, day_index], names=["user_id", "day"])
    panel = pd.DataFrame(index=idx).reset_index()

    # --- trades agg (user-day) ---
    mt["day"] = mt[ts_mt].dt.floor("D")
    mt["gross_token"] = pd.to_numeric(mt[amt_base], errors="coerce").fillna(0.0) * pd.to_numeric(mt[price], errors="coerce").fillna(0.0)

    for c in ["tax_amount", "fee_amount", "slippage_bps", "spread_bps"]:
        if c in mt.columns:
            mt[c] = pd.to_numeric(mt[c], errors="coerce")

    agg = {
        "gross_token": "sum",
    }
    if "trade_id" in mt.columns:
        agg["trade_id"] = "count"
    if "tax_amount" in mt.columns:
        agg["tax_amount"] = "sum"
    if "fee_amount" in mt.columns:
        agg["fee_amount"] = "sum"
    if "slippage_bps" in mt.columns:
        agg["slippage_bps"] = "mean"
    if "spread_bps" in mt.columns:
        agg["spread_bps"] = "mean"

    t = mt.groupby([uid_mt, "day"]).agg(agg).reset_index().rename(columns={uid_mt: "user_id"})
    t = t.rename(columns={
        "gross_token": "volume_token",
        "trade_id": "trades_cnt",
        "tax_amount": "tax_token",
        "fee_amount": "fee_token",
        "slippage_bps": "avg_slippage_bps",
        "spread_bps": "avg_spread_bps",
    })

    panel = panel.merge(t, on=["user_id", "day"], how="left")

    for c in ["volume_token", "trades_cnt", "tax_token", "fee_token"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)

    for c in ["avg_slippage_bps", "avg_spread_bps"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    # effective rates (handle sign)
    panel["tax_abs"] = panel["tax_token"].abs()
    panel["fee_abs"] = panel["fee_token"].abs()
    panel["tax_rate_eff"] = panel["tax_abs"] / panel["volume_token"].replace(0, np.nan)
    panel["fee_rate_eff"] = panel["fee_abs"] / panel["volume_token"].replace(0, np.nan)
    panel["tax_rate_eff"] = panel["tax_rate_eff"].fillna(0.0)
    panel["fee_rate_eff"] = panel["fee_rate_eff"].fillna(0.0)

    # --- sessions agg (user-day) ---
    s["day"] = s[s_start].dt.floor("D")
    if s_len and s_len in s.columns:
        s[s_len] = pd.to_numeric(s[s_len], errors="coerce").fillna(0.0)
        sa = s.groupby([uid_s, "day"]).agg(
            session_time=("session_len_sec" if s_len == "session_len_sec" else s_len, "sum"),
            sessions_cnt=("session_id" if "session_id" in s.columns else s_len, "count"),
        ).reset_index().rename(columns={uid_s: "user_id"})
    else:
        # proxy
        sa = s.groupby([uid_s, "day"]).size().reset_index(name="sessions_cnt").rename(columns={uid_s: "user_id"})
        sa["session_time"] = sa["sessions_cnt"].astype(float)

    panel = panel.merge(sa, on=["user_id", "day"], how="left")
    panel["sessions_cnt"] = pd.to_numeric(panel["sessions_cnt"], errors="coerce").fillna(0.0)
    panel["session_time"] = pd.to_numeric(panel["session_time"], errors="coerce").fillna(0.0)

    # --- join user segments ---
    u2 = u.rename(columns={uid_u: "user_id"}).copy()
    keep_cols = ["user_id"]
    for c in ["whale_flag", "risk_profile", "player_segment", "acq_channel", "country", "device_os"]:
        if c in u2.columns:
            keep_cols.append(c)
    panel = panel.merge(u2[keep_cols], on="user_id", how="left")

    if "whale_flag" in panel.columns:
        panel["whale_flag"] = pd.to_numeric(panel["whale_flag"], errors="coerce").fillna(0).astype(int)
    else:
        panel["whale_flag"] = 0

    # --- daily volatility (optional) ---
    panel["day_volatility"] = 0.0
    if price_oracle_daily is not None and len(price_oracle_daily):
        po = price_oracle_daily.copy()
        day_col = _pick(po, ["day", "date"], name="price_oracle_daily")
        po[day_col] = _dt(po[day_col]).dt.floor("D")
        if "volatility" in po.columns:
            dv = po.groupby(day_col)["volatility"].mean().reset_index().rename(columns={day_col: "day", "volatility": "day_volatility"})
            panel = panel.merge(dv, on="day", how="left")
            panel["day_volatility"] = pd.to_numeric(panel["day_volatility"], errors="coerce").fillna(0.0)

    # --- rolling 7d volume & buckets ---
    panel = panel.sort_values(["user_id", "day"])
    panel["vol_7d"] = (
        panel.groupby("user_id")["volume_token"]
             .rolling(7, min_periods=1)
             .sum()
             .reset_index(level=0, drop=True)
    )

    # bucket edges: qcut on vol_7d positive
    x = panel.loc[panel["vol_7d"] > 0, "vol_7d"]
    if x.nunique() >= 10:
        qs = [0, 0.5, 0.8, 0.95, 0.99, 1.0]
        edges = sorted(set([float(x.quantile(q)) for q in qs]))
        # ensure valid edges
        edges = [0.0] + [e for e in edges if e > 0]
    else:
        # fallback
        edges = [0.0, 10.0, 100.0, 1000.0, 10000.0, np.inf]

    # pd.cut needs monotonic increasing
    edges = np.array(edges, dtype=float)
    edges = np.unique(edges)
    if edges[-1] != np.inf:
        edges = np.append(edges, np.inf)

    labels = [f"B{i}" for i in range(len(edges) - 1)]
    panel["bucket_id"] = pd.cut(panel["vol_7d"], bins=edges, labels=labels, include_lowest=True).astype(str)

    # fill NAs bucket
    panel.loc[panel["bucket_id"] == "nan", "bucket_id"] = labels[0]

    return {
        "panel": panel,
        "bucket_edges": edges.tolist(),
        "bucket_labels": labels,
        "min_day": min_day,
        "max_day": max_day,
    }


def fit_extensive_intensive_models(panel: pd.DataFrame) -> dict:
    """
    Fit:
      - Extensive: P(trade>0)
      - Intensive: log1p(volume) | trade>0

    Back dict with models and feature columns.
    """
    df = panel.copy()

    # target
    df["y_trade"] = (df["volume_token"] > 0).astype(int)
    df["y_logvol"] = np.log1p(df["volume_token"].clip(lower=0))

    # features (numeric)
    feats_num = [
        "tax_rate_eff",
        "fee_rate_eff",
        "day_volatility",
        "avg_spread_bps",
        "avg_slippage_bps",
        "session_time",
        "sessions_cnt",
        "vol_7d",
        "whale_flag",
    ]
    for c in feats_num:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # simple categoricals -> dummies
    cats = [c for c in ["risk_profile", "acq_channel", "player_segment"] if c in df.columns]
    if cats:
        dummies = pd.get_dummies(df[cats].fillna("unknown"), prefix=cats, drop_first=True)
        X = pd.concat([df[feats_num], dummies], axis=1)
    else:
        X = df[feats_num].copy()

    feature_cols = list(X.columns)

    # Extensive model
    y = df["y_trade"].to_numpy()
    unique = np.unique(y)

    ext_model = None
    ext_baseline_p = float(np.mean(y)) if len(y) else 0.0
    if len(unique) >= 2:
        ext_model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=300, n_jobs=None))
        ])
        ext_model.fit(X, y)

    # Intensive (only trade>0)
    mask = df["y_trade"] == 1
    int_model = None
    if mask.sum() >= 50:
        int_model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", Ridge(alpha=1.0))
        ])
        int_model.fit(X.loc[mask, :], df.loc[mask, "y_logvol"].to_numpy())

    return {
        "X_cols": feature_cols,
        "ext_model": ext_model,
        "ext_baseline_p": ext_baseline_p,
        "int_model": int_model,
        "feats_num": feats_num,
        "cats": cats,
    }


def _build_X_from_panel(panel: pd.DataFrame, X_cols: list) -> pd.DataFrame:
    df = panel.copy()

    # make sure needed cols exist
    for c in ["tax_rate_eff", "fee_rate_eff", "day_volatility", "avg_spread_bps", "avg_slippage_bps",
              "session_time", "sessions_cnt", "vol_7d", "whale_flag"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    cats = [c for c in ["risk_profile", "acq_channel", "player_segment"] if c in df.columns]
    if cats:
        dummies = pd.get_dummies(df[cats].fillna("unknown"), prefix=cats, drop_first=True)
        X = pd.concat([df[["tax_rate_eff","fee_rate_eff","day_volatility","avg_spread_bps","avg_slippage_bps",
                           "session_time","sessions_cnt","vol_7d","whale_flag"]], dummies], axis=1)
    else:
        X = df[["tax_rate_eff","fee_rate_eff","day_volatility","avg_spread_bps","avg_slippage_bps",
                "session_time","sessions_cnt","vol_7d","whale_flag"]].copy()

    # align columns
    for c in X_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[X_cols]
    return X


def simulate_tax_policy(
    panel: pd.DataFrame,
    model_pack: dict,
    base_tax_rate: float,
    bucket_multipliers: dict,
    fee_rate_fixed: float = None,
) -> dict:
    """
    Sim:
      new_tax_rate = base_tax_rate * multiplier(bucket_id)
      fee_rate = fee_rate_fixed (если None -> avg fee_rate_eff by trade days)

    Back:
      - user_day_pred (ожидаемые объёмы/выручка)
      - daily_overall, daily_whales, daily_nonwhales
      - baseline_daily (факт из данных)
    """
    df = panel.copy()

    # baseline facts
    baseline_daily = df.groupby("day").agg(
        volume=("volume_token", "sum"),
        traders=("user_id", lambda s: int((df.loc[s.index, "volume_token"] > 0).sum())),
        tax=("tax_abs", "sum"),
        fee=("fee_abs", "sum"),
    ).reset_index()
    baseline_daily["revenue"] = baseline_daily["tax"] + baseline_daily["fee"]

    # fee rate default
    if fee_rate_fixed is None:
        traded = df["volume_token"] > 0
        fee_rate_fixed = float(df.loc[traded, "fee_rate_eff"].replace([np.inf, -np.inf], np.nan).dropna().mean())
        if not np.isfinite(fee_rate_fixed):
            fee_rate_fixed = 0.0

    # apply new policy
    mult = df["bucket_id"].map(bucket_multipliers).fillna(1.0).astype(float)
    df["tax_rate_new"] = float(base_tax_rate) * mult
    df["fee_rate_new"] = float(fee_rate_fixed)

    # build X and replace tax/fee rates
    X = _build_X_from_panel(df, model_pack["X_cols"]).copy()
    # IMPORTANT: X columns include tax_rate_eff/fee_rate_eff (we treat them as policy knobs)
    if "tax_rate_eff" in X.columns:
        X["tax_rate_eff"] = df["tax_rate_new"].to_numpy()
    if "fee_rate_eff" in X.columns:
        X["fee_rate_eff"] = df["fee_rate_new"].to_numpy()

    # extensive
    ext = model_pack["ext_model"]
    if ext is None:
        p_trade = np.full(len(df), model_pack["ext_baseline_p"], dtype=float)
    else:
        p_trade = ext.predict_proba(X)[:, 1]

    # intensive
    inten = model_pack["int_model"]
    if inten is None:
        # fallback: use conditional mean logvol from observed traders
        m = np.log1p(df.loc[df["volume_token"] > 0, "volume_token"]).mean()
        pred_log1p = np.full(len(df), float(m) if np.isfinite(m) else 0.0)
    else:
        pred_log1p = inten.predict(X)

    vol_cond = np.expm1(pred_log1p).clip(min=0.0)
    df["exp_volume"] = p_trade * vol_cond
    df["exp_traders"] = p_trade  # expected count contribution

    df["exp_tax_rev"] = df["exp_volume"] * df["tax_rate_new"]
    df["exp_fee_rev"] = df["exp_volume"] * df["fee_rate_new"]
    df["exp_revenue"] = df["exp_tax_rev"] + df["exp_fee_rev"]

    # aggregates
    daily_overall = df.groupby("day").agg(
        exp_volume=("exp_volume", "sum"),
        exp_traders=("exp_traders", "sum"),
        exp_tax=("exp_tax_rev", "sum"),
        exp_fee=("exp_fee_rev", "sum"),
        exp_revenue=("exp_revenue", "sum"),
        avg_tax_rate=("tax_rate_new", "mean"),
    ).reset_index()

    def seg_agg(mask, name):
        d = df.loc[mask].groupby("day").agg(
            exp_volume=("exp_volume", "sum"),
            exp_traders=("exp_traders", "sum"),
            exp_revenue=("exp_revenue", "sum"),
            avg_tax_rate=("tax_rate_new", "mean"),
        ).reset_index()
        d["segment"] = name
        return d

    whales = seg_agg(df["whale_flag"] == 1, "whales")
    nonwhales = seg_agg(df["whale_flag"] == 0, "non_whales")

    return {
        "user_day_pred": df,
        "daily_overall": daily_overall,
        "daily_segments": pd.concat([whales, nonwhales], ignore_index=True),
        "baseline_daily": baseline_daily,
        "fee_rate_used": fee_rate_fixed,
    }


def grid_optimize_policy(
    panel: pd.DataFrame,
    model_pack: dict,
    bucket_labels: list,
    base_tax_grid: np.ndarray,
    prog_strength_grid: np.ndarray,
    objective: str = "revenue",
    revenue_floor: float = None,
    fee_rate_fixed: float = None,
) -> pd.DataFrame:
    """
      multipliers[bucket_i] = 1 + prog_strength * i
    objective: "revenue" | "volume" | "traders"
    revenue_floor: constraint on exp_revenue total (overall sum)
    """
    results = []

    for base_tax in base_tax_grid:
        for ps in prog_strength_grid:
            multipliers = {}
            for i, b in enumerate(bucket_labels):
                multipliers[b] = float(1.0 + ps * i)

            sim = simulate_tax_policy(
                panel=panel,
                model_pack=model_pack,
                base_tax_rate=float(base_tax),
                bucket_multipliers=multipliers,
                fee_rate_fixed=fee_rate_fixed
            )
            daily = sim["daily_overall"]
            tot_rev = float(daily["exp_revenue"].sum())
            tot_vol = float(daily["exp_volume"].sum())
            tot_tr = float(daily["exp_traders"].sum())

            if revenue_floor is not None and tot_rev < revenue_floor:
                continue

            score = {"revenue": tot_rev, "volume": tot_vol, "traders": tot_tr}.get(objective, tot_rev)

            results.append({
                "base_tax_rate": float(base_tax),
                "prog_strength": float(ps),
                "score": float(score),
                "total_revenue": tot_rev,
                "total_volume": tot_vol,
                "total_traders": tot_tr,
            })

    out = pd.DataFrame(results)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False).reset_index(drop=True)
