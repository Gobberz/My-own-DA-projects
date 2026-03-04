import numpy as np
import pandas as pd

def zscore(s: pd.Series, window: int = 30):
    """
    Rolling z-score (robust-ish): (x - mean)/std on rolling window.
    """
    mu = s.rolling(window, min_periods=max(5, window//3)).mean()
    sd = s.rolling(window, min_periods=max(5, window//3)).std()
    return (s - mu) / sd.replace(0, np.nan)

def build_alerts(panel: pd.DataFrame,
                 z_thr_high: float = 3.0,
                 z_thr_low: float = -3.0) -> pd.DataFrame:
    """
    panel: day-level panel from compute_daily_econ_panel
    returns alerts table with severity & reason
    """
    df = panel.copy().sort_values("day")
    alerts = []

    def add(rule, day, metric, value, z, severity, detail=""):
        alerts.append({
            "day": day,
            "rule": rule,
            "metric": metric,
            "value": float(value) if value is not None and np.isfinite(value) else value,
            "z": float(z) if z is not None and np.isfinite(z) else z,
            "severity": severity,
            "detail": detail
        })

    # 1) net issuance spike
    if "net_issuance" in df.columns:
        zs = zscore(df["net_issuance"])
        for d, v, z in zip(df["day"], df["net_issuance"], zs):
            if np.isfinite(z) and z >= z_thr_high:
                add("NET_ISSUANCE_SPIKE", d, "net_issuance", v, z, "HIGH", "Inflation/exploit risk")

    # 2) velocity drop
    if "velocity" in df.columns:
        zs = zscore(df["velocity"])
        for d, v, z in zip(df["day"], df["velocity"], zs):
            if np.isfinite(z) and z <= z_thr_low:
                add("VELOCITY_DROP", d, "velocity", v, z, "MED", "Tokens stuck / demand drop")

    # 3) spread/slippage spike
    for metric, rule in [("spread_p90", "SPREAD_SPIKE"), ("slippage_p90", "SLIPPAGE_SPIKE")]:
        if metric in df.columns:
            zs = zscore(df[metric])
            for d, v, z in zip(df["day"], df[metric], zs):
                if np.isfinite(z) and z >= z_thr_high:
                    add(rule, d, metric, v, z, "HIGH", "Market quality degraded")

    # 4) concentration up
    if "top1_volume_share" in df.columns:
        zs = zscore(df["top1_volume_share"])
        for d, v, z in zip(df["day"], df["top1_volume_share"], zs):
            if np.isfinite(z) and z >= z_thr_high:
                add("CONCENTRATION_UP", d, "top1_volume_share", v, z, "MED", "Whales dominating volume")

    # 5) trader churn spike
    if "trader_churn_rate_proxy" in df.columns:
        zs = zscore(df["trader_churn_rate_proxy"])
        for d, v, z in zip(df["day"], df["trader_churn_rate_proxy"], zs):
            if np.isfinite(z) and z >= z_thr_high:
                add("TRADER_CHURN_SPIKE", d, "trader_churn_rate_proxy", v, z, "HIGH", "Trader retention issue")

    out = pd.DataFrame(alerts)
    if out.empty:
        out = pd.DataFrame(columns=["day","rule","metric","value","z","severity","detail"])
    return out.sort_values(["day","severity"], ascending=[False, True])
