import numpy as np
import pandas as pd

SINK_TYPES = {"tax", "fee", "market_fee", "burn"}
EMIT_TYPES = {"reward"}
LOCK_TYPES = {"stake", "unstake"}

def _dt(s):
    return pd.to_datetime(s, errors="coerce")

def compute_daily_econ_panel(market_trades: pd.DataFrame, token_ledger: pd.DataFrame) -> pd.DataFrame:

    mt = market_trades.copy()
    tl = token_ledger.copy()

    # --- trades ---
    mt["ts"] = _dt(mt["ts"])
    mt["day"] = mt["ts"].dt.floor("D")
    mt["amount_base"] = pd.to_numeric(mt.get("amount_base"), errors="coerce").fillna(0.0)
    mt["price"] = pd.to_numeric(mt.get("price"), errors="coerce").fillna(0.0)
    mt["gross_token"] = mt["amount_base"] * mt["price"]

    if "maker_user_id" in mt.columns:
        mt["user_id"] = mt["maker_user_id"]
    else:
        mt["user_id"] = mt.get("user_id")

    for c in ["slippage_bps", "spread_bps", "tax_amount", "fee_amount"]:
        if c in mt.columns:
            mt[c] = pd.to_numeric(mt[c], errors="coerce")

    # aggregate day-level
    day = mt.groupby("day").agg(
        volume=("gross_token", "sum"),
        trades=("trade_id", "count") if "trade_id" in mt.columns else ("gross_token", "size"),
        traders=("user_id", pd.Series.nunique),
        slippage_p50=("slippage_bps", lambda x: np.nanmedian(x)) if "slippage_bps" in mt.columns else ("gross_token", lambda x: np.nan),
        slippage_p90=("slippage_bps", lambda x: np.nanpercentile(x.dropna(), 90) if x.dropna().size else np.nan) if "slippage_bps" in mt.columns else ("gross_token", lambda x: np.nan),
        spread_p50=("spread_bps", lambda x: np.nanmedian(x)) if "spread_bps" in mt.columns else ("gross_token", lambda x: np.nan),
        spread_p90=("spread_bps", lambda x: np.nanpercentile(x.dropna(), 90) if x.dropna().size else np.nan) if "spread_bps" in mt.columns else ("gross_token", lambda x: np.nan),
    ).reset_index()

    # concentration: top1% user volume share per day
    uv = mt.groupby(["day", "user_id"])["gross_token"].sum().reset_index()
    def top_share_day(g, p=0.01):
        x = g["gross_token"].to_numpy()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        x = np.sort(x)[::-1]
        k = max(1, int(np.ceil(p * len(x))))
        return float(x[:k].sum() / x.sum()) if x.sum() > 0 else 0.0

    conc = uv.groupby("day").apply(top_share_day).reset_index()
    conc.columns = ["day", "top1_volume_share"]
    day = day.merge(conc, on="day", how="left")

    # trader churn proxy: traders active in last 7d but not in current day
    traders_by_day = mt.groupby("day")["user_id"].apply(lambda s: set(s.dropna().unique())).to_dict()
    days_sorted = sorted(traders_by_day.keys())
    churn_rows = []
    for d in days_sorted:
        prev_window = [x for x in days_sorted if (x < d and x >= d - pd.Timedelta(days=7))]
        prev_traders = set().union(*[traders_by_day[x] for x in prev_window]) if prev_window else set()
        cur_traders = traders_by_day.get(d, set())
        churned = prev_traders - cur_traders
        churn_rows.append({"day": d, "trader_churn_proxy": len(churned), "prev7_traders": len(prev_traders)})
    churn = pd.DataFrame(churn_rows)
    churn["trader_churn_rate_proxy"] = churn["trader_churn_proxy"] / churn["prev7_traders"].replace(0, np.nan)
    day = day.merge(churn[["day", "trader_churn_proxy", "trader_churn_rate_proxy"]], on="day", how="left")

    # --- ledger: issuance / sinks / circulating proxy ---
    tl["ts"] = _dt(tl["ts"])
    tl["day"] = tl["ts"].dt.floor("D")
    tl["amount"] = pd.to_numeric(tl.get("amount"), errors="coerce").fillna(0.0)
    tl["tx_type"] = tl.get("tx_type", "unknown").astype(str)

    led = tl.groupby(["day", "tx_type"])["amount"].sum().reset_index()
    piv = led.pivot_table(index="day", columns="tx_type", values="amount", aggfunc="sum", fill_value=0.0).reset_index()
    piv.columns = [str(c) for c in piv.columns]

    # robust extract
    emission = np.zeros(len(piv))
    for t in EMIT_TYPES:
        if t in piv.columns:
            emission += piv[t].to_numpy()

    sink = np.zeros(len(piv))
    for t in SINK_TYPES:
        if t in piv.columns:
            amt = piv[t].to_numpy()
            # sink should be positive collected value (handle sign)
            sink += np.where(amt < 0, -amt, amt)

    stake = piv["stake"].to_numpy() if "stake" in piv.columns else 0.0
    unstake = piv["unstake"].to_numpy() if "unstake" in piv.columns else 0.0
    locked_delta = stake + unstake  # depending on your sign convention this is proxy

    piv["emission"] = emission
    piv["sinks"] = sink
    piv["net_issuance"] = piv["emission"] - piv["sinks"]

    piv["supply_proxy"] = piv["net_issuance"].cumsum()
    piv["locked_proxy"] = pd.Series(locked_delta).cumsum()
    piv["circulating_proxy"] = (piv["supply_proxy"] - piv["locked_proxy"]).clip(lower=1e-6)

    panel = day.merge(piv[["day", "emission", "sinks", "net_issuance", "supply_proxy", "locked_proxy", "circulating_proxy"]],
                      on="day", how="left")

    panel["velocity"] = panel["volume"] / panel["circulating_proxy"].replace(0, np.nan)
    panel = panel.sort_values("day")
    return panel
