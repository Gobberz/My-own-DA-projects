import numpy as np
import pandas as pd

def double_auction(n_buyers=200, n_sellers=200, v_min=1.0, v_max=10.0, seed=42):
    """
     Double Auction:
    - buyer value ~ U[v_min, v_max]
    - seller cost  ~ U[v_min, v_max]
    - bid = value, ask = cost (truthful baseline)
    Match: highest bids with lowest asks while bid>=ask.
    Clearing price = mid(bid, ask) per matched pair.
    """
    rng = np.random.default_rng(seed)

    buyers = pd.DataFrame({
        "agent": [f"B{i}" for i in range(n_buyers)],
        "value": rng.uniform(v_min, v_max, size=n_buyers)
    })
    sellers = pd.DataFrame({
        "agent": [f"S{i}" for i in range(n_sellers)],
        "cost": rng.uniform(v_min, v_max, size=n_sellers)
    })

    buyers["bid"] = buyers["value"]
    sellers["ask"] = sellers["cost"]

    buyers = buyers.sort_values("bid", ascending=False).reset_index(drop=True)
    sellers = sellers.sort_values("ask", ascending=True).reset_index(drop=True)

    matches = []
    i = j = 0
    while i < len(buyers) and j < len(sellers):
        bid = buyers.loc[i, "bid"]
        ask = sellers.loc[j, "ask"]
        if bid >= ask:
            price = 0.5 * (bid + ask)
            welfare = buyers.loc[i, "value"] - sellers.loc[j, "cost"]
            matches.append({
                "buyer": buyers.loc[i, "agent"],
                "seller": sellers.loc[j, "agent"],
                "bid": float(bid),
                "ask": float(ask),
                "price": float(price),
                "welfare": float(welfare),
            })
            i += 1
            j += 1
        else:
            break

    trades = pd.DataFrame(matches)
    out = {
        "buyers": buyers,
        "sellers": sellers,
        "trades": trades,
        "n_trades": int(len(trades)),
        "clearing_price_avg": float(trades["price"].mean()) if len(trades) else np.nan,
        "total_welfare": float(trades["welfare"].sum()) if len(trades) else 0.0,
    }
    return out
