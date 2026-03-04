# pages/06_GameTheoryLab.py
# Game Theory Lab — 6 simulators (tabs)
# All UI text and comments are in English.
#
# Tabs:
# 1) Double Auction (P2P energy trading) — market mechanism + equilibrium-ish outcomes
# 2) Tragedy of the Commons — resource dynamics + punishment/cooperation levers
# 3) Bullwhip Effect — information asymmetry in supply chains
# 4) Shapley Value — fair payout allocation for coalition value
# 5) Mean Field (Congestion / Route choice) — many agents interact via an aggregate field
# 6) Mechanism Design — truthful auctions (Vickrey/VCG) + ad auction (GSP-like)

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
st.set_page_config(page_title="Game Theory Lab", layout="wide")
st.title("Game Theory Lab — 6 Simulators")
st.caption("Interactive game-theory playground: mechanisms, incentives, equilibria, and system dynamics.")


# =========================
# Helpers
# =========================
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a / np.where(b == 0, np.nan, b)

def _line(df: pd.DataFrame, x: str, y: List[str], title: str):
    if df.empty:
        st.info("No data to plot.")
        return
    if PLOTLY_OK:
        fig = go.Figure()
        for col in y:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=col))
        fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df.set_index(x)[y])

def _hist(series: pd.Series, title: str, nbins: int = 40):
    if series is None or series.dropna().empty:
        st.info("No data to plot.")
        return
    if PLOTLY_OK:
        fig = px.histogram(series.dropna(), nbins=nbins, title=title)
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(series.describe())

def _bar(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        st.info("No data to plot.")
        return
    if PLOTLY_OK:
        fig = px.bar(df, x=x, y=y, title=title)
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index(x)[y])

def _download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")


# =========================
# 1) Double Auction (P2P Energy Trading)
# =========================
@dataclass
class Order:
    agent_id: int
    side: str        # "buy" or "sell"
    qty: float
    price: float

def simulate_double_auction(
    n_agents: int,
    T: int,
    mechanism: str,            # "uniform" or "pay_as_bid"
    bid_markup: float,
    ask_markdown: float,
    noise: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple double auction:
    - Each day t: agents have random net energy (surplus>0 => seller, deficit<0 => buyer)
    - Sellers submit asks; buyers submit bids
    - We match sorted bids/asks while bid>=ask
    - Uniform: clearing price = midpoint(best_bid, best_ask) at each match (or last matched)
    - Pay-as-bid: buyers pay their bid, sellers receive their ask (spread captured by "market")
    Returns:
      panel: per-time metrics
      trades: per-trade record
    """
    rng = _rng(seed)
    rows = []
    trades = []

    for t in range(T):
        # Demand/supply shocks (proxy for solar vs load)
        net = rng.normal(loc=0.0, scale=1.0, size=n_agents)  # >0 surplus, <0 deficit
        net = net + rng.normal(0, noise, size=n_agents)

        # Convert to quantities (kWh-ish)
        qty = np.clip(np.abs(net) * rng.uniform(0.5, 2.0, size=n_agents), 0, None)

        # Base "fundamental" price changes with imbalance
        imbalance = net.sum()  # positive => surplus => cheaper
        p0 = 1.0 + (-0.15 * np.tanh(imbalance / max(1.0, n_agents/3)))
        p0 = max(0.1, p0)

        orders: List[Order] = []
        for i in range(n_agents):
            if net[i] < 0:  # buyer
                # bid a bit above fundamental
                b = p0 * (1.0 + bid_markup) * rng.uniform(0.9, 1.1)
                orders.append(Order(i, "buy", float(qty[i]), float(b)))
            elif net[i] > 0:  # seller
                # ask a bit below fundamental
                a = p0 * (1.0 - ask_markdown) * rng.uniform(0.9, 1.1)
                orders.append(Order(i, "sell", float(qty[i]), float(a)))

        bids = sorted([o for o in orders if o.side == "buy"], key=lambda o: o.price, reverse=True)
        asks = sorted([o for o in orders if o.side == "sell"], key=lambda o: o.price)

        traded_qty = 0.0
        n_trades = 0
        prices = []
        market_spread_profit = 0.0

        bi = 0
        ai = 0
        while bi < len(bids) and ai < len(asks):
            b = bids[bi]
            a = asks[ai]
            if b.price < a.price:
                break

            q = min(b.qty, a.qty)
            if q <= 0:
                break

            if mechanism == "uniform":
                px = 0.5 * (b.price + a.price)
                buyer_pay = px
                seller_get = px
                spread = 0.0
            else:
                # pay-as-bid: buyer pays bid, seller gets ask, spread captured by market
                buyer_pay = b.price
                seller_get = a.price
                px = 0.5 * (buyer_pay + seller_get)
                spread = (buyer_pay - seller_get)

            traded_qty += q
            n_trades += 1
            prices.append(px)
            market_spread_profit += spread * q

            trades.append({
                "t": t,
                "buyer": b.agent_id,
                "seller": a.agent_id,
                "qty": q,
                "bid": b.price,
                "ask": a.price,
                "price": px,
                "mechanism": mechanism,
                "spread_profit": spread * q
            })

            # reduce remaining qty
            b.qty -= q
            a.qty -= q
            if b.qty <= 1e-9:
                bi += 1
            if a.qty <= 1e-9:
                ai += 1

        avg_price = float(np.mean(prices)) if prices else np.nan
        vwap = float(np.average(prices, weights=[tr["qty"] for tr in trades if tr["t"] == t])) if prices else np.nan

        rows.append({
            "t": t,
            "p0_fundamental": p0,
            "imbalance": imbalance,
            "n_buyers": int((net < 0).sum()),
            "n_sellers": int((net > 0).sum()),
            "traded_qty": traded_qty,
            "n_trades": n_trades,
            "avg_trade_price": avg_price,
            "vwap": vwap,
            "market_spread_profit": market_spread_profit,
            "mechanism": mechanism
        })

    panel = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    return panel, trades_df


# =========================
# 2) Tragedy of the Commons (Resource dynamics)
# =========================
def simulate_commons(
    n_agents: int,
    T: int,
    r: float,                 # regeneration rate
    K: float,                 # carrying capacity
    stock0: float,
    strategy: str,            # "greedy", "quota", "punish"
    quota: float,
    punish_strength: float,
    noise: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resource stock follows logistic growth:
      S_{t+1} = S_t + r*S_t*(1 - S_t/K) - total_harvest_t
    Strategies:
      greedy: each agent harvests proportional to stock
      quota: each agent harvests fixed quota (bounded by stock)
      punish: if total harvest exceeds sustainable threshold, next step harvest is reduced by punishment factor
    Returns:
      panel: stock + total harvest + collapse flag per t
      agent_payoffs: per-agent cumulative payoff
    """
    rng = _rng(seed)

    S = float(stock0)
    pay = np.zeros(n_agents, dtype=float)
    rows = []

    # "sustainable" per-step harvest rough heuristic
    sustainable = 0.25 * r * K  # crude: fraction of max growth

    punish_factor = 1.0
    for t in range(T):
        if S <= 0:
            rows.append({"t": t, "stock": 0.0, "total_harvest": 0.0, "punish_factor": punish_factor, "collapsed": 1})
            continue

        # harvest decision
        if strategy == "greedy":
            base = 0.03 * S
            h = rng.uniform(0.5, 1.5, size=n_agents) * base
        elif strategy == "quota":
            h = np.full(n_agents, quota, dtype=float)
        else:  # punish
            base = 0.03 * S
            h = rng.uniform(0.5, 1.5, size=n_agents) * base
            h = h * punish_factor

        # add decision noise
        h = np.clip(h + rng.normal(0, noise, size=n_agents), 0, None)

        total_h = float(h.sum())
        # can't harvest more than stock
        if total_h > S:
            scale = S / max(total_h, 1e-9)
            h *= scale
            total_h = float(h.sum())

        # payoff = harvest amount
        pay += h

        # update stock
        growth = r * S * (1.0 - S / K)
        S = S + growth - total_h
        S = max(0.0, S)

        # punishment update
        if strategy == "punish":
            if total_h > sustainable:
                punish_factor = max(0.2, punish_factor * (1.0 - punish_strength))
            else:
                punish_factor = min(1.0, punish_factor + 0.05)  # recover cooperation

        rows.append({
            "t": t,
            "stock": S,
            "growth": growth,
            "total_harvest": total_h,
            "avg_harvest": total_h / n_agents,
            "sustainable": sustainable,
            "punish_factor": punish_factor,
            "collapsed": int(S <= 1e-9)
        })

    panel = pd.DataFrame(rows)
    agent_payoffs = pd.DataFrame({"agent": np.arange(n_agents), "cumulative_payoff": pay})
    agent_payoffs["share"] = agent_payoffs["cumulative_payoff"] / agent_payoffs["cumulative_payoff"].sum()
    return panel, agent_payoffs


# =========================
# 3) Bullwhip Effect (Supply chain)
# =========================
def simulate_bullwhip(
    T: int,
    demand_mean: float,
    demand_sigma: float,
    lead_time: int,
    smoothing: float,            # 0..1 (higher = more smoothing)
    info_sharing: bool,
    seed: int
) -> pd.DataFrame:
    """
    4-stage chain: Retail -> Wholesaler -> Distributor -> Factory
    Each stage uses an order-up-to-ish rule based on demand forecast.
    Classic outcome: order variance amplifies upstream (bullwhip), especially without info sharing.
    """
    rng = _rng(seed)
    stages = ["Retail", "Wholesaler", "Distributor", "Factory"]

    # arrays
    demand = np.clip(rng.normal(demand_mean, demand_sigma, size=T), 0, None)

    # Each stage maintains forecast and places orders
    forecast = {s: demand_mean for s in stages}
    orders = {s: np.zeros(T) for s in stages}

    # pipeline / lead time buffers
    pipeline = {s: [0.0] * lead_time for s in stages}

    for t in range(T):
        # observed demand at retail
        d_obs = demand[t]

        # downstream signal (what stage sees)
        downstream = d_obs
        for idx, s in enumerate(stages):
            if info_sharing and idx > 0:
                # upstream can see true retail demand (perfect sharing)
                signal = d_obs
            else:
                # upstream sees only orders from immediate downstream
                signal = downstream

            # update exponential smoothing forecast
            forecast[s] = smoothing * forecast[s] + (1.0 - smoothing) * signal

            # simple order rule: forecast + pipeline correction
            target = forecast[s] * (1.0 + 0.3)  # safety factor
            in_pipe = sum(pipeline[s])
            order = max(0.0, target - in_pipe)

            orders[s][t] = order

            # push into pipeline
            pipeline[s].append(order)
            arrived = pipeline[s].pop(0)

            # pass upstream the order as "demand"
            downstream = order

    df = pd.DataFrame({"t": np.arange(T), "Demand_Retail": demand})
    for s in stages:
        df[f"Order_{s}"] = orders[s]

    # Variance amplification (bullwhip ratio)
    base_var = np.var(demand) if np.var(demand) > 0 else np.nan
    for s in stages:
        df[f"VarRatio_{s}"] = np.var(orders[s]) / base_var if base_var == base_var else np.nan  # keep nan-safe
    return df


# =========================
# 4) Shapley Value (Fair payouts)
# =========================
def shapley_exact(players: List[str], v_func) -> Dict[str, float]:
    """
    Exact Shapley for n <= ~10 (2^n coalitions expensive).
    v_func(S) returns value for coalition S (tuple of players).
    """
    n = len(players)
    phi = {p: 0.0 for p in players}
    fact = math.factorial

    for p in players:
        others = [x for x in players if x != p]
        for k in range(n):
            for S in itertools.combinations(others, k):
                S = tuple(S)
                S_with = tuple(sorted(S + (p,)))
                w = fact(k) * fact(n - k - 1) / fact(n)
                phi[p] += w * (v_func(S_with) - v_func(S))
    return phi

def shapley_monte_carlo(players: List[str], v_func, n_perm: int, seed: int) -> Dict[str, float]:
    rng = _rng(seed)
    phi = {p: 0.0 for p in players}
    for _ in range(n_perm):
        perm = players.copy()
        rng.shuffle(perm)
        S = tuple()
        prev = v_func(S)
        for p in perm:
            S = tuple(sorted(S + (p,)))
            cur = v_func(S)
            phi[p] += (cur - prev)
            prev = cur
    for p in players:
        phi[p] /= n_perm
    return phi


# =========================
# 5) Mean Field (Congestion / Route Choice)
# =========================
def simulate_mean_field_routes(
    N: int,
    T: int,
    alpha: float,     # congestion slope
    beta: float,      # choice sensitivity
    baseA: float,
    baseB: float,
    seed: int
) -> pd.DataFrame:
    """
    Many agents choose route A or B.
    Cost depends on fraction on the same route (mean field):
      cA = baseA + alpha * fracA
      cB = baseB + alpha * fracB
    Choice via logit:
      P(A) = exp(-beta*cA) / (exp(-beta*cA)+exp(-beta*cB))
    Iterate to equilibrium-ish fixed point.
    """
    rng = _rng(seed)

    fracA = 0.5
    rows = []
    for t in range(T):
        fracB = 1.0 - fracA
        cA = baseA + alpha * fracA
        cB = baseB + alpha * fracB

        # Logit choice
        pA = math.exp(-beta*cA) / (math.exp(-beta*cA) + math.exp(-beta*cB))
        # finite population sampling
        nA = rng.binomial(N, pA)
        fracA = nA / N

        rows.append({
            "t": t,
            "fracA": fracA,
            "fracB": 1.0 - fracA,
            "costA": cA,
            "costB": cB,
            "pA_logit": pA,
        })
    return pd.DataFrame(rows)


# =========================
# 6) Mechanism Design (Auctions)
# =========================
def simulate_auction_truthfulness(
    n_bidders: int,
    n_rounds: int,
    value_low: float,
    value_high: float,
    shading: float,        # misreport factor for "lie": bid = value*(1-shading)
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare First-price vs Second-price (Vickrey) single-item auction.
    Each bidder has private value ~ Uniform[value_low, value_high].
    We compare two bidding strategies:
      - truthful: bid = value
      - shaded: bid = value*(1-shading)
    Under Vickrey, truthful is (weakly) dominant for utility.
    """
    rng = _rng(seed)
    rounds = []
    bidder_rows = []

    for t in range(n_rounds):
        values = rng.uniform(value_low, value_high, size=n_bidders)

        for strat in ["truthful", "shaded"]:
            bids = values.copy() if strat == "truthful" else values * (1.0 - shading)

            # first-price outcome
            w_fp = int(np.argmax(bids))
            bid_fp = float(bids[w_fp])
            val_fp = float(values[w_fp])
            util_fp = val_fp - bid_fp

            # second-price outcome
            # winner is highest bid, pays second-highest bid
            order = np.argsort(bids)[::-1]
            w_sp = int(order[0])
            pay_sp = float(bids[order[1]]) if n_bidders >= 2 else 0.0
            val_sp = float(values[w_sp])
            util_sp = val_sp - pay_sp

            rounds.append({
                "round": t,
                "strategy": strat,
                "fp_winner": w_fp,
                "fp_price": bid_fp,
                "fp_winner_value": val_fp,
                "fp_winner_utility": util_fp,
                "sp_winner": w_sp,
                "sp_price": pay_sp,
                "sp_winner_value": val_sp,
                "sp_winner_utility": util_sp,
            })

            for i in range(n_bidders):
                bidder_rows.append({
                    "round": t,
                    "strategy": strat,
                    "bidder": i,
                    "value": float(values[i]),
                    "bid": float(bids[i]),
                    "won_fp": int(i == w_fp),
                    "won_sp": int(i == w_sp),
                    "utility_fp": float(val_fp - bid_fp) if i == w_fp else 0.0,
                    "utility_sp": float(val_sp - pay_sp) if i == w_sp else 0.0,
                })

    df_round = pd.DataFrame(rounds)
    df_bidder = pd.DataFrame(bidder_rows)
    return df_round, df_bidder

def simulate_gsp_ads(
    n_bidders: int,
    n_slots: int,
    n_rounds: int,
    seed: int
) -> pd.DataFrame:
    """
    Simple GSP-like ad auction:
      - Each bidder has value-per-click v and quality q
      - Rank score = bid * q
      - Pay per click = next_score / own_q
      - Utility = (v - price) * clicks
    This is not a full equilibrium solver; it's a sandbox for incentive effects.
    """
    rng = _rng(seed)
    rows = []

    ctr = np.linspace(1.0, 0.3, n_slots)  # slot CTRs
    for t in range(n_rounds):
        v = rng.uniform(0.5, 2.0, size=n_bidders)
        q = rng.uniform(0.5, 1.5, size=n_bidders)
        # naive bidding: bid equals value (truthful-ish) or small shading
        bid = v * rng.uniform(0.8, 1.0, size=n_bidders)

        score = bid * q
        order = np.argsort(score)[::-1]
        winners = order[:n_slots]

        # compute per-click prices
        prices = np.zeros(n_slots)
        for s in range(n_slots):
            if s == n_slots - 1:
                next_score = 0.0
            else:
                next_score = score[order[s+1]]
            prices[s] = next_score / max(q[winners[s]], 1e-9)

        for slot, bidder in enumerate(winners):
            clicks = ctr[slot]
            price = prices[slot]
            util = (v[bidder] - price) * clicks
            rows.append({
                "round": t,
                "slot": slot,
                "bidder": int(bidder),
                "value": float(v[bidder]),
                "quality": float(q[bidder]),
                "bid": float(bid[bidder]),
                "score": float(score[bidder]),
                "price_per_click": float(price),
                "clicks": float(clicks),
                "utility": float(util),
                "revenue": float(price * clicks),
            })

    return pd.DataFrame(rows)


# =========================
# Layout — Tabs
# =========================
tabs = st.tabs([
    "1) Double Auction (P2P Energy)",
    "2) Commons (Resource)",
    "3) Bullwhip (Supply Chain)",
    "4) Shapley (Fair Payouts)",
    "5) Mean Field (Congestion)",
    "6) Mechanism Design (Auctions)"
])

# -------------------------------------------------
# TAB 1: Double Auction
# -------------------------------------------------
with tabs[0]:
    st.subheader("Double Auction — P2P energy trading between households")
    st.markdown(
        """
**What this tab is about (the idea):**  
Households have surplus/deficit energy each time step. They submit **bids** (buyers) and **asks** (sellers).  
A double auction matches orders and produces prices/volumes. You can compare mechanisms:

- **Uniform clearing price**: trades clear at a common “market” price (midpoint here).
- **Pay-as-bid**: buyers pay their bid, sellers receive their ask; spread becomes market profit.

**Why it matters:**  
Mechanism choice changes **efficiency**, **price stability**, and **who captures surplus** (users vs protocol/market).
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    n_agents = c1.slider("Number of households", 50, 2000, 400, 50)
    T = c2.slider("Time steps", 20, 365, 120, 10)
    mechanism = c3.selectbox("Mechanism", ["uniform", "pay_as_bid"], index=0)
    seed = c4.number_input("Seed", 1, 10_000_000, 42, 1)

    c5, c6, c7 = st.columns(3)
    bid_markup = c5.slider("Buyer bid markup", 0.0, 0.50, 0.10, 0.01)
    ask_markdown = c6.slider("Seller ask markdown", 0.0, 0.50, 0.10, 0.01)
    noise = c7.slider("Shock noise", 0.0, 2.0, 0.30, 0.05)

    run = st.button("Run Double Auction simulation", type="primary")
    if run:
        with st.spinner("Simulating..."):
            panel, trades = simulate_double_auction(
                n_agents=n_agents, T=T, mechanism=mechanism,
                bid_markup=bid_markup, ask_markdown=ask_markdown,
                noise=noise, seed=int(seed)
            )

        left, right = st.columns([1.2, 1.0])
        with left:
            _line(panel, "t", ["traded_qty", "n_trades"], "Market activity over time")
        with right:
            _line(panel, "t", ["avg_trade_price", "vwap", "p0_fundamental"], "Prices vs fundamental")

        st.metric("Total traded qty", f"{panel['traded_qty'].sum():,.2f}")
        st.metric("Total trades", f"{panel['n_trades'].sum():,}")
        st.metric("Total market spread profit (pay-as-bid)", f"{panel['market_spread_profit'].sum():,.4f}")

        with st.expander("Show raw tables"):
            st.dataframe(panel, use_container_width=True)
            st.dataframe(trades.head(5000), use_container_width=True)

        _download_df(panel, "double_auction_panel.csv", "Download panel CSV")
        _download_df(trades, "double_auction_trades.csv", "Download trades CSV")


# -------------------------------------------------
# TAB 2: Commons
# -------------------------------------------------
with tabs[1]:
    st.subheader("Tragedy of the Commons — shared resource, over-harvesting, and punishment")
    st.markdown(
        """
**What this tab is about:**  
A shared resource stock regenerates (logistic growth). Agents harvest each step.  
If harvesting is too aggressive, the resource collapses. We test “institutions”:

- **Greedy**: everyone takes as much as they can.
- **Quota**: fixed per-agent limit (rule-based regulation).
- **Punishment**: if overuse happens, next harvest is reduced (social enforcement).

**Why it matters:**  
This mirrors DeFi/game economies: without sinks/limits/policies, systems can get exploited and collapse.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    n_agents = c1.slider("Agents", 10, 500, 80, 10)
    T = c2.slider("Steps", 30, 500, 200, 10)
    seed = c3.number_input("Seed", 1, 10_000_000, 123, 1)
    strategy = c4.selectbox("Strategy / Institution", ["greedy", "quota", "punish"], index=2)

    c5, c6, c7, c8 = st.columns(4)
    r = c5.slider("Regeneration rate r", 0.01, 0.40, 0.10, 0.01)
    K = c6.slider("Carrying capacity K", 100.0, 5000.0, 1000.0, 50.0)
    stock0 = c7.slider("Initial stock", 10.0, 5000.0, 800.0, 50.0)
    noise = c8.slider("Decision noise", 0.0, 10.0, 1.0, 0.1)

    c9, c10 = st.columns(2)
    quota = c9.slider("Quota per agent (if quota)", 0.0, 50.0, 5.0, 0.5)
    punish_strength = c10.slider("Punishment strength (if punish)", 0.0, 0.5, 0.15, 0.01)

    run = st.button("Run Commons simulation", type="primary")
    if run:
        with st.spinner("Simulating..."):
            panel, payoff = simulate_commons(
                n_agents=n_agents, T=T, r=r, K=K, stock0=stock0,
                strategy=strategy, quota=quota, punish_strength=punish_strength,
                noise=noise, seed=int(seed)
            )

        colA, colB = st.columns([1.2, 1.0])
        with colA:
            _line(panel, "t", ["stock", "total_harvest", "sustainable"], "Stock and harvesting")
        with colB:
            _hist(payoff["cumulative_payoff"], "Distribution of cumulative payoffs")

        collapse_rate = panel["collapsed"].max()
        st.metric("Collapsed?", "YES" if collapse_rate == 1 else "NO")
        st.metric("Final stock", f"{panel['stock'].iloc[-1]:,.2f}")
        st.metric("Total harvested", f"{panel['total_harvest'].sum():,.2f}")

        with st.expander("Show raw tables"):
            st.dataframe(panel, use_container_width=True)
            st.dataframe(payoff.sort_values("cumulative_payoff", ascending=False).head(200), use_container_width=True)

        _download_df(panel, "commons_panel.csv", "Download panel CSV")
        _download_df(payoff, "commons_payoffs.csv", "Download payoffs CSV")


# -------------------------------------------------
# TAB 3: Bullwhip
# -------------------------------------------------
with tabs[2]:
    st.subheader("Bullwhip Effect — demand distortion from information asymmetry")
    st.markdown(
        """
**What this tab is about:**  
Retail sees real customer demand, but upstream stages often see only **orders** from downstream.  
Forecasting + safety stock can amplify variance upstream (the bullwhip effect).

**Why it matters:**  
In game economies and DeFi supply (resources, liquidity), delayed signals can create volatility and inefficiency.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    T = c1.slider("Steps (days)", 30, 365, 180, 10)
    demand_mean = c2.slider("Demand mean", 1.0, 200.0, 50.0, 1.0)
    demand_sigma = c3.slider("Demand sigma", 0.0, 100.0, 15.0, 1.0)
    seed = c4.number_input("Seed", 1, 10_000_000, 77, 1)

    c5, c6 = st.columns(2)
    lead_time = c5.slider("Lead time", 1, 20, 6, 1)
    smoothing = c6.slider("Forecast smoothing (higher = smoother)", 0.0, 0.95, 0.70, 0.05)

    info_sharing = st.checkbox("Enable perfect info sharing (upstream sees true retail demand)", value=False)

    run = st.button("Run Bullwhip simulation", type="primary")
    if run:
        with st.spinner("Simulating..."):
            df = simulate_bullwhip(
                T=T, demand_mean=demand_mean, demand_sigma=demand_sigma,
                lead_time=lead_time, smoothing=smoothing, info_sharing=info_sharing,
                seed=int(seed)
            )

        _line(df, "t", ["Demand_Retail", "Order_Retail", "Order_Wholesaler", "Order_Distributor", "Order_Factory"],
              "Demand and orders across the chain")

        ratios = pd.DataFrame({
            "Stage": ["Retail","Wholesaler","Distributor","Factory"],
            "Variance ratio vs retail demand": [
                float(df["VarRatio_Retail"].iloc[0]),
                float(df["VarRatio_Wholesaler"].iloc[0]),
                float(df["VarRatio_Distributor"].iloc[0]),
                float(df["VarRatio_Factory"].iloc[0]),
            ]
        })
        _bar(ratios, "Stage", "Variance ratio vs retail demand", "Bullwhip: variance amplification upstream")

        with st.expander("Show raw table"):
            st.dataframe(df.head(400), use_container_width=True)

        _download_df(df, "bullwhip_panel.csv", "Download CSV")


# -------------------------------------------------
# TAB 4: Shapley
# -------------------------------------------------
with tabs[3]:
    st.subheader("Shapley Value — fair payouts for coalition value")
    st.markdown(
        """
**What this tab is about:**  
Shapley value allocates total value to players based on their **marginal contribution** across all coalition orders.

**Why it matters:**  
Perfect for alliances/guilds in your DeFi game: you can justify reward distribution using a principled method.
        """
    )

    st.caption("Choose number of players and define a simple coalition value function via skill weights + synergy.")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Players", 2, 12, 6, 1)
    synergy = c2.slider("Synergy strength", 0.0, 2.0, 0.5, 0.05)
    seed = c3.number_input("Seed", 1, 10_000_000, 2024, 1)

    rng = _rng(int(seed))
    players = [f"P{i+1}" for i in range(n)]
    base_skill = rng.uniform(0.5, 2.0, size=n)
    st.write("Base skills (randomized by seed):")
    st.dataframe(pd.DataFrame({"player": players, "skill": base_skill}).round(3), use_container_width=True)

    # Define coalition value function
    def v_func(S: Tuple[str, ...]) -> float:
        if len(S) == 0:
            return 0.0
        idx = [players.index(p) for p in S]
        s = float(base_skill[idx].sum())
        # synergy increases with coalition size (diminishing returns)
        syn = synergy * math.log1p(len(S)) * (np.mean(base_skill[idx]) if len(idx) else 0.0)
        return s + syn

    method = st.selectbox("Computation method", ["Exact (n<=10 recommended)", "Monte Carlo (fast for n>10)"], index=0)
    if method.startswith("Exact") and n > 10:
        st.warning("Exact Shapley gets expensive above ~10 players. Consider Monte Carlo.")

    perms = st.slider("Monte Carlo permutations (if MC)", 200, 20_000, 4000, 200)

    run = st.button("Compute Shapley payouts", type="primary")
    if run:
        with st.spinner("Computing..."):
            if method.startswith("Exact") and n <= 10:
                phi = shapley_exact(players, v_func)
            else:
                phi = shapley_monte_carlo(players, v_func, n_perm=int(perms), seed=int(seed))

        df_phi = pd.DataFrame({"player": list(phi.keys()), "shapley": list(phi.values())})
        df_phi = df_phi.sort_values("shapley", ascending=False)
        total_value = v_func(tuple(players))
        df_phi["share"] = df_phi["shapley"] / total_value if total_value > 0 else np.nan

        st.metric("Coalition total value v(All)", f"{total_value:.4f}")
        st.dataframe(df_phi.round(6), use_container_width=True)

        _bar(df_phi, "player", "shapley", "Shapley payouts")
        _download_df(df_phi, "shapley_payouts.csv", "Download CSV")


# -------------------------------------------------
# TAB 5: Mean Field
# -------------------------------------------------
with tabs[4]:
    st.subheader("Mean Field Game — congestion / route choice with many agents")
    st.markdown(
        """
**What this tab is about:**  
When the number of agents is huge, we model interaction through an aggregate “field” (like congestion).  
Agents choose between A and B; costs depend on how many choose each route.

**Why it matters:**  
DeFi markets often behave like mean-field systems: individual actions are small, but collectively shape the field (price, congestion, slippage).
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    N = c1.slider("Agents N", 100, 200_000, 10_000, 100)
    T = c2.slider("Iterations", 10, 500, 120, 10)
    seed = c3.number_input("Seed", 1, 10_000_000, 999, 1)
    beta = c4.slider("Choice sensitivity β (higher = more rational)", 0.1, 30.0, 4.0, 0.1)

    c5, c6, c7 = st.columns(3)
    alpha = c5.slider("Congestion slope α", 0.0, 10.0, 2.0, 0.1)
    baseA = c6.slider("Base cost A", 0.0, 10.0, 1.0, 0.1)
    baseB = c7.slider("Base cost B", 0.0, 10.0, 1.2, 0.1)

    run = st.button("Run Mean Field simulation", type="primary")
    if run:
        with st.spinner("Simulating..."):
            df = simulate_mean_field_routes(
                N=int(N), T=int(T), alpha=float(alpha), beta=float(beta),
                baseA=float(baseA), baseB=float(baseB),
                seed=int(seed)
            )

        _line(df, "t", ["fracA", "fracB"], "Population split over time")
        _line(df, "t", ["costA", "costB"], "Mean-field costs over time")
        st.metric("Final fracA", f"{df['fracA'].iloc[-1]:.4f}")
        st.metric("Final costA", f"{df['costA'].iloc[-1]:.4f}")
        st.metric("Final costB", f"{df['costB'].iloc[-1]:.4f}")

        with st.expander("Show raw table"):
            st.dataframe(df, use_container_width=True)

        _download_df(df, "mean_field_routes.csv", "Download CSV")


# -------------------------------------------------
# TAB 6: Mechanism Design
# -------------------------------------------------
with tabs[5]:
    st.subheader("Mechanism Design — auctions and incentive compatibility")
    st.markdown(
        """
**What this tab is about:**  
We compare how auction rules shape incentives:
- **First-price vs Second-price (Vickrey)**: Truthful bidding is incentive-compatible in Vickrey.
- **GSP-like ad auction**: ranking by bid×quality, paying next score / own quality.

**Why it matters:**  
This is the core of tokenomics + marketplace design: the rules decide whether players can “game” the system.
        """
    )

    st.markdown("### 6A) Truthfulness sandbox (First-price vs Second-price)")
    c1, c2, c3, c4 = st.columns(4)
    n_bidders = c1.slider("Bidders", 2, 200, 30, 1)
    n_rounds = c2.slider("Rounds", 10, 5000, 800, 50)
    shading = c3.slider("Shading (lie) factor", 0.0, 0.8, 0.2, 0.01)
    seed = c4.number_input("Seed", 1, 10_000_000, 31415, 1)

    c5, c6 = st.columns(2)
    vlow = c5.slider("Value low", 0.1, 10.0, 1.0, 0.1)
    vhigh = c6.slider("Value high", 0.2, 20.0, 2.0, 0.1)

    run1 = st.button("Run auction truthfulness simulation", type="primary")
    if run1:
        with st.spinner("Simulating..."):
            df_round, df_bidder = simulate_auction_truthfulness(
                n_bidders=int(n_bidders),
                n_rounds=int(n_rounds),
                value_low=float(vlow),
                value_high=float(vhigh),
                shading=float(shading),
                seed=int(seed)
            )

        # aggregate winner utility by strategy
        agg = df_round.groupby("strategy").agg(
            fp_utility_mean=("fp_winner_utility","mean"),
            sp_utility_mean=("sp_winner_utility","mean"),
            fp_price_mean=("fp_price","mean"),
            sp_price_mean=("sp_price","mean"),
        ).reset_index()

        st.dataframe(agg.round(6), use_container_width=True)

        if PLOTLY_OK:
            fig = px.bar(agg, x="strategy", y=["fp_utility_mean","sp_utility_mean"], barmode="group",
                         title="Winner utility: truthful vs shaded")
            fig.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Interpretation: in Vickrey, shading typically does not improve expected utility, while in first-price it can.")
        _download_df(df_round, "auction_rounds.csv", "Download rounds CSV")
        _download_df(df_bidder, "auction_bidders.csv", "Download bidders CSV")

    st.divider()

    st.markdown("### 6B) GSP-like ad auction sandbox")
    c1, c2, c3, c4 = st.columns(4)
    n_bidders2 = c1.slider("Advertisers", 2, 200, 25, 1)
    n_slots = c2.slider("Slots", 1, 10, 4, 1)
    n_rounds2 = c3.slider("Rounds", 10, 3000, 600, 50)
    seed2 = c4.number_input("Seed (GSP)", 1, 10_000_000, 27182, 1)

    run2 = st.button("Run GSP simulation", type="primary")
    if run2:
        with st.spinner("Simulating..."):
            df = simulate_gsp_ads(
                n_bidders=int(n_bidders2),
                n_slots=int(n_slots),
                n_rounds=int(n_rounds2),
                seed=int(seed2)
            )

        # revenue and utility by slot
        slot = df.groupby("slot").agg(
            avg_price=("price_per_click","mean"),
            avg_utility=("utility","mean"),
            avg_revenue=("revenue","mean"),
            winners=("bidder","nunique"),
        ).reset_index()

        st.dataframe(slot.round(6), use_container_width=True)

        _line(slot, "slot", ["avg_price","avg_revenue","avg_utility"], "Slot outcomes (averages)")
        _download_df(df, "gsp_ads.csv", "Download CSV")


# =========================
# Footer
# =========================
st.caption(
    "Tip: You can connect these simulators to your DeFi game synthetic datasets later "
    "(e.g., drive demand/values from real segments: whales vs non-whales, risk profiles, resource types)."
)
