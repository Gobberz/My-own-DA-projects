# DeFi Game — Analytics Playground

A modular, multi-layer analytics and simulation platform for a DeFi-based game economy. Built with **Streamlit**, it combines real-time dashboards, tokenomics monitoring, policy simulation, fraud detection, and game-theoretic experiments into a single unified interface.

---

## Project Overview

The playground is designed around one core question: **how do you keep a synthetic in-game economy stable, fair, and resistant to exploitation?**

It ingests CSV data from two configurable directories (core game tables + enriched/synthetic datasets), caches them in memory, and exposes them through a growing set of analytical pages — from basic market health to advanced mechanism design simulations.

---

## Data Architecture

### Input Directories

| Variable | Default Path | Description |
|---|---|---|
| `BASE_MAIN` | `synthetic_defi_game_data/` | Core game tables |
| `BASE_EXTRA` | `synthetic_data/` | Enriched & auxiliary datasets |

Both paths are configurable at runtime via the sidebar — no code changes required.

### Core Tables

| Key | File | Description |
|---|---|---|
| `users` | `users.csv` | Player registry and attributes |
| `sessions` | `sessions.csv` | Session-level activity |
| `market_trades` | `market_trades.csv` | PvP market trade log |
| `token_ledger` | `token_ledger.csv` | Token movements and balances |

### Enriched Tables

| Key | File | Description |
|---|---|---|
| `price_oracle` | `price_oracle_daily.csv` | Daily asset prices |
| `econ_events` | `econ_events.csv` | Economic shock events |
| `resource_production_daily` | `resource_production_daily.csv` | Per-planet resource output |
| `planet_state_daily` | `planet_state_daily.csv` | Planet-level daily snapshot |
| `guild_membership` | `guild_membership.csv` | Alliance/guild rosters |
| `guild_events` | `guild_events.csv` | Alliance activity log |
| `nft_ownership` | `nft_ownership.csv` | NFT holdings by player |
| `nft_usage_events` | `nft_usage_events.csv` | In-game NFT interactions |
| `funnel_events` | `funnel_events.csv` | Player onboarding funnel |
| `marketing_spend` | `marketing_spend.csv` | Channel-level spend data |

Any additional CSVs placed in either directory are automatically discovered and loaded — no configuration needed.

---

## Application Structure

```
app.py                  ← Entry point, data loader, sidebar config
pages/
  ├── Home              ← Economy health summary + active alerts
  ├── Economy Desk      ← Sinks/sources, velocity, net issuance, Sankey flows
  ├── Market Quality    ← Slippage, spread decomposition, liquidity stress index
  ├── Policy Simulator  ← Tax/fee sliders, progressive scales, Pareto optimizer
  ├── Experiment Ops    ← A/B design, guardrails, sequential testing, ship/rollback
  ├── Fraud Lab         ← Wash trading detection, bot scoring, multi-account flags
  ├── Alliances         ← ASI, Shapley payouts, retention curves
  └── Game Theory Lab   ← 6 simulation modules (see below)
```

---

## Analytical Modules

### Economy Desk — Tokenomics Monitor

Tracks the full emission-to-sink loop: Reward → Hold/Stake → Trade → Tax/Fee/Burn.

Key metrics: token velocity, net issuance, sink/source ratio, fee elasticity, 7/30-day token retention rate.

The "Find Leak" tool highlights where tokens are leaving circulation without contributing to sinks.

### Market Quality — Liquidity Desk

Since the player market functions as an exchange (PvP trading), market quality directly affects gameplay experience.

Key metrics: bid-ask spread, slippage by resource and hour, liquidity heatmap, maker concentration, adverse selection proxy during shocks.

### Policy Simulator

Test economic rule changes before deploying them to live players.

Supported interventions: flat/progressive tax rates, transaction fees, emission multipliers, extraction limits.

Output: projected effect on trade volume, revenue, player retention by segment (whales vs. non-whales). Displays a Welfare vs. Revenue Pareto frontier with user-defined constraints.

### Experiment Ops

Structured A/B and bandit testing for all policy changes.

Features: experiment design generator (A/B, multi-armed bandit, switchback), guardrails (churn ceiling, slippage cap), sequential testing for faster decisions, automatic ship/rollback recommendation, stratified results by player segment.

### Fraud Lab

Detects market manipulation patterns before they destabilize the economy.

Detects: wash trading rings, bot activity via session anomalies, multi-account coordination, liquidity spoofing.

### Alliance Analytics

Measures cooperation health and fairness of reward distribution within guilds.

Key metrics: Alliance Stability Index (ASI), Shapley-based payout attribution, member retention curves, contribution Gini coefficient.

### Incident Response

When an alert fires, the platform provides automated drill-down rather than raw data dumps.

Features: alert-to-tab navigation, root cause assistant (top drivers of metric change), action log with lagged impact tracking (1h / 6h / 24h), time-to-detect and time-to-mitigate KPIs, resilience score (recovery time after shock).

---

## Game Theory Lab

Six standalone simulation modules, each tied directly to a game economy mechanic.

| Tab | Model | Game Economy Connection |
|---|---|---|
| P2P Auction Simulator | Double Auction / VCG-lite | In-game resource exchange, slippage reduction |
| Commons & Enforcement | MARL + punishment | Emission control, anti-bot/anti-exploit mechanics |
| Bullwhip Lab | Supply chain information asymmetry | Craft/trade chain speculation, volatility amplification |
| Alliance Treasury Splitter | Shapley Value (cooperative game theory) | Fair reward distribution without PvP |
| Mass Behavior / Herding | Mean Field Games | Policy effects at population scale |
| Mechanism Designer | Incentive compatibility, auction theory | Designing rules where honest play dominates |

Each tab exposes adjustable parameters ("levers"), runs the simulation, and outputs welfare, stability, and abuse-resistance metrics. The results feed directly into the **Incentive Compatibility Score** — a cross-module metric indicating how much the current ruleset incentivizes manipulation.

---

## Getting Started

### Requirements

```
streamlit
pandas
```

Install additional dependencies as modules are added (numpy, scipy, plotly, shap, etc.).

### Run

```bash
streamlit run app.py
```

On first launch, set your data directories in the sidebar and click **Reload data**. The app will auto-discover all CSV files and display core table status on the home page.

---

## Key Design Principles

**Separation of concerns** — data loading is fully decoupled from analytics pages via `st.session_state["TABLES"]`. Any page can access any table without re-reading from disk.

**Fault tolerance** — missing files generate warnings, not crashes. The app is functional with only the four core tables loaded.

**Policy-first thinking** — every analytical view connects back to a specific lever that game designers can adjust. The platform is not just observational; it is built to support decisions.

**Anti-fragility by design** — stress testing and shock simulation are first-class features, not afterthoughts. The goal is to find failure modes in a sandbox before players discover them in production.

---

## Roadmap

- [ ] Connect Game Theory Lab outputs to Policy Simulator as priors
- [ ] Add G1: Policy → Simulation → A/B pipeline (end-to-end)
- [ ] Add G2: Incentive Compatibility Score (cross-module)
- [ ] Add G3: Welfare vs. Revenue Pareto frontier with constraint solver
- [ ] Docker packaging for portable deployment
- [ ] Export: simulation results to CSV / PDF report
